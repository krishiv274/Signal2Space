"""Training, early stopping and internal-validation model selection.

Selection discipline
--------------------
* Weights are fitted on ``internal_train`` only.
* Early stopping and the choice between candidate configurations use
  ``internal_validation`` only, through :func:`selection_score`.
* The supplied held-out file is not loaded by anything in this module.

Selection score (lower is better)
---------------------------------
For the multitask model::

    score = mean(building_error_rate, floor_error_rate, mean_euclidean_error / D)

where ``D`` is the mean Euclidean error of predicting the internal-train
coordinate centroid for every internal-validation row (see
``data.centroid_baseline_error``). Dividing by ``D`` puts coordinate error on
roughly the same 0-1 scale as the error rates. Single-task runs use only the
term(s) for their enabled head(s).
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .data import Preprocessor, SplitArrays
from .model import MultiTaskNet, count_parameters, multitask_loss


@dataclass
class TrainResult:
    config: C.TrainConfig
    model: MultiTaskNet
    history: pd.DataFrame
    best_epoch: int
    best_score: float
    seconds: float
    n_parameters: int
    extra: dict = field(default_factory=dict)


def _tensors(arrays: SplitArrays):
    return (torch.from_numpy(arrays.x), torch.from_numpy(arrays.building),
            torch.from_numpy(arrays.floor), torch.from_numpy(arrays.coords_scaled))


@torch.no_grad()
def predict(model: MultiTaskNet, arrays: SplitArrays, prep: Preprocessor, batch_size: int = 2048) -> dict:
    """Return class indices, probabilities and unscaled coordinates for enabled heads."""
    model.eval()
    x = torch.from_numpy(arrays.x)
    outs: dict[str, list] = {t: [] for t in model.tasks}
    for start in range(0, len(x), batch_size):
        o = model(x[start:start + batch_size])
        for t in model.tasks:
            outs[t].append(o[t])
    result: dict = {}
    if "building" in model.tasks:
        p = torch.softmax(torch.cat(outs["building"]), dim=1).numpy()
        result["building_proba"] = p
        result["building_idx"] = p.argmax(axis=1)
        result["building"] = np.asarray(prep.building_classes)[result["building_idx"]]
    if "floor" in model.tasks:
        p = torch.softmax(torch.cat(outs["floor"]), dim=1).numpy()
        result["floor_proba"] = p
        result["floor_idx"] = p.argmax(axis=1)
        result["floor"] = np.asarray(prep.floor_classes)[result["floor_idx"]]
    if "coord" in model.tasks:
        result["coords"] = prep.unscale_coords(torch.cat(outs["coord"]).numpy())
    return result


def selection_score(pred: dict, arrays: SplitArrays, prep: Preprocessor, coord_norm: float) -> dict:
    """Validation metrics and the composite selection score (lower is better)."""
    terms, metrics = [], {}
    if "building_idx" in pred:
        acc = float((pred["building_idx"] == arrays.building).mean())
        metrics["val_building_acc"] = acc
        terms.append(1.0 - acc)
    if "floor_idx" in pred:
        acc = float((pred["floor_idx"] == arrays.floor).mean())
        metrics["val_floor_acc"] = acc
        terms.append(1.0 - acc)
    if "coords" in pred:
        err = np.sqrt(((pred["coords"] - arrays.coords) ** 2).sum(axis=1)).mean()
        metrics["val_mean_euclidean"] = float(err)
        terms.append(float(err) / coord_norm)
    metrics["val_selection_score"] = float(np.mean(terms))
    return metrics


def train_model(cfg: C.TrainConfig, train: SplitArrays, ival: SplitArrays, prep: Preprocessor,
                coord_norm: float, verbose: bool = False) -> TrainResult:
    """Train one configuration with early stopping on the internal-validation score."""
    C.set_global_seed(cfg.seed)
    model = MultiTaskNet(train.x.shape[1], prep.n_buildings, prep.n_floors, cfg.hidden_sizes,
                         cfg.head_hidden, cfg.dropout, cfg.input_dropout, cfg.tasks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(TensorDataset(*_tensors(train)), batch_size=cfg.batch_size, shuffle=True,
                        generator=generator,
                        # BatchNorm cannot train on a final batch of one row.
                        drop_last=len(train) % cfg.batch_size == 1)

    best_state, best_score, best_epoch, stale = None, float("inf"), -1, 0
    rows, t0 = [], time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        sums, n = {}, 0
        for xb, bb, fb, cb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss, parts = multitask_loss(outputs, bb, fb, cb, cfg.w_building, cfg.w_floor, cfg.w_coord,
                                         cfg.coord_beta)
            loss.backward()
            optimizer.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v * len(xb)
            n += len(xb)
        train_parts = {f"train_{k}": v / n for k, v in sums.items()}
        val = selection_score(predict(model, ival, prep), ival, prep, coord_norm)
        scheduler.step(val["val_selection_score"])
        rows.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **train_parts, **val})
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"[{cfg.name}] epoch {epoch:3d} score={val['val_selection_score']:.4f}")
        if val["val_selection_score"] < best_score - 1e-6:
            best_score, best_epoch, stale = val["val_selection_score"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    history = pd.DataFrame(rows)
    history.insert(0, "config", cfg.name)
    return TrainResult(cfg, model, history, best_epoch, best_score, time.time() - t0, count_parameters(model))


def select_configuration(grid, train: SplitArrays, ival: SplitArrays, prep: Preprocessor,
                         coord_norm: float, verbose: bool = True):
    """Train every candidate; return (best TrainResult, selection table, all histories)."""
    results, rows = [], []
    for cfg in grid:
        res = train_model(cfg, train, ival, prep, coord_norm)
        best = res.history.loc[res.history.epoch == res.best_epoch].iloc[0]
        rows.append({"config": cfg.name, **{k: v for k, v in cfg.to_dict().items() if k != "name"},
                     "best_epoch": res.best_epoch, "epochs_run": int(res.history.epoch.max()),
                     "val_selection_score": res.best_score,
                     "val_building_acc": best.get("val_building_acc", np.nan),
                     "val_floor_acc": best.get("val_floor_acc", np.nan),
                     "val_mean_euclidean": best.get("val_mean_euclidean", np.nan),
                     "n_parameters": res.n_parameters, "train_seconds": round(res.seconds, 1)})
        if verbose:
            print(f"{cfg.name:28s} score={res.best_score:.4f} best_epoch={res.best_epoch} ({res.seconds:.0f}s)")
        results.append(res)
    table = pd.DataFrame(rows).sort_values("val_selection_score", kind="stable").reset_index(drop=True)
    table["selected"] = table.index == 0
    best = next(r for r in results if r.config.name == table.loc[0, "config"])
    histories = pd.concat([r.history for r in results], ignore_index=True)
    return best, table, histories


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_model(result: TrainResult, prep: Preprocessor, path: Path) -> None:
    """Save weights + config + preprocessing so predictions can be reproduced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result.model.state_dict(), "config": result.config.to_dict(),
                "preprocessor": prep.to_dict(), "best_epoch": result.best_epoch,
                "best_selection_score": result.best_score, "n_parameters": result.n_parameters}, path)


def load_model(path: Path) -> tuple[MultiTaskNet, C.TrainConfig, Preprocessor]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfgd = dict(ckpt["config"])
    cfgd["hidden_sizes"] = tuple(cfgd["hidden_sizes"])
    cfgd["tasks"] = tuple(cfgd["tasks"])
    cfg = C.TrainConfig(**cfgd)
    pd_ = ckpt["preprocessor"]
    prep = Preprocessor(np.asarray(pd_["summary_mean"]), np.asarray(pd_["summary_std"]),
                        np.asarray(pd_["coord_mean"]), np.asarray(pd_["coord_std"]),
                        pd_["building_classes"], pd_["floor_classes"], pd_["fit_rows"], pd_["fit_split"])
    model = MultiTaskNet(len(C.FEATURE_COLUMNS), prep.n_buildings, prep.n_floors, cfg.hidden_sizes,
                         cfg.head_hidden, cfg.dropout, cfg.input_dropout, cfg.tasks)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, prep


def write_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
