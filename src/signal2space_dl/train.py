"""Seeded training with validation-based early stopping.

``train_experiment`` receives internal-train and internal-validation arrays only; it has
no access to the held-out partition. The checkpoint it writes is the frozen model that
``evaluate`` later scores on held-out data.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .data import PreparedData, file_sha256
from .model import MultitaskLoss, MultitaskMLP, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def select_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tensors(X: np.ndarray, y: dict, tasks) -> list[torch.Tensor]:
    out = [torch.from_numpy(X)]
    for t in tasks:
        out.append(torch.from_numpy(y[t]))
    return out


def _class_weights(y_train: dict, tasks, n_classes: dict) -> dict[str, torch.Tensor]:
    """Inverse-frequency ('balanced') weights from internal-train labels only."""
    w = {}
    for t in tasks:
        if t in ("building", "floor"):
            counts = np.bincount(y_train[t], minlength=n_classes[t]).astype(np.float64)
            w[t] = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32)
    return w


@torch.no_grad()
def _evaluate_loss(model, loader, loss_fn, tasks, device) -> tuple[float, dict, dict]:
    model.eval()
    total, parts_sum, n = 0.0, {t: 0.0 for t in tasks}, 0
    correct = {t: 0 for t in tasks if t != "coordinate"}
    coord_err = 0.0
    for batch in loader:
        xb = batch[0].to(device)
        targets = {t: b.to(device) for t, b in zip(tasks, batch[1:])}
        out = model(xb)
        loss, parts = loss_fn(out, targets)
        bs = xb.shape[0]
        total += float(loss) * bs
        n += bs
        for t in tasks:
            parts_sum[t] += parts[t] * bs
            if t == "coordinate":
                coord_err += float(torch.linalg.vector_norm(out[t] - targets[t], dim=1).sum())
            else:
                correct[t] += int((out[t].argmax(1) == targets[t]).sum())
    metrics = {f"{t}_accuracy": correct[t] / n for t in correct}
    if "coordinate" in tasks:
        metrics["coordinate_mean_error_scaled"] = coord_err / n
    return total / n, {t: v / n for t, v in parts_sum.items()}, metrics


def train_experiment(name: str, data: PreparedData, paths: C.Paths, cfg: C.TrainConfig,
                     device: str = "auto", verbose: bool = True) -> dict:
    """Train one experiment, select the best epoch on internal validation, save checkpoint."""
    tasks = tuple(cfg.tasks)
    set_seed(cfg.seed)
    dev = select_device(device)
    pre = data.preprocessor
    n_classes = {"building": len(pre.building_classes), "floor": len(pre.floor_classes)}

    Xtr, ytr = data.X["internal_train"], data.y["internal_train"]
    Xva, yva = data.X["internal_validation"], data.y["internal_validation"]
    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(TensorDataset(*_tensors(Xtr, ytr, tasks)), batch_size=cfg.batch_size,
                              shuffle=True, generator=gen, drop_last=False)
    val_loader = DataLoader(TensorDataset(*_tensors(Xva, yva, tasks)), batch_size=1024, shuffle=False)

    model = MultitaskMLP(Xtr.shape[1], n_classes["building"], n_classes["floor"], tasks,
                         cfg.hidden_sizes, cfg.head_hidden, cfg.input_dropout, cfg.dropout).to(dev)
    cw = _class_weights(ytr, tasks, n_classes) if cfg.class_weighting else None
    loss_fn = MultitaskLoss(tasks, cfg.loss_weights, cfg.huber_delta,
                            {k: v.to(dev) for k, v in cw.items()} if cw else None).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.lr_factor,
                                                       patience=cfg.lr_patience)

    history, best_state, best_val, best_epoch, stale = [], None, float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        run, nrows = 0.0, 0
        for batch in train_loader:
            xb = batch[0].to(dev)
            targets = {t: b.to(dev) for t, b in zip(tasks, batch[1:])}
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_fn(model(xb), targets)
            loss.backward()
            opt.step()
            run += loss.item() * xb.shape[0]
            nrows += xb.shape[0]
        val_loss, val_parts, val_metrics = _evaluate_loss(model, val_loader, loss_fn, tasks, dev)
        sched.step(val_loss)
        row = {"epoch": epoch, "train_loss": run / nrows, "val_loss": val_loss,
               "lr": opt.param_groups[0]["lr"],
               **{f"val_loss_{t}": v for t, v in val_parts.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if val_loss < best_val - cfg.min_delta:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"[{name}] epoch {epoch:3d} train {row['train_loss']:.4f} val {val_loss:.4f} "
                  f"best@{best_epoch} lr {row['lr']:.1e}")
        if stale >= cfg.patience:
            break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    prefix = C.EXPERIMENT_PREFIX[name]
    paths.ensure()
    hist = pd.DataFrame(history)
    hist["selected_epoch"] = hist["epoch"] == best_epoch
    hist.to_csv(paths.tables / f"{prefix}_training_history.csv", index=False)

    ckpt_path = paths.models / f"{prefix}_model.pt"
    torch.save({
        "experiment": name,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": cfg.to_dict(),
        "n_inputs": int(Xtr.shape[1]),
        "n_classes": n_classes,
        "preprocessor": pre.to_dict(),
    }, ckpt_path)
    meta = {
        "experiment": name,
        "tasks": list(tasks),
        "checkpoint": str(ckpt_path.relative_to(paths.root)),
        "checkpoint_sha256": file_sha256(ckpt_path),
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "best_internal_validation_loss": best_val,
        "selection_rule": "minimum internal-validation weighted loss; early stopping patience "
                          f"{cfg.patience}; held-out partition not used",
        "n_parameters": count_parameters(model),
        "training_seconds": round(elapsed, 1),
        "device": str(dev),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "train_rows": int(len(Xtr)),
        "validation_rows": int(len(Xva)),
        "config": cfg.to_dict(),
    }
    (paths.models / f"{prefix}_model.json").write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f"[{name}] selected epoch {best_epoch}/{len(history)}; val loss {best_val:.4f}; "
              f"{meta['n_parameters']:,} params; {elapsed:.0f}s on {dev}")
    return meta


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[MultitaskMLP, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MultitaskMLP(ckpt["n_inputs"], ckpt["n_classes"]["building"], ckpt["n_classes"]["floor"],
                         tuple(cfg["tasks"]), tuple(cfg["hidden_sizes"]), cfg["head_hidden"],
                         cfg["input_dropout"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
