"""Frozen-model prediction, metrics and comparison with the AML baseline.

Held-out evaluation loads the checkpoint from disk and verifies its SHA-256 against the
metadata written at training time, so the scored model is provably the one selected on
internal validation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from . import config as C
from .data import PreparedData, file_sha256
from .train import load_checkpoint

EVAL_SPLITS = ("internal_validation", "heldout_validation")


@torch.no_grad()
def predict(model, X: np.ndarray, data: PreparedData, frame: pd.DataFrame) -> pd.DataFrame:
    pre = data.preprocessor
    out = model(torch.from_numpy(X))
    df = frame[["SOURCE_PARTITION", "ROW_ID", "BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]].copy()
    df = df.rename(columns={"BUILDINGID": "BUILDINGID_TRUE", "FLOOR": "FLOOR_TRUE",
                            "LONGITUDE": "LONGITUDE_TRUE", "LATITUDE": "LATITUDE_TRUE"})
    if "building" in out:
        p = torch.softmax(out["building"], 1).numpy()
        df["BUILDINGID_PRED"] = np.asarray(pre.building_classes)[p.argmax(1)]
        for i, c in enumerate(pre.building_classes):
            df[f"PROBA_BUILDING_{c}"] = p[:, i]
    if "floor" in out:
        p = torch.softmax(out["floor"], 1).numpy()
        df["FLOOR_PRED"] = np.asarray(pre.floor_classes)[p.argmax(1)]
        for i, c in enumerate(pre.floor_classes):
            df[f"PROBA_FLOOR_{c}"] = p[:, i]
    if "coordinate" in out:
        xy = pre.decode_coordinates(out["coordinate"].numpy())
        df["LONGITUDE_PRED"], df["LATITUDE_PRED"] = xy[:, 0], xy[:, 1]
        df["ERROR_LONGITUDE"] = df["LONGITUDE_PRED"] - df["LONGITUDE_TRUE"]
        df["ERROR_LATITUDE"] = df["LATITUDE_PRED"] - df["LATITUDE_TRUE"]
        df["EUCLIDEAN_ERROR"] = np.hypot(df["ERROR_LONGITUDE"], df["ERROR_LATITUDE"])
    return df


def task_metrics(pred: pd.DataFrame, task: str) -> dict:
    if task in ("building", "floor"):
        col = "BUILDINGID" if task == "building" else "FLOOR"
        t, p = pred[f"{col}_TRUE"], pred[f"{col}_PRED"]
        return {"accuracy": accuracy_score(t, p), "macro_f1": f1_score(t, p, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(t, p), "n_correct": int((t == p).sum())}
    e = pred["EUCLIDEAN_ERROR"]
    return {"mae_longitude": float(pred["ERROR_LONGITUDE"].abs().mean()),
            "mae_latitude": float(pred["ERROR_LATITUDE"].abs().mean()),
            "mean_euclidean_error": float(e.mean()), "median_euclidean_error": float(e.median()),
            "p75_euclidean_error": float(e.quantile(0.75)), "p90_euclidean_error": float(e.quantile(0.90)),
            "rmse_euclidean": float(np.sqrt((e ** 2).mean())),
            "coordinate_unit": C.COORDINATE_UNIT_LABEL}


def evaluate_experiment(name: str, data: PreparedData, paths: C.Paths) -> tuple[dict, list[dict]]:
    prefix = C.EXPERIMENT_PREFIX[name]
    ckpt_path = paths.models / f"{prefix}_model.pt"
    meta = json.loads((paths.models / f"{prefix}_model.json").read_text())
    sha = file_sha256(ckpt_path)
    if sha != meta["checkpoint_sha256"]:
        raise RuntimeError(f"{ckpt_path.name} changed after training; refusing held-out evaluation")
    model, ckpt = load_checkpoint(ckpt_path)
    tasks = tuple(ckpt["config"]["tasks"])
    preds, rows = {}, []
    for split in EVAL_SPLITS:
        df = predict(model, data.X[split], data, data.frames[split])
        df.to_csv(paths.tables / f"{prefix}_{split.replace('_validation', '')}_predictions.csv", index=False)
        preds[split] = df
        for t in tasks:
            rows.append({"model": f"DL {name}", "experiment": name, "split": split, "task": t,
                         "n_rows": len(df), "selected_epoch": meta["selected_epoch"],
                         "checkpoint_sha256": sha[:12], **task_metrics(df, t)})
    return preds, rows


# ------------------------------------------------------------------ AML comparison
def load_aml_heldout(paths: C.Paths) -> pd.DataFrame | None:
    p = paths.tables / "ml_05_aml_heldout_integrated_predictions.csv"
    return pd.read_csv(p) if p.is_file() else None


AML_PREDICTIONS = {"internal_validation": "ml_05_aml_internal_validation_integrated_predictions.csv",
                   "heldout_validation": "ml_05_aml_heldout_integrated_predictions.csv"}


def aml_metric_rows(paths: C.Paths) -> list[dict]:
    """AML metrics recomputed from ML_05 row-level predictions with the same functions as DL."""
    rows = []
    for split, fname in AML_PREDICTIONS.items():
        p = paths.tables / fname
        if not p.is_file():
            continue
        aml = pd.read_csv(p)
        aml["ERROR_LONGITUDE"] = aml.LONGITUDE_PRED - aml.LONGITUDE_TRUE
        aml["ERROR_LATITUDE"] = aml.LATITUDE_PRED - aml.LATITUDE_TRUE
        aml["EUCLIDEAN_ERROR"] = np.hypot(aml.ERROR_LONGITUDE, aml.ERROR_LATITUDE)
        for t in C.TASKS:
            rows.append({"model": "AML Random Forest", "experiment": "aml_baseline", "split": split,
                         "task": t, "n_rows": len(aml), **task_metrics(aml, t)})
    return rows


def recompute_aml_heldout(aml: pd.DataFrame) -> dict:
    """Recompute AML held-out metrics from its row-level predictions (consistency check)."""
    return {
        "building_accuracy": accuracy_score(aml.BUILDINGID_TRUE, aml.BUILDINGID_PRED),
        "floor_accuracy": accuracy_score(aml.FLOOR_TRUE, aml.FLOOR_PRED),
        "mean_euclidean_error": float(np.hypot(aml.LONGITUDE_PRED - aml.LONGITUDE_TRUE,
                                               aml.LATITUDE_PRED - aml.LATITUDE_TRUE).mean()),
        "joint_building_floor_correct": int(((aml.BUILDINGID_TRUE == aml.BUILDINGID_PRED)
                                             & (aml.FLOOR_TRUE == aml.FLOOR_PRED)).sum()),
    }


def paired_bootstrap(dl: pd.DataFrame, aml: pd.DataFrame, model: str, n_boot: int = 2000,
                     seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap of DL minus AML on identical held-out rows (descriptive uncertainty).

    Only the tasks present in ``dl`` are compared. Positive accuracy differences favour DL;
    negative error differences favour DL.
    """
    m = dl.merge(aml, on="ROW_ID", suffixes=("_DL", "_AML"), validate="one_to_one")
    assert len(m) == len(dl) == len(aml), "held-out rows differ between DL and AML"
    stats = {}
    if "BUILDINGID_PRED" in dl:
        stats["building_accuracy"] = ((m.BUILDINGID_PRED_DL == m.BUILDINGID_TRUE_DL).to_numpy(float),
                                      (m.BUILDINGID_PRED_AML == m.BUILDINGID_TRUE_AML).to_numpy(float))
    if "FLOOR_PRED" in dl:
        stats["floor_accuracy"] = ((m.FLOOR_PRED_DL == m.FLOOR_TRUE_DL).to_numpy(float),
                                   (m.FLOOR_PRED_AML == m.FLOOR_TRUE_AML).to_numpy(float))
    if "EUCLIDEAN_ERROR" in dl:
        stats["mean_euclidean_error"] = (m.EUCLIDEAN_ERROR_DL.to_numpy(),
                                         np.hypot(m.LONGITUDE_PRED_AML - m.LONGITUDE_TRUE_AML,
                                                  m.LATITUDE_PRED_AML - m.LATITUDE_TRUE_AML).to_numpy())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(n_boot, len(m)))
    rows = []
    for k, (a, b) in stats.items():
        diff = a - b
        boots = diff[idx].mean(1)
        rows.append({"model": model, "metric": k, "dl": a.mean(), "aml": b.mean(), "dl_minus_aml": diff.mean(),
                     "ci95_low": np.quantile(boots, 0.025), "ci95_high": np.quantile(boots, 0.975),
                     "n_rows": len(m), "n_bootstrap": n_boot})
    return pd.DataFrame(rows)


def hierarchy_breakdown(pred: pd.DataFrame, split: str, model: str) -> pd.DataFrame:
    b = pred.BUILDINGID_TRUE == pred.BUILDINGID_PRED
    f = pred.FLOOR_TRUE == pred.FLOOR_PRED
    outcome = np.select([b & f, b & ~f, ~b & f], ["both_correct", "building_correct_floor_wrong",
                                                  "building_wrong_floor_correct"], default="both_wrong")
    rows = []
    for o in ["both_correct", "building_correct_floor_wrong", "building_wrong_floor_correct", "both_wrong"]:
        sel = outcome == o
        e = pred.loc[sel, "EUCLIDEAN_ERROR"] if "EUCLIDEAN_ERROR" in pred else pd.Series(dtype=float)
        rows.append({"model": model, "split": split, "hierarchy_outcome": o, "n_rows": int(sel.sum()),
                     "proportion": float(sel.mean()),
                     "mean_coordinate_error": float(e.mean()) if len(e) else np.nan,
                     "median_coordinate_error": float(e.median()) if len(e) else np.nan})
    return pd.DataFrame(rows)


def per_building_coordinate_error(pred: pd.DataFrame, model: str) -> pd.DataFrame:
    g = pred.groupby("BUILDINGID_TRUE")["EUCLIDEAN_ERROR"]
    out = g.agg(["count", "mean", "median"]).reset_index()
    out.insert(0, "model", model)
    return out.rename(columns={"BUILDINGID_TRUE": "building", "count": "n_rows",
                               "mean": "mean_euclidean_error", "median": "median_euclidean_error"})


def per_class_report(pred: pd.DataFrame, task: str, model: str) -> pd.DataFrame:
    col = "BUILDINGID" if task == "building" else "FLOOR"
    rows = []
    for c in sorted(pred[f"{col}_TRUE"].unique()):
        t, p = pred[f"{col}_TRUE"] == c, pred[f"{col}_PRED"] == c
        tp = int((t & p).sum())
        prec = tp / max(int(p.sum()), 1)
        rec = tp / max(int(t.sum()), 1)
        rows.append({"model": model, "task": task, "class": int(c), "support": int(t.sum()),
                     "precision": prec, "recall": rec,
                     "f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)})
    return pd.DataFrame(rows)
