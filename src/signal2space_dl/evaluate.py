"""Metrics and error analysis shared by the DL notebooks and the AML comparison.

This module only needs NumPy, pandas and scikit-learn (no PyTorch), so it can be
tested independently. Column names follow the AML integrated prediction tables
(``ml_05_aml_*``) so that DL and AML outputs can be joined on
``(SOURCE_PARTITION, ROW_ID)`` and compared row by row.

Coordinate errors are in the dataset's own coordinate units; see
``config.COORDINATE_UNIT_LABEL``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import config as C

HIERARCHY_ORDER = ["both_correct", "building_correct_floor_wrong", "building_wrong_floor_correct", "both_wrong"]


# --------------------------------------------------------------------------
# Basic metrics
# --------------------------------------------------------------------------


def classification_metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def euclidean_errors(true_xy, pred_xy) -> np.ndarray:
    d = np.asarray(true_xy, dtype=np.float64) - np.asarray(pred_xy, dtype=np.float64)
    return np.sqrt((d ** 2).sum(axis=1))


def coordinate_metrics(true_xy, pred_xy) -> dict:
    true_xy, pred_xy = np.asarray(true_xy, float), np.asarray(pred_xy, float)
    e = euclidean_errors(true_xy, pred_xy)
    return {"mae_longitude": float(np.abs(true_xy[:, 0] - pred_xy[:, 0]).mean()),
            "mae_latitude": float(np.abs(true_xy[:, 1] - pred_xy[:, 1]).mean()),
            "mean_euclidean_error": float(e.mean()),
            "median_euclidean_error": float(np.median(e)),
            "p75_euclidean_error": float(np.percentile(e, 75)),
            "p95_euclidean_error": float(np.percentile(e, 95))}


def confusion_frame(y_true, y_pred, labels) -> pd.DataFrame:
    """Confusion matrix with rows = actual, columns = predicted (AML layout)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    frame = pd.DataFrame(cm, index=pd.Index(labels, name="actual"), columns=[str(l) for l in labels])
    return frame


# --------------------------------------------------------------------------
# Prediction table
# --------------------------------------------------------------------------


def build_prediction_frame(meta: pd.DataFrame, building_true, floor_true, coords_true, pred: dict,
                           building_classes, floor_classes) -> pd.DataFrame:
    """Assemble an AML-compatible prediction table for whichever heads exist in ``pred``."""
    out = meta[["ROW_ID", "SOURCE_PARTITION", "USERID", "PHONEID"]].reset_index(drop=True).copy()
    out["BUILDINGID_TRUE"] = np.asarray(building_true)
    out["FLOOR_TRUE"] = np.asarray(floor_true)
    out["LONGITUDE_TRUE"] = np.asarray(coords_true)[:, 0]
    out["LATITUDE_TRUE"] = np.asarray(coords_true)[:, 1]
    if "building" in pred:
        out["BUILDINGID_PRED"] = pred["building"]
        for i, c in enumerate(building_classes):
            out[f"PROBA_BUILDING_{c}"] = pred["building_proba"][:, i]
        out["BUILDING_CORRECT"] = out.BUILDINGID_PRED == out.BUILDINGID_TRUE
    if "floor" in pred:
        out["FLOOR_PRED"] = pred["floor"]
        for i, c in enumerate(floor_classes):
            out[f"PROBA_FLOOR_{c}"] = pred["floor_proba"][:, i]
        out["FLOOR_CORRECT"] = out.FLOOR_PRED == out.FLOOR_TRUE
    if "coords" in pred:
        out["LONGITUDE_PRED"] = pred["coords"][:, 0]
        out["LATITUDE_PRED"] = pred["coords"][:, 1]
        out["ERROR_LONGITUDE"] = (out.LONGITUDE_TRUE - out.LONGITUDE_PRED).abs()
        out["ERROR_LATITUDE"] = (out.LATITUDE_TRUE - out.LATITUDE_PRED).abs()
        out["EUCLIDEAN_ERROR"] = euclidean_errors(out[["LONGITUDE_TRUE", "LATITUDE_TRUE"]],
                                                  out[["LONGITUDE_PRED", "LATITUDE_PRED"]])
        out["COORDINATE_UNIT"] = C.COORDINATE_UNIT_LABEL
    if "building" in pred and "floor" in pred:
        out["BUILDING_FLOOR_CORRECT"] = out.BUILDING_CORRECT & out.FLOOR_CORRECT
        out["HIERARCHY_OUTCOME"] = hierarchy_outcome(out.BUILDING_CORRECT, out.FLOOR_CORRECT)
    return out


def hierarchy_outcome(building_correct, floor_correct) -> np.ndarray:
    b = np.asarray(building_correct, bool)
    f = np.asarray(floor_correct, bool)
    return np.select([b & f, b & ~f, ~b & f], HIERARCHY_ORDER[:3], default=HIERARCHY_ORDER[3])


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def summary_rows(pred_frame: pd.DataFrame, split: str, model_name: str) -> list[dict]:
    """One row per task in the AML ``ml_05_aml_summary_metrics.csv`` layout."""
    rows = []
    base = {"split": split, "model": model_name, "n_rows": len(pred_frame)}
    if "BUILDINGID_PRED" in pred_frame:
        rows.append({**base, "task": "building",
                     **classification_metrics(pred_frame.BUILDINGID_TRUE, pred_frame.BUILDINGID_PRED)})
    if "FLOOR_PRED" in pred_frame:
        rows.append({**base, "task": "floor",
                     **classification_metrics(pred_frame.FLOOR_TRUE, pred_frame.FLOOR_PRED)})
    if "LONGITUDE_PRED" in pred_frame:
        rows.append({**base, "task": "coordinate", "coordinate_unit": C.COORDINATE_UNIT_LABEL,
                     **coordinate_metrics(pred_frame[["LONGITUDE_TRUE", "LATITUDE_TRUE"]],
                                          pred_frame[["LONGITUDE_PRED", "LATITUDE_PRED"]])})
    if "BUILDING_FLOOR_CORRECT" in pred_frame:
        rows.append({**base, "task": "building_and_floor",
                     "accuracy": float(pred_frame.BUILDING_FLOOR_CORRECT.mean())})
    return rows


def hierarchical_breakdown(pred_frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows / proportion / coordinate error for each building-floor correctness outcome."""
    rows = []
    for outcome in HIERARCHY_ORDER:
        sub = pred_frame[pred_frame.HIERARCHY_OUTCOME == outcome]
        has_err = "EUCLIDEAN_ERROR" in sub and len(sub)
        rows.append({"split": split, "hierarchy_outcome": outcome, "n_rows": len(sub),
                     "proportion": len(sub) / len(pred_frame),
                     "mean_coordinate_error": float(sub.EUCLIDEAN_ERROR.mean()) if has_err else np.nan,
                     "median_coordinate_error": float(sub.EUCLIDEAN_ERROR.median()) if has_err else np.nan,
                     "coordinate_unit": C.COORDINATE_UNIT_LABEL})
    return pd.DataFrame(rows)


def hierarchical_by_building_floor(pred_frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Per true (building, floor) cell: accuracies and coordinate error."""
    rows = []
    for (b, f), sub in pred_frame.groupby(["BUILDINGID_TRUE", "FLOOR_TRUE"]):
        bc = sub.BUILDING_CORRECT
        rows.append({"split": split, "building_true": b, "floor_true": f, "n_rows": len(sub),
                     "building_accuracy": float(bc.mean()),
                     "floor_accuracy": float(sub.FLOOR_CORRECT.mean()),
                     "joint_building_floor_accuracy": float(sub.BUILDING_FLOOR_CORRECT.mean()),
                     "conditional_floor_accuracy_given_building_correct":
                         float(sub.FLOOR_CORRECT[bc].mean()) if bc.any() else np.nan,
                     "mean_coordinate_error": float(sub.EUCLIDEAN_ERROR.mean()),
                     "median_coordinate_error": float(sub.EUCLIDEAN_ERROR.median())})
    return pd.DataFrame(rows)


def floor_error_distance(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """How many floors away wrong floor predictions are (building-correct rows only)."""
    sub = pred_frame[pred_frame.BUILDING_CORRECT]
    gap = (sub.FLOOR_PRED - sub.FLOOR_TRUE).abs()
    return gap.value_counts().sort_index().rename_axis("floor_gap").reset_index(name="n_rows")


def robustness_by_group(pred_frame: pd.DataFrame, group_col: str, split: str,
                        seen_in_training: set | None = None, min_rows: int = 1) -> pd.DataFrame:
    """Post-hoc metrics per USERID or PHONEID. Identifiers are never model inputs."""
    rows = []
    for g, sub in pred_frame.groupby(group_col):
        if len(sub) < min_rows:
            continue
        row = {"split": split, "group_column": group_col, "group": g, "n_rows": len(sub)}
        if seen_in_training is not None:
            row["group_seen_in_internal_train"] = g in seen_in_training
        if "BUILDING_CORRECT" in sub:
            row["building_accuracy"] = float(sub.BUILDING_CORRECT.mean())
        if "FLOOR_CORRECT" in sub:
            row["floor_accuracy"] = float(sub.FLOOR_CORRECT.mean())
        if "EUCLIDEAN_ERROR" in sub:
            row["mean_euclidean_error"] = float(sub.EUCLIDEAN_ERROR.mean())
            row["median_euclidean_error"] = float(sub.EUCLIDEAN_ERROR.median())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# AML vs DL
# --------------------------------------------------------------------------


def load_aml_summary(path=C.AML_SUMMARY_PATH) -> pd.DataFrame:
    """AML summary metrics with harmonised column names (unit relabelled as dataset units)."""
    aml = pd.read_csv(path).rename(columns={"model_config": "model"})
    aml["model"] = "AML Random Forest (" + aml["task"] + ")"
    aml.loc[aml.task == "coordinate", "coordinate_unit"] = C.COORDINATE_UNIT_LABEL
    return aml


def paired_comparison(dl: pd.DataFrame, aml: pd.DataFrame, n_boot: int = 2000, seed: int = C.GLOBAL_SEED) -> dict:
    """Row-aligned comparison of DL vs AML predictions on the same split.

    Returns discordant counts for building/floor correctness (inputs for a
    McNemar test), an exact two-sided McNemar p-value, and a bootstrap 95%
    interval for the mean paired difference in Euclidean error (DL - AML).
    """
    from scipy.stats import binomtest

    key = list(C.KEY_COLUMNS)
    m = dl.merge(aml, on=key, suffixes=("_dl", "_aml"), validate="one_to_one")
    assert len(m) == len(dl) == len(aml), "DL and AML predictions are not row-aligned"
    for col in ["BUILDINGID_TRUE", "FLOOR_TRUE"]:
        assert (m[f"{col}_dl"] == m[f"{col}_aml"]).all(), f"{col} differs between DL and AML rows"
    out = {"n_rows": len(m)}
    for task, col in [("building", "BUILDING_CORRECT"), ("floor", "FLOOR_CORRECT"),
                      ("building_and_floor", "BUILDING_FLOOR_CORRECT")]:
        d, a = m[f"{col}_dl"].astype(bool), m[f"{col}_aml"].astype(bool)
        b01, b10 = int((d & ~a).sum()), int((~d & a).sum())
        p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        out[task] = {"dl_correct_aml_wrong": b01, "dl_wrong_aml_correct": b10, "mcnemar_exact_p": float(p)}
    diff = (m.EUCLIDEAN_ERROR_dl - m.EUCLIDEAN_ERROR_aml).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(diff, len(diff)).mean() for _ in range(n_boot)])
    out["coordinate"] = {"mean_diff_dl_minus_aml": float(diff.mean()),
                         "ci95_low": float(np.percentile(boots, 2.5)),
                         "ci95_high": float(np.percentile(boots, 97.5)),
                         "share_rows_dl_closer": float((diff < 0).mean()),
                         "unit": C.COORDINATE_UNIT_LABEL}
    return out
