"""Data and model logic for the Signal2Space Streamlit app (no Streamlit imports).

The app scores Wi-Fi scans live with the saved Phase 2 networks. To score a scan
that is not in the saved feature tables (an uploaded scan, or a held-out scan
with access points hidden), it rebuilds the Phase 1 feature transform from the
raw training file exactly as ``ML_01_data_preparation_features.ipynb`` does:

* the same grouped internal split (GroupShuffleSplit, test size 0.20, seed 42);
* per-WAP mean and population standard deviation of *detected* readings on
  internal-training rows only, with the same global fallback for WAPs never
  detected there;
* undetected readings (raw value 100) filled one unit below each WAP's lowest
  standardised training reading;
* 520 presence indicators and the same six signal summaries.

``tests/test_core.py`` checks that this transform reproduces the stored
held-out feature table and that live predictions match the saved DL results.
Random Forest (AML) models were not saved as files, so AML results come from
the saved prediction tables and exist only for unmodified held-out scans.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from signal2space_dl import config as C  # noqa: E402

MISSING = C.MISSING_SIGNAL
AP = C.AP_COLUMNS
UNIT = "dataset units"
MODEL_COLORS = {"True position": "#243A50", "DL multitask": "#2A78D6", "DL coordinate network": "#1BAF7A",
                "AML Random Forest": "#EB6834"}


# --------------------------------------------------------------------------
# Raw data and the Phase 1 feature transform
# --------------------------------------------------------------------------


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw training and held-out files with ROW_ID added (as in ML_01)."""
    out = []
    for path in (C.RAW_TRAIN_PATH, C.RAW_HELDOUT_PATH):
        f = pd.read_csv(path)
        out.append(pd.concat([pd.Series(np.arange(len(f), dtype=np.int64), name="ROW_ID"), f], axis=1))
    return out[0], out[1]


def internal_train_rows(train: pd.DataFrame) -> pd.DataFrame:
    """Rebuild ML_01's grouped split and return the internal-training rows."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = train.BUILDINGID.astype(str) + "_" + train.FLOOR.astype(str) + "_" + train.SPACEID.astype(str)
    tr_pos, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(train, groups=groups))
    return train.iloc[tr_pos]


@dataclass
class RssiScaler:
    means: np.ndarray
    scales: np.ndarray
    impute: np.ndarray


def fit_rssi_scaler(internal_train: pd.DataFrame) -> RssiScaler:
    """Same arithmetic as ML_01 ``fit_rssi_scaler`` (internal-training rows only)."""
    raw = internal_train[AP].to_numpy(dtype=np.float64)
    seen = raw != MISSING
    allv = raw[seen]
    g_mean, g_std = allv.mean(), allv.std(ddof=0)
    g_scale = g_std if g_std > 0 else 1.0
    g_impute = ((allv - g_mean) / g_scale).min() - 1.0
    means = np.full(len(AP), g_mean)
    scales = np.full(len(AP), g_scale)
    impute = np.full(len(AP), g_impute)
    for j in range(len(AP)):
        v = raw[seen[:, j], j]
        if v.size:
            means[j] = v.mean()
            sd = v.std(ddof=0)
            scales[j] = sd if sd > 0 else 1.0
            impute[j] = ((v - means[j]) / scales[j]).min() - 1.0
    return RssiScaler(means, scales, impute)


def features_from_raw(raw: np.ndarray, scaler: RssiScaler) -> pd.DataFrame:
    """1,046 AML predictors for raw RSSI rows (shape n x 520, 100 = not detected)."""
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    seen = raw != MISSING
    z = (raw - scaler.means) / scaler.scales
    z = np.where(seen, z, scaler.impute)
    n_heard = seen.sum(axis=1)
    has = n_heard > 0
    obs = np.where(seen, raw, np.nan)
    with np.errstate(all="ignore"):
        strongest = np.where(has, np.nanmax(np.where(seen, raw, -np.inf), axis=1), -110.0)
        mean = np.where(has, np.nanmean(obs, axis=1), -110.0)
        std = np.where(has, np.nanstd(obs, axis=1), 0.0)
    top1 = np.where(has, np.argmax(np.where(seen, raw, -np.inf), axis=1), -1)
    n_strong = ((raw > -70.0) & seen).sum(axis=1)
    frame = pd.DataFrame(np.hstack([z.astype(np.float32), seen.astype(np.uint8)]), columns=C.RSSI_COLUMNS + C.SEEN_COLUMNS)
    frame["n_heard"] = n_heard
    frame["max_rssi_dbm"] = strongest.astype(np.float32)
    frame["mean_rssi_dbm"] = mean.astype(np.float32)
    frame["std_rssi_dbm"] = std.astype(np.float32)
    frame["n_strong"] = n_strong
    frame["top1_ap_index"] = top1
    return frame[C.FEATURE_COLUMNS]


# --------------------------------------------------------------------------
# Scan editing and upload validation
# --------------------------------------------------------------------------


def perturb_scan(raw_row: np.ndarray, hide_strongest: int = 0, drop_fraction: float = 0.0,
                 noise_db: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a degraded scan. Returns (new_raw, removed_mask)."""
    raw = np.asarray(raw_row, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    removed = np.zeros(len(raw), dtype=bool)
    detected = np.flatnonzero(raw != MISSING)
    if hide_strongest > 0 and detected.size:
        order = detected[np.argsort(raw[detected])[::-1]]
        removed[order[:hide_strongest]] = True
    remaining = np.flatnonzero((raw != MISSING) & ~removed)
    if drop_fraction > 0 and remaining.size:
        k = int(round(drop_fraction * remaining.size))
        removed[rng.choice(remaining, size=k, replace=False)] = True
    raw[removed] = MISSING
    keep = raw != MISSING
    if noise_db > 0 and keep.any():
        raw[keep] = np.clip(raw[keep] + rng.normal(0.0, noise_db, keep.sum()), -110.0, 0.0)
    return raw, removed


def parse_upload(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Validate an uploaded scan table. Returns (raw matrix, problems)."""
    problems = []
    missing = [c for c in AP if c not in frame.columns]
    if missing:
        problems.append(f"Missing {len(missing)} WAP columns (for example {', '.join(missing[:3])}). "
                        "The file needs WAP001 to WAP520.")
        return np.empty((0, len(AP))), problems
    raw = frame[AP].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(raw).any():
        problems.append("Some WAP cells are empty or not numbers. Use 100 for an access point that was not detected.")
    bad = ~np.isnan(raw) & (raw != MISSING) & ((raw > 0) | (raw < -110))
    if bad.any():
        problems.append(f"{int(bad.sum())} WAP values are outside −110 to 0 dBm and are not 100.")
    if len(frame) == 0:
        problems.append("The file has no rows.")
    return raw, problems


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass
class Models:
    multitask: object
    coord: object
    prep: object


def load_models() -> Models:
    from signal2space_dl.train import load_model

    mt, _, prep = load_model(C.MODELS_DIR / "dl_05_multitask.pt")
    co, _, _ = load_model(C.MODELS_DIR / "dl_04_coordinate.pt")
    return Models(mt, co, prep)


def predict(models: Models, features: pd.DataFrame) -> dict:
    """Live predictions from the multitask network and the coordinate-only network."""
    import torch

    prep = models.prep
    x = torch.from_numpy(prep.features(features))
    with torch.no_grad():
        mt = models.multitask(x)
        co = models.coord(x)
    pb = torch.softmax(mt["building"], 1).numpy()
    pf = torch.softmax(mt["floor"], 1).numpy()
    return {
        "building_proba": pb, "floor_proba": pf,
        "building": np.asarray(prep.building_classes)[pb.argmax(1)],
        "floor": np.asarray(prep.floor_classes)[pf.argmax(1)],
        "coords_multitask": prep.unscale_coords(mt["coord"].numpy()),
        "coords_single": prep.unscale_coords(co["coord"].numpy()),
        "building_classes": list(prep.building_classes), "floor_classes": list(prep.floor_classes),
    }


# --------------------------------------------------------------------------
# Saved results
# --------------------------------------------------------------------------


def load_saved_predictions() -> dict[str, pd.DataFrame]:
    """Held-out predictions written by ML_05 and DL_02–DL_05, keyed by model name."""
    t = C.TABLES_DIR
    aml = pd.read_csv(t / "ml_05_aml_heldout_integrated_predictions.csv")
    mt = pd.read_csv(t / "dl_05_multitask_heldout_validation_predictions.csv")
    b = pd.read_csv(t / "dl_02_building_heldout_validation_predictions.csv")
    f = pd.read_csv(t / "dl_03_floor_heldout_validation_predictions.csv")
    c = pd.read_csv(t / "dl_04_coordinate_heldout_validation_predictions.csv")
    single = b[["ROW_ID", "BUILDINGID_TRUE", "BUILDINGID_PRED", "PHONEID"]].merge(
        f[["ROW_ID", "FLOOR_TRUE", "FLOOR_PRED"]], on="ROW_ID").merge(
        c[["ROW_ID", "LONGITUDE_TRUE", "LATITUDE_TRUE", "LONGITUDE_PRED", "LATITUDE_PRED", "EUCLIDEAN_ERROR"]], on="ROW_ID")
    keep = ["ROW_ID", "BUILDINGID_TRUE", "BUILDINGID_PRED", "FLOOR_TRUE", "FLOOR_PRED", "LONGITUDE_TRUE", "LATITUDE_TRUE",
            "LONGITUDE_PRED", "LATITUDE_PRED", "EUCLIDEAN_ERROR"]
    aml = aml[keep].merge(mt[["ROW_ID", "PHONEID"]], on="ROW_ID")
    return {"AML Random Forest": aml, "DL multitask": mt[keep + ["PHONEID"]], "DL single-task": single[keep + ["PHONEID"]]}


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(C.TABLES_DIR / name)


def hierarchy_label(b_ok: bool, f_ok: bool) -> str:
    return {(True, True): "Building and floor correct", (True, False): "Floor wrong",
            (False, True): "Building wrong", (False, False): "Both wrong"}[(bool(b_ok), bool(f_ok))]
