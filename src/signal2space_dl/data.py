"""Data loading, schema checks, the shared group split and leakage-aware preprocessing.

Leakage rules enforced here:
- The split is made on trainingData.csv only, grouping by BUILDINGID/FLOOR/SPACEID so
  every reference space lands entirely in internal-train or internal-validation.
- RSSI scaling statistics, imputation values, label encodings and coordinate
  normalisation are fitted on internal-train rows only.
- validationData.csv (the held-out partition) is only ever transformed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as C


class SchemaError(ValueError):
    """Raised when a CSV does not match the verified UJIIndoorLoc schema."""


# --------------------------------------------------------------------------- loading
def load_raw(paths: C.Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both supplied partitions, validate them and add a positional ROW_ID."""
    for p in (paths.train_csv, paths.heldout_csv):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}. Place trainingData.csv and validationData.csv in data/raw/.")
    train = pd.read_csv(paths.train_csv)
    heldout = pd.read_csv(paths.heldout_csv)
    validate_schema(train, "trainingData.csv")
    validate_schema(heldout, "validationData.csv")
    def with_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        keys = pd.DataFrame({"ROW_ID": np.arange(len(frame), dtype=np.int64), "SOURCE_PARTITION": name})
        return pd.concat([keys, frame.reset_index(drop=True)], axis=1)

    return with_keys(train, "trainingData"), with_keys(heldout, "validationData")


def validate_schema(frame: pd.DataFrame, name: str) -> None:
    """Check columns, order, blanks, dtypes and the RSSI value convention."""
    if frame.columns.tolist() != C.RAW_COLUMNS:
        missing = [c for c in C.RAW_COLUMNS if c not in frame.columns]
        extra = [c for c in frame.columns if c not in C.RAW_COLUMNS]
        raise SchemaError(f"{name}: unexpected columns/order (missing={missing[:5]}, extra={extra[:5]})")
    if frame.isna().any().any():
        bad = frame.columns[frame.isna().any()].tolist()
        raise SchemaError(f"{name}: blank cells in {bad[:5]}")
    non_numeric = [c for c in C.RAW_COLUMNS if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise SchemaError(f"{name}: non-numeric columns {non_numeric[:5]}")
    rssi = frame[C.WAP_COLUMNS].to_numpy()
    observed = rssi != C.MISSING_SIGNAL
    if not np.all(rssi[observed] <= 0) or not np.all(rssi[observed] >= -110):
        raise SchemaError(f"{name}: RSSI outside [-110, 0] dBm (excluding sentinel {C.MISSING_SIGNAL})")
    if not set(frame["BUILDINGID"].unique()) <= {0, 1, 2}:
        raise SchemaError(f"{name}: unexpected BUILDINGID values")
    if not set(frame["FLOOR"].unique()) <= {0, 1, 2, 3, 4}:
        raise SchemaError(f"{name}: unexpected FLOOR values")


def describe_quality(train: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    """Descriptive data-quality summary (held-out values are reported, never fitted)."""
    rows = []
    for name, f in (("trainingData", train), ("validationData", heldout)):
        seen = f[C.WAP_COLUMNS].to_numpy() != C.MISSING_SIGNAL
        rows.append({
            "partition": name,
            "rows": len(f),
            "duplicate_raw_rows": int(f[C.RAW_COLUMNS].duplicated().sum()),
            "rows_with_zero_detected_waps": int((seen.sum(1) == 0).sum()),
            "missing_signal_fraction": float((~seen).mean()),
            "median_detected_waps": float(np.median(seen.sum(1))),
            "waps_never_detected": int((seen.sum(0) == 0).sum()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- split
def make_group_split(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (internal_train_positions, internal_validation_positions).

    Identical arguments to ML_01 so the AML and DL internal partitions match row for row.
    """
    groups = (train["BUILDINGID"].astype(str) + "_" + train["FLOOR"].astype(str) + "_"
              + train["SPACEID"].astype(str))
    splitter = GroupShuffleSplit(n_splits=1, test_size=C.GROUP_TEST_SIZE, random_state=C.GROUP_RANDOM_STATE)
    tr, va = next(splitter.split(train, groups=groups))
    assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va])), "group leakage across internal split"
    assert len(tr) + len(va) == len(train)
    return tr, va


def build_split_manifest(train: pd.DataFrame, heldout: pd.DataFrame,
                         tr: np.ndarray, va: np.ndarray) -> pd.DataFrame:
    m_train = train[["SOURCE_PARTITION", "ROW_ID"]].copy()
    m_train["DL_SPLIT"] = "internal_train"
    m_train.iloc[va, m_train.columns.get_loc("DL_SPLIT")] = "internal_validation"
    m_held = heldout[["SOURCE_PARTITION", "ROW_ID"]].copy()
    m_held["DL_SPLIT"] = "heldout_validation"
    return pd.concat([m_train, m_held], ignore_index=True)


def check_aml_split_parity(paths: C.Paths, internal_validation_row_ids: np.ndarray) -> dict:
    """Compare DL internal-validation ROW_IDs with the AML prediction tables, if present."""
    ref = paths.tables / "ml_02_building_internal_validation_predictions.csv"
    if not ref.is_file():
        return {"checked": False, "reason": f"{ref.name} not found"}
    aml_ids = pd.read_csv(ref, usecols=["ROW_ID"])["ROW_ID"].to_numpy()
    same = set(aml_ids.tolist()) == set(internal_validation_row_ids.tolist())
    return {"checked": True, "identical_internal_validation_rows": bool(same),
            "aml_rows": int(len(aml_ids)), "dl_rows": int(len(internal_validation_row_ids))}


# ---------------------------------------------------------------------- preprocessing
@dataclass
class Preprocessor:
    """Per-WAP standardisation fitted on internal-train observed RSSI only.

    Inputs to the network (1,040 features):
    - 520 standardised RSSI values; undetected APs are filled with (per-WAP min z - 1),
      i.e. "weaker than anything seen in training" (same rule as ML_01);
    - 520 binary presence indicators (1 = detected).
    """

    means: np.ndarray
    scales: np.ndarray
    impute: np.ndarray
    building_classes: list[int]
    floor_classes: list[int]
    coord_mean: np.ndarray
    coord_scale: float
    fit_rows: int

    @classmethod
    def fit(cls, internal_train: pd.DataFrame, coord_scale: float) -> "Preprocessor":
        raw = internal_train[C.WAP_COLUMNS].to_numpy(dtype=np.float64)
        seen = raw != C.MISSING_SIGNAL
        all_obs = raw[seen]
        g_mean, g_std = all_obs.mean(), all_obs.std() or 1.0
        g_impute = ((all_obs - g_mean) / g_std).min() - 1.0
        means = np.full(C.N_WAPS, g_mean)
        scales = np.full(C.N_WAPS, g_std)
        impute = np.full(C.N_WAPS, g_impute)
        for j in range(C.N_WAPS):
            v = raw[seen[:, j], j]
            if v.size:
                means[j] = v.mean()
                scales[j] = v.std() if v.std() > 0 else 1.0
                impute[j] = ((v - means[j]) / scales[j]).min() - 1.0
        return cls(
            means=means, scales=scales, impute=impute,
            building_classes=sorted(internal_train["BUILDINGID"].unique().tolist()),
            floor_classes=sorted(internal_train["FLOOR"].unique().tolist()),
            coord_mean=internal_train[["LONGITUDE", "LATITUDE"]].to_numpy().mean(0),
            coord_scale=float(coord_scale),
            fit_rows=int(len(internal_train)),
        )

    # -- features
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[C.WAP_COLUMNS].to_numpy(dtype=np.float64)
        seen = raw != C.MISSING_SIGNAL
        z = (raw - self.means) / self.scales
        z = np.where(seen, z, self.impute)
        x = np.concatenate([z, seen.astype(np.float64)], axis=1).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("non-finite network inputs")
        return x

    # -- targets
    def encode_targets(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        b_map = {c: i for i, c in enumerate(self.building_classes)}
        f_map = {c: i for i, c in enumerate(self.floor_classes)}
        unknown_b = set(frame["BUILDINGID"]) - set(b_map)
        unknown_f = set(frame["FLOOR"]) - set(f_map)
        if unknown_b or unknown_f:
            raise ValueError(f"labels unseen in internal-train: building={unknown_b} floor={unknown_f}")
        coords = frame[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=np.float64)
        return {
            "building": frame["BUILDINGID"].map(b_map).to_numpy(np.int64),
            "floor": frame["FLOOR"].map(f_map).to_numpy(np.int64),
            "coordinate": ((coords - self.coord_mean) / self.coord_scale).astype(np.float32),
        }

    def decode_coordinates(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.coord_scale + self.coord_mean

    # -- persistence
    def to_dict(self) -> dict:
        return {
            "fit_partition": "trainingData/internal_train",
            "fit_rows": self.fit_rows,
            "missing_signal_sentinel": C.MISSING_SIGNAL,
            "input_layout": "520 standardised RSSI (WAP001..WAP520) + 520 presence indicators",
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "impute": self.impute.tolist(),
            "building_classes": self.building_classes,
            "floor_classes": self.floor_classes,
            "coord_mean": self.coord_mean.tolist(),
            "coord_scale": self.coord_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(np.asarray(d["means"]), np.asarray(d["scales"]), np.asarray(d["impute"]),
                   list(d["building_classes"]), list(d["floor_classes"]),
                   np.asarray(d["coord_mean"]), float(d["coord_scale"]), int(d["fit_rows"]))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        return cls.from_dict(json.loads(path.read_text()))


# --------------------------------------------------------------------- full dataset
@dataclass
class PreparedData:
    frames: dict[str, pd.DataFrame]          # split -> raw rows (targets + metadata)
    X: dict[str, np.ndarray]                 # split -> (n, 1040) float32
    y: dict[str, dict[str, np.ndarray]]      # split -> task -> encoded targets
    preprocessor: Preprocessor
    manifest: pd.DataFrame
    quality: pd.DataFrame
    parity: dict


SPLITS = ("internal_train", "internal_validation", "heldout_validation")


def prepare(paths: C.Paths, coord_scale: float = C.TrainConfig().coord_scale,
            save: bool = True) -> PreparedData:
    """Load -> validate -> split -> fit preprocessor on internal train -> transform all splits."""
    train, heldout = load_raw(paths)
    tr, va = make_group_split(train)
    frames = {
        "internal_train": train.iloc[tr].reset_index(drop=True),
        "internal_validation": train.iloc[va].reset_index(drop=True),
        "heldout_validation": heldout,
    }
    pre = Preprocessor.fit(frames["internal_train"], coord_scale)
    X = {s: pre.transform(f) for s, f in frames.items()}
    y = {s: pre.encode_targets(f) for s, f in frames.items()}
    manifest = build_split_manifest(train, heldout, tr, va)
    quality = describe_quality(train, heldout)
    parity = check_aml_split_parity(paths, frames["internal_validation"]["ROW_ID"].to_numpy())
    if save:
        paths.ensure()
        manifest.to_csv(paths.processed / "dl_01_split_manifest.csv", index=False)
        pre.save(paths.processed / "dl_01_preprocessor.json")
        quality.to_csv(paths.tables / "dl_01_data_quality.csv", index=False)
        summary = pd.DataFrame([{
            "split": s, "rows": len(f), "n_features": X[s].shape[1],
            **{f"building_{b}": int((f.BUILDINGID == b).sum()) for b in pre.building_classes},
            **{f"floor_{fl}": int((f.FLOOR == fl).sum()) for fl in pre.floor_classes},
        } for s, f in frames.items()])
        summary.to_csv(paths.tables / "dl_01_split_summary.csv", index=False)
        (paths.tables / "dl_01_split_parity.json").write_text(json.dumps(parity, indent=1))
    return PreparedData(frames, X, y, pre, manifest, quality, parity)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
