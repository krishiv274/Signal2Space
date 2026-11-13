"""Data loading, validation and train-only preprocessing for the DL phase.

Design
------
The DL phase reuses the three AML feature tables written by ML_01 instead of
re-deriving a second feature set. Those tables already contain:

* 520 ``RSSI_Z__WAPxxx`` values standardized per access point with means and
  scales fitted on the internal-training split only; undetected readings (raw
  value ``100``) are imputed with a train-only value one unit below the lowest
  standardized observation for that access point;
* 520 binary ``SEEN__WAPxxx`` missingness indicators;
* 6 raw signal summaries (``n_heard``, ``max_rssi_dbm``, ``mean_rssi_dbm``,
  ``std_rssi_dbm``, ``n_strong``, ``top1_ap_index``).

Neural networks need the summaries on a comparable scale, so this module fits
one additional standardizer on the internal-training rows only (summary
columns), plus a coordinate standardizer and label mappings, also on
internal-training rows only. Held-out rows are transform-only.

Identifiers (``USERID``, ``PHONEID``, ...) and targets are carried alongside the
arrays for evaluation and robustness analysis but are never model inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_EXPECTED_PARTITION = {
    "internal_train": "trainingData",
    "internal_validation": "trainingData",
    "heldout_validation": "validationData",
}


def load_split(split: str, path: Path | None = None) -> pd.DataFrame:
    """Load one AML feature table by split name."""
    if split not in C.AML_FEATURE_FILES:
        raise KeyError(f"Unknown split {split!r}; expected one of {list(C.AML_FEATURE_FILES)}")
    path = Path(path) if path is not None else C.AML_FEATURE_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'Machine Learning Pipeline/ML_01_data_preparation_features.ipynb' first."
        )
    return pd.read_csv(path)


def load_development_splits() -> dict[str, pd.DataFrame]:
    """Load the internal-training and internal-validation tables (no held-out data)."""
    return {s: load_split(s) for s in ("internal_train", "internal_validation")}


def load_heldout_split() -> pd.DataFrame:
    """Load the supplied held-out table. Call only after a configuration is frozen."""
    return load_split("heldout_validation")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_feature_table(frame: pd.DataFrame, split: str) -> list[str]:
    """Validate one feature table against the AML contract; return check messages.

    Raises ``AssertionError`` on the first violated check.
    """
    msgs: list[str] = []
    required = C.METADATA_COLUMNS + C.TARGET_COLUMNS + C.FEATURE_COLUMNS
    missing = [c for c in required if c not in frame.columns]
    assert not missing, f"{split}: missing columns {missing[:5]}... ({len(missing)} total)"
    msgs.append(f"{split}: all {len(C.FEATURE_COLUMNS)} feature columns and targets/metadata present")

    assert not (set(C.FEATURE_COLUMNS) & set(C.TARGET_COLUMNS + C.METADATA_COLUMNS)), "feature/target overlap"
    assert frame["ROW_ID"].is_unique, f"{split}: ROW_ID not unique"
    expected = _EXPECTED_PARTITION[split]
    assert set(frame["SOURCE_PARTITION"]) == {expected}, f"{split}: SOURCE_PARTITION must be {expected}"
    msgs.append(f"{split}: {len(frame)} rows, unique ROW_ID, SOURCE_PARTITION={expected}")

    values = frame[C.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), f"{split}: non-finite feature values"
    seen = frame[C.SEEN_COLUMNS].to_numpy()
    assert set(np.unique(seen)).issubset({0, 1}), f"{split}: SEEN indicators are not binary"
    assert frame[C.TARGET_COLUMNS].notna().all().all(), f"{split}: missing targets"
    msgs.append(f"{split}: finite predictors, binary indicators, complete targets")
    return msgs


def validate_development_contract(train: pd.DataFrame, ival: pd.DataFrame) -> list[str]:
    """Cross-split checks that use development data only."""
    msgs = validate_feature_table(train, "internal_train") + validate_feature_table(ival, "internal_validation")
    assert set(train.ROW_ID).isdisjoint(set(ival.ROW_ID)), "internal train/validation ROW_IDs overlap"
    tg = set(zip(train.BUILDINGID, train.FLOOR, train.SPACEID))
    vg = set(zip(ival.BUILDINGID, ival.FLOOR, ival.SPACEID))
    assert tg.isdisjoint(vg), "building/floor/space groups overlap across internal splits"
    msgs.append("internal splits: disjoint ROW_IDs and disjoint (BUILDINGID, FLOOR, SPACEID) groups")

    # The AML RSSI scaler must have been fitted on internal-training rows only:
    # observed standardized values have ~zero mean there.
    z = train[C.RSSI_COLUMNS].to_numpy(dtype=np.float64)
    s = train[C.SEEN_COLUMNS].to_numpy(dtype=bool)
    col_counts = s.sum(axis=0)
    col_means = np.where(col_counts > 0, (z * s).sum(axis=0) / np.maximum(col_counts, 1), 0.0)
    assert np.abs(col_means).max() < 1e-3, "RSSI_Z is not centred on internal-train observations"
    msgs.append("AML RSSI scaler verified as fitted on internal-training observations (per-WAP mean ~ 0)")
    return msgs


def verify_split_against_raw(train: pd.DataFrame, ival: pd.DataFrame,
                             raw_path: Path = C.RAW_TRAIN_PATH) -> list[str]:
    """Rebuild ML_01's grouped split from the raw CSV and confirm it matches."""
    from sklearn.model_selection import GroupShuffleSplit

    raw = pd.read_csv(raw_path)
    raw.insert(0, "ROW_ID", np.arange(len(raw), dtype=np.int64))
    groups = raw.BUILDINGID.astype(str) + "_" + raw.FLOOR.astype(str) + "_" + raw.SPACEID.astype(str)
    tr_pos, va_pos = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(raw, groups=groups))
    assert set(raw.ROW_ID.iloc[tr_pos]) == set(train.ROW_ID), "internal_train rows differ from raw rebuild"
    assert set(raw.ROW_ID.iloc[va_pos]) == set(ival.ROW_ID), "internal_validation rows differ from raw rebuild"
    merged = train[["ROW_ID"] + C.TARGET_COLUMNS].merge(raw[["ROW_ID"] + C.TARGET_COLUMNS], on="ROW_ID",
                                                        suffixes=("", "_raw"))
    for col in C.TARGET_COLUMNS:
        assert np.allclose(merged[col], merged[f"{col}_raw"]), f"{col} differs from raw"
    return [f"raw rebuild: GroupShuffleSplit(test_size=0.20, random_state=42) reproduces "
            f"{len(tr_pos)} / {len(va_pos)} rows and targets match trainingData.csv"]


def describe_overlap(reference: pd.DataFrame, other: pd.DataFrame, other_name: str) -> dict:
    """Descriptive (never used for fitting) fingerprint/position overlap counts."""
    ref_fp = pd.util.hash_pandas_object(reference[C.RSSI_COLUMNS + C.SEEN_COLUMNS], index=False)
    oth_fp = pd.util.hash_pandas_object(other[C.RSSI_COLUMNS + C.SEEN_COLUMNS], index=False)
    ref_pos = set(zip(reference.LONGITUDE.round(6), reference.LATITUDE.round(6)))
    oth_pos = list(zip(other.LONGITUDE.round(6), other.LATITUDE.round(6)))
    return {
        "comparison": f"internal_train vs {other_name}",
        "rows": len(other),
        "rows_with_identical_fingerprint_in_internal_train": int(oth_fp.isin(set(ref_fp)).sum()),
        "duplicate_fingerprints_within_split": int(oth_fp.duplicated().sum()),
        "rows_with_exact_position_in_internal_train": int(sum(p in ref_pos for p in oth_pos)),
        "distinct_positions": len(set(oth_pos)),
        "zero_detected_ap_rows": int((other["n_heard"] == 0).sum()),
    }


# --------------------------------------------------------------------------
# Train-only preprocessing for the neural network
# --------------------------------------------------------------------------


@dataclass
class Preprocessor:
    """Transforms fitted on internal-training rows only."""

    summary_mean: np.ndarray
    summary_std: np.ndarray
    coord_mean: np.ndarray
    coord_std: np.ndarray
    building_classes: list[int]
    floor_classes: list[int]
    fit_rows: int
    fit_split: str = "internal_train"

    # ---- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "fit_split": self.fit_split,
            "fit_rows": int(self.fit_rows),
            "feature_columns": C.FEATURE_COLUMNS,
            "summary_columns": C.SUMMARY_COLUMNS,
            "summary_mean": self.summary_mean.tolist(),
            "summary_std": self.summary_std.tolist(),
            "coordinate_columns": C.COORDINATE_COLUMNS,
            "coord_mean": self.coord_mean.tolist(),
            "coord_std": self.coord_std.tolist(),
            "building_classes": [int(x) for x in self.building_classes],
            "floor_classes": [int(x) for x in self.floor_classes],
            "coordinate_unit": C.COORDINATE_UNIT_LABEL,
        }

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        d = json.loads(Path(path).read_text())
        assert d["feature_columns"] == C.FEATURE_COLUMNS, "saved feature contract differs"
        return cls(
            summary_mean=np.asarray(d["summary_mean"]), summary_std=np.asarray(d["summary_std"]),
            coord_mean=np.asarray(d["coord_mean"]), coord_std=np.asarray(d["coord_std"]),
            building_classes=d["building_classes"], floor_classes=d["floor_classes"],
            fit_rows=d["fit_rows"], fit_split=d["fit_split"],
        )

    # ---- transforms --------------------------------------------------------
    @property
    def n_buildings(self) -> int:
        return len(self.building_classes)

    @property
    def n_floors(self) -> int:
        return len(self.floor_classes)

    def features(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[C.FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
        k = len(C.RSSI_COLUMNS) + len(C.SEEN_COLUMNS)
        x[:, k:] = (x[:, k:] - self.summary_mean) / self.summary_std
        return x.astype(np.float32)

    def encode_labels(self, values, classes: list[int], name: str) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(classes)}
        unknown = sorted(set(np.unique(values)) - set(lookup))
        if unknown:
            raise ValueError(f"{name} labels {unknown} were not seen in internal training")
        return np.array([lookup[v] for v in values], dtype=np.int64)

    def scale_coords(self, coords: np.ndarray) -> np.ndarray:
        return ((coords - self.coord_mean) / self.coord_std).astype(np.float32)

    def unscale_coords(self, scaled: np.ndarray) -> np.ndarray:
        return np.asarray(scaled, dtype=np.float64) * self.coord_std + self.coord_mean

    def transform(self, frame: pd.DataFrame) -> "SplitArrays":
        coords = frame[C.COORDINATE_COLUMNS].to_numpy(dtype=np.float64)
        return SplitArrays(
            x=self.features(frame),
            building=self.encode_labels(frame.BUILDINGID.to_numpy(), self.building_classes, "BUILDINGID"),
            floor=self.encode_labels(frame.FLOOR.to_numpy(), self.floor_classes, "FLOOR"),
            coords=coords,
            coords_scaled=self.scale_coords(coords),
            meta=frame[C.METADATA_COLUMNS].reset_index(drop=True).copy(),
        )


@dataclass
class SplitArrays:
    """Model-ready arrays for one split plus the metadata kept for analysis."""

    x: np.ndarray
    building: np.ndarray
    floor: np.ndarray
    coords: np.ndarray
    coords_scaled: np.ndarray
    meta: pd.DataFrame

    def __len__(self) -> int:
        return len(self.x)


def fit_preprocessor(train: pd.DataFrame) -> Preprocessor:
    """Fit summary/coordinate standardizers and label maps on internal training only."""
    assert set(train["SOURCE_PARTITION"]) == {"trainingData"}, "fit on internal training only"
    summaries = train[C.SUMMARY_COLUMNS].to_numpy(dtype=np.float64)
    s_mean = summaries.mean(axis=0)
    s_std = summaries.std(axis=0)
    s_std[s_std == 0] = 1.0
    coords = train[C.COORDINATE_COLUMNS].to_numpy(dtype=np.float64)
    c_std = coords.std(axis=0)
    c_std[c_std == 0] = 1.0
    return Preprocessor(
        summary_mean=s_mean, summary_std=s_std,
        coord_mean=coords.mean(axis=0), coord_std=c_std,
        building_classes=sorted(int(v) for v in train.BUILDINGID.unique()),
        floor_classes=sorted(int(v) for v in train.FLOOR.unique()),
        fit_rows=len(train),
    )


def centroid_baseline_error(train: pd.DataFrame, ival: pd.DataFrame) -> float:
    """Mean Euclidean error of predicting the internal-train centroid for every ival row.

    Used only to normalise coordinate error inside the model-selection score, so
    that building error, floor error and coordinate error contribute on
    comparable scales. Uses development data only.
    """
    centroid = train[C.COORDINATE_COLUMNS].to_numpy().mean(axis=0)
    d = ival[C.COORDINATE_COLUMNS].to_numpy() - centroid
    return float(np.sqrt((d ** 2).sum(axis=1)).mean())


# --------------------------------------------------------------------------
# DL_01 report
# --------------------------------------------------------------------------


def prepare_and_report(tables_dir: Path = C.TABLES_DIR, models_dir: Path = C.MODELS_DIR) -> dict:
    """DL_01: validate the AML contract, fit/save train-only transforms and write report tables."""
    tables_dir, models_dir = Path(tables_dir), Path(models_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    dev = load_development_splits()
    checks = validate_development_contract(dev["internal_train"], dev["internal_validation"])
    checks += verify_split_against_raw(dev["internal_train"], dev["internal_validation"])
    prep = fit_preprocessor(dev["internal_train"])
    prep.save(models_dir / "preprocessor.json")
    coord_norm = centroid_baseline_error(dev["internal_train"], dev["internal_validation"])

    # The held-out table is only schema-checked and described here; nothing is fitted on it.
    heldout = load_heldout_split()
    checks += validate_feature_table(heldout, "heldout_validation")
    prep.transform(heldout)  # raises if held-out contains labels unseen in internal training
    checks.append("heldout_validation: all building/floor labels exist in the internal-train mappings")

    overlap = pd.DataFrame([describe_overlap(dev["internal_train"], dev["internal_validation"], "internal_validation"),
                            describe_overlap(dev["internal_train"], heldout, "heldout_validation")])
    train_groups = {
        "USERID": set(dev["internal_train"].USERID), "PHONEID": set(dev["internal_train"].PHONEID)}
    identity = pd.DataFrame([{
        "split": name, "rows": len(f), "distinct_USERID": f.USERID.nunique(), "distinct_PHONEID": f.PHONEID.nunique(),
        "USERID_values": ",".join(map(str, sorted(f.USERID.unique()))),
        "PHONEID_values": ",".join(map(str, sorted(f.PHONEID.unique()))),
        "rows_with_PHONEID_seen_in_internal_train": int(f.PHONEID.isin(train_groups["PHONEID"]).sum()),
        "timestamp_min_utc": pd.to_datetime(f.TIMESTAMP.min(), unit="s"),
        "timestamp_max_utc": pd.to_datetime(f.TIMESTAMP.max(), unit="s"),
    } for name, f in [("internal_train", dev["internal_train"]), ("internal_validation", dev["internal_validation"]),
                      ("heldout_validation", heldout)]])
    summary = pd.DataFrame({
        "item": ["fit_split", "fit_rows", "n_features", "rssi_features", "seen_indicators", "summary_features",
                 "building_classes", "floor_classes", "coord_mean_longitude", "coord_mean_latitude",
                 "coord_std_longitude", "coord_std_latitude", "centroid_baseline_mean_euclidean_internal_val"],
        "value": [prep.fit_split, prep.fit_rows, len(C.FEATURE_COLUMNS), len(C.RSSI_COLUMNS), len(C.SEEN_COLUMNS),
                  len(C.SUMMARY_COLUMNS), str(prep.building_classes), str(prep.floor_classes),
                  prep.coord_mean[0], prep.coord_mean[1], prep.coord_std[0], prep.coord_std[1], coord_norm]})
    pd.DataFrame({"check": checks}).to_csv(tables_dir / "dl_01_data_validation_checks.csv", index=False)
    overlap.to_csv(tables_dir / "dl_01_partition_overlap.csv", index=False)
    identity.to_csv(tables_dir / "dl_01_partition_identity_summary.csv", index=False)
    summary.to_csv(tables_dir / "dl_01_preprocessing_summary.csv", index=False)
    return {"checks": checks, "overlap": overlap, "identity": identity, "summary": summary, "preprocessor": prep}
