"""Shared configuration for the Signal2Space DL phase.

Everything that two or more modules must agree on lives here: repository paths,
the AML feature/target/metadata column contract, random seeds and the default
training settings. Keep this file free of heavy imports so it can be loaded by
tests and notebooks cheaply.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root (the folder holding ``data/raw``)."""
    candidate = Path.cwd() if start is None else Path(start)
    for parent in (candidate, *candidate.parents):
        if (parent / "data" / "raw").is_dir() and (parent / "Machine Learning Pipeline").is_dir():
            return parent
    # Fall back to the location of this file: <root>/src/signal2space_dl/config.py
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TABLES_DIR = DATA_DIR / "tables"
FIGURES_DIR = DATA_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "dl"

RAW_TRAIN_PATH = RAW_DIR / "trainingData.csv"
RAW_HELDOUT_PATH = RAW_DIR / "validationData.csv"

# Feature tables written by ML_01 (the shared AML contract).
AML_FEATURE_FILES = {
    "internal_train": PROCESSED_DIR / "aml_internal_train_features.csv",
    "internal_validation": PROCESSED_DIR / "aml_internal_validation_features.csv",
    "heldout_validation": PROCESSED_DIR / "aml_heldout_validation_features.csv",
}

# AML results used for the fair comparison (written by ML_05).
AML_SUMMARY_PATH = TABLES_DIR / "ml_05_aml_summary_metrics.csv"
AML_HIERARCHY_PATH = TABLES_DIR / "ml_05_aml_hierarchical_error_breakdown.csv"
AML_HELDOUT_PREDICTIONS_PATH = TABLES_DIR / "ml_05_aml_heldout_integrated_predictions.csv"

# --------------------------------------------------------------------------
# Column contract (mirrors ML_01)
# --------------------------------------------------------------------------

N_WAPS = 520
MISSING_SIGNAL = 100  # raw RSSI value meaning "access point not detected"
AP_COLUMNS = [f"WAP{i:03d}" for i in range(1, N_WAPS + 1)]
RSSI_COLUMNS = [f"RSSI_Z__{ap}" for ap in AP_COLUMNS]
SEEN_COLUMNS = [f"SEEN__{ap}" for ap in AP_COLUMNS]
SUMMARY_COLUMNS = ["n_heard", "max_rssi_dbm", "mean_rssi_dbm", "std_rssi_dbm", "n_strong", "top1_ap_index"]
FEATURE_COLUMNS = RSSI_COLUMNS + SEEN_COLUMNS + SUMMARY_COLUMNS  # 1,046 predictors

TARGET_COLUMNS = ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]
COORDINATE_COLUMNS = ["LONGITUDE", "LATITUDE"]
METADATA_COLUMNS = ["ROW_ID", "SOURCE_PARTITION", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]
KEY_COLUMNS = ["SOURCE_PARTITION", "ROW_ID"]  # ROW_ID restarts in each source file

# Coordinates are reported in the dataset's own units. The repository has no
# authoritative CRS metadata, so DL outputs never label them as metres.
COORDINATE_UNIT_LABEL = "dataset coordinate units"

SPLITS = ("internal_train", "internal_validation", "heldout_validation")

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

GLOBAL_SEED = 42
VARIABILITY_SEEDS = (42, 7, 2026)  # used only after the configuration is frozen


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for repeatable runs."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# --------------------------------------------------------------------------
# Training settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one multitask (or single-task) training run."""

    name: str = "default"
    hidden_sizes: tuple[int, ...] = (512, 256)
    head_hidden: int = 128
    dropout: float = 0.3
    input_dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 150
    patience: int = 15
    # Loss weights: building CE, floor CE, coordinate Smooth-L1 on standardized coords.
    w_building: float = 1.0
    w_floor: float = 1.0
    w_coord: float = 1.0
    # Smooth-L1 transition point in standardized coordinate units (std of LONGITUDE ~ 124,
    # LATITUDE ~ 67 dataset units). 0.1 behaves close to L1 for errors above ~7-12 units.
    coord_beta: float = 0.1
    # Which heads are trained ("building", "floor", "coord"); all three = multitask.
    tasks: tuple[str, ...] = ("building", "floor", "coord")
    seed: int = GLOBAL_SEED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        d["tasks"] = list(self.tasks)
        return d


# Small, CPU-friendly selection grid. Every candidate is scored on the internal
# validation split only; the held-out file is never used to choose among them.
SELECTION_GRID: tuple[TrainConfig, ...] = (
    TrainConfig(name="mt_512x256_d30"),
    TrainConfig(name="mt_1024x512_d30", hidden_sizes=(1024, 512)),
    TrainConfig(name="mt_512x256_d50", dropout=0.5, input_dropout=0.2),
    TrainConfig(name="mt_1024x512x256_d30", hidden_sizes=(1024, 512, 256)),
    TrainConfig(name="mt_512x256_d30_floorx2", w_floor=2.0),
    TrainConfig(name="mt_512x256_d30_coordx2", w_coord=2.0),
    TrainConfig(name="mt_512x256_d30_beta1", coord_beta=1.0),
)
