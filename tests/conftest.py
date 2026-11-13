"""Pytest configuration: make ``src/`` importable and provide small synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signal2space_dl import config as C  # noqa: E402


def make_feature_frame(n: int = 60, partition: str = "trainingData", seed: int = 0) -> pd.DataFrame:
    """A tiny frame that follows the AML feature-table contract."""
    rng = np.random.default_rng(seed)
    data = {
        "ROW_ID": np.arange(n), "SOURCE_PARTITION": partition,
        "SPACEID": rng.integers(1, 50, n), "RELATIVEPOSITION": rng.integers(1, 3, n),
        "USERID": rng.integers(1, 5, n), "PHONEID": rng.integers(1, 4, n), "TIMESTAMP": 1_370_000_000 + np.arange(n),
        "BUILDINGID": np.arange(n) % 3, "FLOOR": np.arange(n) % 4,
        "LONGITUDE": -7500 + rng.normal(0, 100, n), "LATITUDE": 4_864_900 + rng.normal(0, 60, n),
    }
    frame = pd.DataFrame(data)
    seen = rng.random((n, len(C.AP_COLUMNS))) < 0.05
    rssi = np.where(seen, rng.normal(0, 1, seen.shape), -3.0)
    feats = pd.DataFrame(np.hstack([rssi, seen.astype(np.uint8)]), columns=C.RSSI_COLUMNS + C.SEEN_COLUMNS)
    feats["n_heard"] = seen.sum(axis=1)
    feats["max_rssi_dbm"] = rng.uniform(-90, -30, n)
    feats["mean_rssi_dbm"] = rng.uniform(-95, -50, n)
    feats["std_rssi_dbm"] = rng.uniform(0, 15, n)
    feats["n_strong"] = rng.integers(0, 6, n)
    feats["top1_ap_index"] = rng.integers(0, 520, n)
    return pd.concat([frame, feats], axis=1)


@pytest.fixture
def train_frame():
    return make_feature_frame(90, "trainingData", seed=1)


@pytest.fixture
def heldout_frame():
    return make_feature_frame(30, "validationData", seed=2)
