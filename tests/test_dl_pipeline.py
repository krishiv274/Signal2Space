"""Fast checks for the DL pipeline. Run from the repository root:  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signal2space_dl import config as C  # noqa: E402
from signal2space_dl.data import Preprocessor, SchemaError, make_group_split, validate_schema  # noqa: E402
from signal2space_dl.model import MultitaskLoss, MultitaskMLP  # noqa: E402


def synthetic_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rssi = np.where(rng.random((n, C.N_WAPS)) < 0.9, C.MISSING_SIGNAL, rng.integers(-104, -30, (n, C.N_WAPS)))
    df = pd.DataFrame(rssi, columns=C.WAP_COLUMNS)
    df["LONGITUDE"] = rng.uniform(-7690, -7300, n)
    df["LATITUDE"] = rng.uniform(4864746, 4865017, n)
    df["FLOOR"] = rng.integers(0, 4, n)
    df["BUILDINGID"] = rng.integers(0, 3, n)
    df["SPACEID"] = rng.integers(1, 8, n)
    for c in ["RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]:
        df[c] = 1
    return df[C.RAW_COLUMNS]


def test_schema_accepts_valid_frame():
    validate_schema(synthetic_frame(), "synthetic")


@pytest.mark.parametrize("mutate", [
    lambda f: f.drop(columns=["WAP001"]),
    lambda f: f.assign(WAP002=5),
    lambda f: f.assign(BUILDINGID=7),
    lambda f: f.assign(FLOOR=np.nan),
])
def test_schema_rejects_bad_frames(mutate):
    with pytest.raises(SchemaError):
        validate_schema(mutate(synthetic_frame()), "bad")


def test_group_split_has_no_group_leakage():
    f = synthetic_frame(400)
    tr, va = make_group_split(f)
    key = f.BUILDINGID.astype(str) + "_" + f.FLOOR.astype(str) + "_" + f.SPACEID.astype(str)
    assert set(key.iloc[tr]).isdisjoint(set(key.iloc[va]))
    assert len(tr) + len(va) == len(f)


def test_preprocessor_fits_on_train_only_and_handles_missing():
    train, other = synthetic_frame(80, 1), synthetic_frame(30, 2)
    pre = Preprocessor.fit(train, coord_scale=10.0)
    pre2 = Preprocessor.fit(train, coord_scale=10.0)
    _ = pre.transform(other)  # transforming another split must not change fitted state
    assert np.array_equal(pre.means, pre2.means) and np.array_equal(pre.impute, pre2.impute)
    x = pre.transform(train)
    assert x.shape == (80, 2 * C.N_WAPS) and np.isfinite(x).all()
    seen = train[C.WAP_COLUMNS].to_numpy() != C.MISSING_SIGNAL
    assert np.array_equal(x[:, C.N_WAPS:], seen.astype(np.float32))
    z = x[:, :C.N_WAPS]
    for j in np.where(seen.any(0))[0][:20]:
        assert z[~seen[:, j], j].max(initial=-np.inf) < z[seen[:, j], j].min()
    y = pre.encode_targets(train)
    xy = pre.decode_coordinates(y["coordinate"])
    assert np.allclose(xy, train[["LONGITUDE", "LATITUDE"]].to_numpy(), atol=1e-3)


def test_unseen_label_is_rejected():
    train = synthetic_frame(50)
    train["FLOOR"] = 0
    pre = Preprocessor.fit(train, 10.0)
    other = synthetic_frame(5)
    other["FLOOR"] = 3
    with pytest.raises(ValueError):
        pre.encode_targets(other)


@pytest.mark.parametrize("tasks", [("building",), ("floor",), ("coordinate",), C.TASKS])
def test_model_shapes_and_loss(tasks):
    m = MultitaskMLP(2 * C.N_WAPS, 3, 5, tasks)
    out = m(torch.randn(8, 2 * C.N_WAPS))
    assert set(out) == set(tasks)
    dims = {"building": 3, "floor": 5, "coordinate": 2}
    assert all(out[t].shape == (8, dims[t]) for t in tasks)
    targets = {"building": torch.randint(0, 3, (8,)), "floor": torch.randint(0, 5, (8,)),
               "coordinate": torch.randn(8, 2)}
    loss, parts = MultitaskLoss(tasks, C.TrainConfig().loss_weights)(out, targets)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == set(tasks)
