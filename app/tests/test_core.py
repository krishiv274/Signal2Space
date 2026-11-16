"""The app's live path must reproduce the saved Phase 1 features and Phase 2 predictions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402
from signal2space_dl import config as C  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    train, heldout = core.load_raw()
    scaler = core.fit_rssi_scaler(core.internal_train_rows(train))
    return train, heldout, scaler, core.load_models()


def test_transform_reproduces_stored_heldout_features(ctx):
    _, heldout, scaler, _ = ctx
    ours = core.features_from_raw(heldout[core.AP].to_numpy(), scaler)
    stored = pd.read_csv(C.AML_FEATURE_FILES["heldout_validation"])[C.FEATURE_COLUMNS]
    np.testing.assert_allclose(ours.to_numpy(np.float64), stored.to_numpy(np.float64), rtol=0, atol=1e-4)


def test_live_predictions_match_saved_dl_results(ctx):
    _, heldout, scaler, models = ctx
    pred = core.predict(models, core.features_from_raw(heldout[core.AP].to_numpy(), scaler))
    saved = core.load_saved_predictions()
    mt, single = saved["DL multitask"], saved["DL single-task"]
    np.testing.assert_array_equal(pred["floor"], mt.FLOOR_PRED.to_numpy())
    np.testing.assert_array_equal(pred["building"], mt.BUILDINGID_PRED.to_numpy())
    np.testing.assert_allclose(pred["coords_multitask"], mt[["LONGITUDE_PRED", "LATITUDE_PRED"]].to_numpy(), atol=1e-3)
    np.testing.assert_allclose(pred["coords_single"], single[["LONGITUDE_PRED", "LATITUDE_PRED"]].to_numpy(), atol=1e-3)


def test_perturb_hides_strongest_and_is_seeded():
    raw = np.full(520, 100.0)
    raw[[3, 10, 20, 30]] = [-40, -60, -80, -90]
    new, removed = core.perturb_scan(raw, hide_strongest=2)
    assert list(np.flatnonzero(removed)) == [3, 10] and new[3] == 100 and new[20] == -80
    a, _ = core.perturb_scan(raw, drop_fraction=0.5, noise_db=3, seed=1)
    b, _ = core.perturb_scan(raw, drop_fraction=0.5, noise_db=3, seed=1)
    np.testing.assert_array_equal(a, b)
    assert np.all(a[a != 100] <= 0)


def test_parse_upload_reports_problems():
    ok = pd.DataFrame([[100.0] * 520], columns=core.AP)
    raw, problems = core.parse_upload(ok)
    assert raw.shape == (1, 520) and problems == []
    bad = ok.copy()
    bad.loc[0, "WAP001"] = 25
    assert core.parse_upload(bad)[1]
    assert core.parse_upload(ok.drop(columns=["WAP520"]))[1]
