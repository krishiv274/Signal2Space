"""Tests for signal2space_dl.data (train-only preprocessing and validation)."""

import numpy as np
import pytest

from signal2space_dl import config as C
from signal2space_dl import data as D


def test_feature_contract_size():
    assert len(C.FEATURE_COLUMNS) == 2 * 520 + 6
    assert not set(C.FEATURE_COLUMNS) & set(C.TARGET_COLUMNS + C.METADATA_COLUMNS)


def test_preprocessor_fits_on_training_only(train_frame, heldout_frame):
    with pytest.raises(AssertionError):
        D.fit_preprocessor(heldout_frame)  # validationData rows must never be used for fitting
    prep = D.fit_preprocessor(train_frame)
    assert prep.fit_rows == len(train_frame)
    x = prep.features(train_frame)
    k = len(C.RSSI_COLUMNS) + len(C.SEEN_COLUMNS)
    np.testing.assert_allclose(x[:, k:].mean(axis=0), 0, atol=1e-5)
    # RSSI and SEEN columns pass through unchanged (already train-scaled by ML_01).
    np.testing.assert_allclose(x[:, :k], train_frame[C.FEATURE_COLUMNS[:k]].to_numpy(np.float32))


def test_transform_and_coordinate_roundtrip(train_frame, heldout_frame):
    prep = D.fit_preprocessor(train_frame)
    arr = prep.transform(heldout_frame)
    assert arr.x.shape == (len(heldout_frame), len(C.FEATURE_COLUMNS))
    assert arr.x.dtype == np.float32
    np.testing.assert_allclose(prep.unscale_coords(arr.coords_scaled), arr.coords, rtol=0, atol=1e-2)
    assert list(arr.meta.columns) == C.METADATA_COLUMNS  # identifiers kept for analysis, not in x


def test_unknown_label_is_rejected(train_frame, heldout_frame):
    prep = D.fit_preprocessor(train_frame)
    bad = heldout_frame.copy()
    bad.loc[0, "FLOOR"] = 9
    with pytest.raises(ValueError, match="FLOOR"):
        prep.transform(bad)


def test_preprocessor_save_load(tmp_path, train_frame):
    prep = D.fit_preprocessor(train_frame)
    prep.save(tmp_path / "prep.json")
    again = D.Preprocessor.load(tmp_path / "prep.json")
    np.testing.assert_allclose(again.coord_mean, prep.coord_mean)
    assert again.floor_classes == prep.floor_classes


def test_validate_feature_table_flags_bad_indicator(train_frame):
    D.validate_feature_table(train_frame, "internal_train")
    bad = train_frame.copy()
    bad.loc[0, C.SEEN_COLUMNS[0]] = 2
    with pytest.raises(AssertionError, match="binary"):
        D.validate_feature_table(bad, "internal_train")


def test_centroid_baseline_positive(train_frame):
    assert D.centroid_baseline_error(train_frame.iloc[:60], train_frame.iloc[60:]) > 0
