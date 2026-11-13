"""Reproducibility checks for the saved DL_05 artifacts.

These tests read the real AML feature tables and the committed model files, so
they are skipped automatically when those artifacts are not present.
"""

import json

import numpy as np
import pandas as pd
import pytest

from signal2space_dl import config as C
from signal2space_dl import data as D

MODEL = C.MODELS_DIR / "dl_05_multitask.pt"
PRED = C.TABLES_DIR / "dl_05_multitask_heldout_validation_predictions.csv"
needs_artifacts = pytest.mark.skipif(
    not (MODEL.is_file() and PRED.is_file() and C.AML_FEATURE_FILES["heldout_validation"].is_file()),
    reason="DL_05 artifacts or AML feature tables not present")


@needs_artifacts
def test_checkpoint_reproduces_saved_heldout_predictions():
    from signal2space_dl.train import load_model, predict

    model, cfg, prep = load_model(MODEL)
    arrays = prep.transform(D.load_heldout_split())
    pred = predict(model, arrays, prep)
    saved = pd.read_csv(PRED)
    assert (saved.ROW_ID.to_numpy() == arrays.meta.ROW_ID.to_numpy()).all()
    np.testing.assert_array_equal(saved.FLOOR_PRED.to_numpy(), pred["floor"])
    np.testing.assert_array_equal(saved.BUILDINGID_PRED.to_numpy(), pred["building"])
    np.testing.assert_allclose(saved[["LONGITUDE_PRED", "LATITUDE_PRED"]].to_numpy(), pred["coords"], atol=1e-3)


@needs_artifacts
def test_selected_config_was_chosen_on_internal_validation():
    selection = pd.read_csv(C.TABLES_DIR / "dl_05_multitask_selection.csv")
    chosen = json.loads((C.MODELS_DIR / "dl_05_multitask_selected_config.json").read_text())
    best = selection.sort_values("val_selection_score").iloc[0]
    assert best.config == chosen["name"]
    assert bool(best.selected)


@needs_artifacts
def test_comparison_uses_identical_heldout_rows():
    dl = pd.read_csv(PRED)
    aml = pd.read_csv(C.AML_HELDOUT_PREDICTIONS_PATH)
    assert len(dl) == len(aml) == 1111
    m = dl.merge(aml, on=C.KEY_COLUMNS, suffixes=("_dl", "_aml"), validate="one_to_one")
    assert (m.FLOOR_TRUE_dl == m.FLOOR_TRUE_aml).all()
