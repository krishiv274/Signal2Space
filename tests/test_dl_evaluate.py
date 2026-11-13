"""Tests for signal2space_dl.evaluate (metrics and error analysis)."""

import numpy as np
import pandas as pd
import pytest

from signal2space_dl import evaluate as E


def _frame():
    meta = pd.DataFrame({"ROW_ID": range(4), "SOURCE_PARTITION": "validationData", "USERID": 0,
                         "PHONEID": [1, 1, 2, 2]})
    pred = {"building": np.array([0, 0, 1, 1]), "building_proba": np.eye(3)[[0, 0, 1, 1]],
            "floor": np.array([0, 1, 1, 3]), "floor_proba": np.eye(5)[[0, 1, 1, 3]],
            "coords": np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])}
    return E.build_prediction_frame(meta, [0, 0, 2, 2], [0, 0, 1, 2], np.zeros((4, 2)), pred, [0, 1, 2],
                                    [0, 1, 2, 3, 4])


def test_coordinate_metrics_known_values():
    m = E.coordinate_metrics(np.zeros((2, 2)), np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert m["mae_longitude"] == 1.5 and m["mae_latitude"] == 2.0
    assert m["mean_euclidean_error"] == 2.5


def test_hierarchy_outcomes_and_breakdown():
    f = _frame()
    assert list(f.HIERARCHY_OUTCOME) == ["both_correct", "building_correct_floor_wrong",
                                         "building_wrong_floor_correct", "both_wrong"]
    h = E.hierarchical_breakdown(f, "heldout_validation")
    assert h.n_rows.sum() == 4 and h.proportion.sum() == pytest.approx(1)
    assert h.set_index("hierarchy_outcome").loc["building_correct_floor_wrong", "mean_coordinate_error"] == 5.0


def test_confusion_layout_rows_are_actual():
    cm = E.confusion_frame([0, 0, 1], [0, 1, 1], [0, 1])
    assert cm.loc[0, "1"] == 1 and cm.loc[1, "1"] == 1 and cm.index.name == "actual"


def test_summary_rows_and_robustness():
    f = _frame()
    rows = pd.DataFrame(E.summary_rows(f, "heldout_validation", "x"))
    assert set(rows.task) == {"building", "floor", "coordinate", "building_and_floor"}
    assert rows.set_index("task").loc["building_and_floor", "accuracy"] == 0.25
    r = E.robustness_by_group(f, "PHONEID", "heldout_validation", seen_in_training={1})
    assert list(r.group_seen_in_internal_train) == [True, False]


def test_paired_comparison_counts():
    a = _frame()
    b = a.copy()
    b["FLOOR_CORRECT"] = True
    b["BUILDING_FLOOR_CORRECT"] = b.BUILDING_CORRECT
    b["EUCLIDEAN_ERROR"] = a.EUCLIDEAN_ERROR + 1
    out = E.paired_comparison(a, b, n_boot=50)
    assert out["floor"]["dl_wrong_aml_correct"] == 2
    assert out["coordinate"]["mean_diff_dl_minus_aml"] == pytest.approx(-1)
