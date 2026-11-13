"""Tests for signal2space_dl.model and signal2space_dl.train."""

import dataclasses

import numpy as np
import pytest
import torch

from signal2space_dl import config as C
from signal2space_dl import data as D
from signal2space_dl.model import MultiTaskNet, multitask_loss
from signal2space_dl.train import load_model, predict, save_model, train_model


def test_multitask_heads_and_shapes():
    net = MultiTaskNet(1046, 3, 5, hidden_sizes=(32, 16), head_hidden=8)
    out = net(torch.randn(4, 1046))
    assert out["building"].shape == (4, 3)
    assert out["floor"].shape == (4, 5)
    assert out["coord"].shape == (4, 2)


def test_single_task_has_one_head():
    net = MultiTaskNet(1046, 3, 5, hidden_sizes=(16,), tasks=("floor",))
    assert set(net(torch.randn(3, 1046))) == {"floor"}
    with pytest.raises(ValueError):
        MultiTaskNet(1046, 3, 5, tasks=("altitude",))


def test_loss_uses_weights():
    torch.manual_seed(0)
    out = {"building": torch.randn(6, 3), "floor": torch.randn(6, 5), "coord": torch.randn(6, 2)}
    b, f, c = torch.randint(0, 3, (6,)), torch.randint(0, 5, (6,)), torch.randn(6, 2)
    total1, parts = multitask_loss(out, b, f, c)
    total2, _ = multitask_loss(out, b, f, c, w_floor=2.0)
    assert total2 - total1 == pytest.approx(parts["loss_floor"], rel=1e-5)


def test_training_runs_restores_best_and_roundtrips(tmp_path, train_frame, heldout_frame):
    prep = D.fit_preprocessor(train_frame)
    tr, va = prep.transform(train_frame), prep.transform(heldout_frame)
    cfg = dataclasses.replace(C.TrainConfig(), hidden_sizes=(32,), head_hidden=8, max_epochs=4, patience=2,
                              batch_size=32)
    res = train_model(cfg, tr, va, prep, coord_norm=100.0)
    assert 1 <= res.best_epoch <= 4
    assert res.best_score == pytest.approx(res.history.val_selection_score.min())
    p1 = predict(res.model, va, prep)
    save_model(res, prep, tmp_path / "m.pt")
    model, cfg2, prep2 = load_model(tmp_path / "m.pt")
    p2 = predict(model, va, prep2)
    np.testing.assert_allclose(p1["coords"], p2["coords"], rtol=1e-6)
    assert cfg2.hidden_sizes == (32,)


def test_training_is_seed_deterministic(train_frame, heldout_frame):
    prep = D.fit_preprocessor(train_frame)
    tr, va = prep.transform(train_frame), prep.transform(heldout_frame)
    cfg = dataclasses.replace(C.TrainConfig(), hidden_sizes=(16,), head_hidden=8, max_epochs=2, patience=2)
    a = predict(train_model(cfg, tr, va, prep, 100.0).model, va, prep)["coords"]
    b = predict(train_model(cfg, tr, va, prep, 100.0).model, va, prep)["coords"]
    np.testing.assert_allclose(a, b)
