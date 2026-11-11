"""Shared-trunk multitask MLP and its loss.

Architecture (default TrainConfig):

    x (1,040) -> Dropout(0.2)                          # input dropout = simulated missing APs
      -> [Linear -> BatchNorm -> ReLU -> Dropout(0.3)] x (512, 256)   # shared trunk
      -> building head : Linear(256,128) -> ReLU -> Linear(128, 3)    # logits
      -> floor head    : Linear(256,128) -> ReLU -> Linear(128, 5)    # logits
      -> coord head    : Linear(256,128) -> ReLU -> Linear(128, 2)    # normalised (lon, lat)

A single-task ablation is the same network with only one head enabled.
"""
from __future__ import annotations

import torch
from torch import nn


class MultitaskMLP(nn.Module):
    def __init__(self, n_inputs: int, n_buildings: int, n_floors: int, tasks: tuple[str, ...],
                 hidden_sizes=(512, 256), head_hidden: int = 128,
                 input_dropout: float = 0.2, dropout: float = 0.3):
        super().__init__()
        self.tasks = tuple(tasks)
        layers: list[nn.Module] = [nn.Dropout(input_dropout)]
        width = n_inputs
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.trunk = nn.Sequential(*layers)
        out_dims = {"building": n_buildings, "floor": n_floors, "coordinate": 2}
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(width, head_hidden), nn.ReLU(), nn.Linear(head_hidden, out_dims[t]))
            for t in self.tasks
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        return {t: head(z) for t, head in self.heads.items()}


class MultitaskLoss(nn.Module):
    """L = wb * CE(building) + wf * CE(floor) + wc * Huber(coord).

    Coordinates are centred on the internal-train mean and divided by a fixed
    ``coord_scale`` (10 dataset units), so a Huber delta of 1 switches from quadratic to
    linear at a 10-unit error. That puts the coordinate term on the same order of
    magnitude as the cross-entropies (instead of the near-zero values produced by
    z-scoring a campus-wide coordinate range), keeps the two axes isotropic, and bounds
    gradients from large early errors. Weights are fixed (1, 1, 1) rather than tuned.
    """

    def __init__(self, tasks, weights: dict, huber_delta: float = 1.0,
                 class_weights: dict[str, torch.Tensor] | None = None):
        super().__init__()
        self.tasks = tuple(tasks)
        self.weights = {t: float(weights[t]) for t in self.tasks}
        cw = class_weights or {}
        self.ce = nn.ModuleDict({t: nn.CrossEntropyLoss(weight=cw.get(t)) for t in self.tasks
                                 if t in ("building", "floor")})
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict[str, float]]:
        parts = {}
        for t in self.tasks:
            if t == "coordinate":
                parts[t] = self.huber(outputs[t], targets[t])
            else:
                parts[t] = self.ce[t](outputs[t], targets[t])
        total = sum(self.weights[t] * parts[t] for t in self.tasks)
        return total, {t: float(v.detach()) for t, v in parts.items()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
