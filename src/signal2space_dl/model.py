"""Shared-trunk multitask network for building, floor and coordinate prediction.

Architecture
------------
::

    1,046 predictors
        -> input dropout
        -> [Linear -> BatchNorm1d -> ReLU -> Dropout] x len(hidden_sizes)   (shared trunk)
        |-> building head:   Linear(h, head_hidden) -> ReLU -> Linear(head_hidden, n_buildings)
        |-> floor head:      Linear(h, head_hidden) -> ReLU -> Linear(head_hidden, n_floors)
        |-> coordinate head: Linear(h, head_hidden) -> ReLU -> Linear(head_hidden, 2)

The classification heads output logits; the coordinate head outputs
standardized (train-fitted) LONGITUDE/LATITUDE values.

Loss
----
``L = w_b * CE(building) + w_f * CE(floor) + w_c * SmoothL1_beta(coords_scaled)``

with ``beta`` = ``TrainConfig.coord_beta`` (default 0.1 standardized units).

Single-task ablations use the same trunk with only one head enabled and only
that head's loss term.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ALL_TASKS = ("building", "floor", "coord")


def _head(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features))


class MultiTaskNet(nn.Module):
    """Shared MLP trunk with one head per enabled task."""

    def __init__(self, n_features: int, n_buildings: int, n_floors: int,
                 hidden_sizes=(512, 256), head_hidden: int = 128, dropout: float = 0.3,
                 input_dropout: float = 0.1, tasks=ALL_TASKS):
        super().__init__()
        unknown = set(tasks) - set(ALL_TASKS)
        if unknown or not tasks:
            raise ValueError(f"tasks must be a non-empty subset of {ALL_TASKS}; got {tasks}")
        self.tasks = tuple(tasks)
        layers: list[nn.Module] = [nn.Dropout(input_dropout)]
        width = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict()
        if "building" in self.tasks:
            self.heads["building"] = _head(width, head_hidden, n_buildings)
        if "floor" in self.tasks:
            self.heads["floor"] = _head(width, head_hidden, n_floors)
        if "coord" in self.tasks:
            self.heads["coord"] = _head(width, head_hidden, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        return {name: head(z) for name, head in self.heads.items()}


def multitask_loss(outputs: dict[str, torch.Tensor], building: torch.Tensor, floor: torch.Tensor,
                   coords_scaled: torch.Tensor, w_building: float = 1.0, w_floor: float = 1.0,
                   w_coord: float = 1.0, coord_beta: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of the enabled task losses; also returns the unweighted terms."""
    total = torch.zeros((), device=next(iter(outputs.values())).device)
    parts: dict[str, float] = {}
    if "building" in outputs:
        lb = F.cross_entropy(outputs["building"], building)
        total = total + w_building * lb
        parts["loss_building"] = float(lb.detach())
    if "floor" in outputs:
        lf = F.cross_entropy(outputs["floor"], floor)
        total = total + w_floor * lf
        parts["loss_floor"] = float(lf.detach())
    if "coord" in outputs:
        lc = F.smooth_l1_loss(outputs["coord"], coords_scaled, beta=coord_beta)
        total = total + w_coord * lc
        parts["loss_coord"] = float(lc.detach())
    parts["loss_total"] = float(total.detach())
    return total, parts


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
