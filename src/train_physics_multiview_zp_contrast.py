"""Variant that applies multi-view SupCon directly to deployed z_p as well."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import train_physics_multiview_contrastive as base


class DirectZPContrastHead(base.PhysicsMultiViewHead):
    """Include final physical embedding as a fifth contrastive view."""

    def __init__(self, deep_dim: int, classes: int):
        super().__init__(deep_dim, classes)
        self.zp_projector = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32)
        )

    def forward_parts(self, deep: torch.Tensor, physics: torch.Tensor):
        physical, projected, weights = self.encode_physics(physics)
        logits = self.classifier(
            torch.cat([F.normalize(deep, dim=1), physical], dim=1)
        )
        return logits, [*projected, self.zp_projector(physical)], weights


base.PhysicsMultiViewHead = DirectZPContrastHead
base.__file__ = __file__


if __name__ == "__main__":
    base.main()
