"""Flat, physical-group and random-group controls for fixed RF descriptors.

This wrapper reuses the established optimizer, losses, data protocol and head
layers.  It changes only the descriptor columns and/or the grouping assignment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import train_physics_multiview_contrastive as base
from rf_descriptor_pruning import KEEP_INDICES, PRUNED_DIMENSION, PRUNED_GROUPS, PRUNED_GROUP_SIZES, manifest
from train_physics_multiview_zp_contrast import DirectZPContrastHead


FULL_GROUP_SIZES = (56, 91, 96, 96)
DEFAULT_RANDOM_PARTITION_SEED = 20260907
ACTIVE_INPUT_DIM = 339
ACTIVE_GROUPS = base.VIEW_INDICES
ACTIVE_GROUPING_SEED = None


def random_indices(total_dim: int, group_sizes: tuple[int, ...], seed: int):
    if sum(group_sizes) != total_dim:
        raise ValueError((total_dim, group_sizes))
    permutation = np.random.default_rng(seed).permutation(total_dim)
    boundaries = np.cumsum((0,) + group_sizes)
    return tuple(permutation[boundaries[i] : boundaries[i + 1]] for i in range(4))


class FlatRFHead(nn.Module):
    """Established flat encoder with a training-only z_p projector."""

    def __init__(self, deep_dim: int, classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ACTIVE_INPUT_DIM, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.zp_projector = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
        self.classifier = nn.Linear(deep_dim + 64, classes)

    def encode_physics(self, physics):
        physical = F.normalize(self.encoder(physics), dim=1)
        projected = [self.zp_projector(physical)]
        display_weight = torch.ones(len(physics), 1, device=physics.device)
        return physical, projected, display_weight

    def forward_parts(self, deep, physics):
        physical, projected, weights = self.encode_physics(physics)
        logits = self.classifier(torch.cat([F.normalize(deep, dim=1), physical], dim=1))
        return logits, projected, weights

    def forward(self, deep, physics):
        return self.forward_parts(deep, physics)[0]

    def deployed_parameters(self):
        return sum(parameter.numel() for module in (self.encoder, self.classifier) for parameter in module.parameters())


class GroupedRFHead(DirectZPContrastHead):
    """Established four-view head using the active fixed grouping."""

    def __init__(self, deep_dim: int, classes: int):
        original = base.VIEW_INDICES
        try:
            base.VIEW_INDICES = ACTIVE_GROUPS
            super().__init__(deep_dim, classes)
        finally:
            base.VIEW_INDICES = original


def patch_metadata(output_dir: Path, structure: str, descriptor: str, grouping_seed: int | None):
    descriptor_manifest = manifest() if descriptor == "pruned304" else None
    checkpoint_path = output_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.update(
        {
            "head_arch": structure,
            "descriptor": descriptor,
            "descriptor_dimension": ACTIVE_INPUT_DIM,
            "descriptor_keep_indices": KEEP_INDICES if descriptor == "pruned304" else None,
            "descriptor_manifest": descriptor_manifest,
            "view_indices": ACTIVE_GROUPS if structure != "flat" else None,
            "grouping_seed": grouping_seed,
        }
    )
    torch.save(checkpoint, checkpoint_path)

    result_path = output_dir / "result.json"
    if not result_path.exists():
        result_path = output_dir / "fit_result.json"
    result = json.loads(result_path.read_text())
    result.update(
        {
            "method": f"{descriptor}_{structure}_same_protocol",
            "head_arch": structure,
            "descriptor": descriptor,
            "descriptor_dimension": ACTIVE_INPUT_DIM,
            "descriptor_manifest": descriptor_manifest,
            "group_sizes": [len(value) for value in ACTIVE_GROUPS] if structure != "flat" else [ACTIVE_INPUT_DIM],
            "grouping_seed": grouping_seed,
        }
    )
    result.setdefault("protocol", {}).update(
        {
            "diagnostic_only": True,
            "network_layers_changed": False,
            "same_base_checkpoint_as_physical_4view": True,
            "same_optimizer_schedule_and_loss_hyperparameters": True,
        }
    )
    if structure == "flat" and "mean_view_reliability" in result:
        result["mean_view_reliability"] = {"flat": 1.0}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def head_from_checkpoint(deep_dim: int, classes: int, checkpoint: dict):
    """Reconstruct one of these control heads without changing its layers."""
    global ACTIVE_INPUT_DIM, ACTIVE_GROUPS, ACTIVE_GROUPING_SEED
    ACTIVE_INPUT_DIM = int(checkpoint.get("descriptor_dimension", 339))
    stored_groups = checkpoint.get("view_indices")
    if stored_groups is not None:
        ACTIVE_GROUPS = tuple(np.asarray(value, dtype=np.int64) for value in stored_groups)
    ACTIVE_GROUPING_SEED = checkpoint.get("grouping_seed")
    if checkpoint.get("head_arch") == "flat":
        return FlatRFHead(deep_dim, classes)
    return GroupedRFHead(deep_dim, classes)


def main():
    global ACTIVE_INPUT_DIM, ACTIVE_GROUPS, ACTIVE_GROUPING_SEED
    control = argparse.ArgumentParser(add_help=False)
    control.add_argument("--structure", choices=["flat", "grouped", "random4view"], required=True)
    control.add_argument("--descriptor", choices=["full339", "pruned304"], default="full339")
    control.add_argument("--grouping-seed", type=int, default=DEFAULT_RANDOM_PARTITION_SEED)
    known, remaining = control.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output-dir", required=True)
    output_known, _ = output_parser.parse_known_args(remaining)

    original_extract = base.extract_all
    if known.descriptor == "pruned304":
        ACTIVE_INPUT_DIM = PRUNED_DIMENSION
        physical_groups = PRUNED_GROUPS

        def active_extract(values, batch_size):
            return original_extract(values, batch_size)[:, KEEP_INDICES]

        base.extract_all = active_extract
    else:
        ACTIVE_INPUT_DIM = 339
        physical_groups = base.VIEW_INDICES

    if known.structure == "random4view":
        sizes = PRUNED_GROUP_SIZES if known.descriptor == "pruned304" else FULL_GROUP_SIZES
        ACTIVE_GROUPS = random_indices(ACTIVE_INPUT_DIM, sizes, known.grouping_seed)
        ACTIVE_GROUPING_SEED = known.grouping_seed
    else:
        ACTIVE_GROUPS = physical_groups
        ACTIVE_GROUPING_SEED = None

    base.EXPECTED_PHYSICS_DIM = ACTIVE_INPUT_DIM
    base.PhysicsMultiViewHead = FlatRFHead if known.structure == "flat" else GroupedRFHead
    base.__file__ = __file__
    sys.argv = [sys.argv[0], *remaining]
    base.main()
    patch_metadata(Path(output_known.output_dir), known.structure, known.descriptor, ACTIVE_GROUPING_SEED)


if __name__ == "__main__":
    main()
