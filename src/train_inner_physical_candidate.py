#!/usr/bin/env python3
"""Train one pre-registered physical-head candidate for inner LOEO selection.

The caller must pass ``--fit-only``.  Only the learned physical branch changes;
the independent pruned-304 explicit-RF evidence channel remains unchanged.
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
from rf_descriptor_pruning import KEEP_INDICES, PRUNED_DIMENSION, PRUNED_GROUPS, manifest


VIEW_NAMES = ("amplitude_statistics", "phase_correlation", "spectral", "temporal")
ALPHAS = {"0": 0.0, "025": 0.25, "050": 0.5, "100": 1.0}
CANDIDATES = tuple(
    f"{views}_alpha{suffix}"
    for views in ("four", "nophase", "apst", "ast", "st", "at", "as")
    for suffix in ALPHAS
)

ACTIVE_GROUPS = PRUNED_GROUPS
ACTIVE_NAMES = VIEW_NAMES
RESIDUAL_ALPHA = 1.0


class InnerPhysicalCandidateHead(nn.Module):
    def __init__(self, deep_dim: int, classes: int):
        super().__init__()
        self.encoders = nn.ModuleList([base.ViewEncoder(len(index)) for index in ACTIVE_GROUPS])
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
            for _ in ACTIVE_GROUPS
        )
        self.reliability = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.complementary_fusion = nn.Sequential(
            nn.Linear(64 * len(ACTIVE_GROUPS), 64), nn.LayerNorm(64), nn.GELU()
        )
        self.zp_projector = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
        self.classifier = nn.Linear(deep_dim + 64, classes)
        for position, index in enumerate(ACTIVE_GROUPS):
            self.register_buffer(f"view_index_{position}", torch.as_tensor(index, dtype=torch.long))

    def encode_physics(self, physics):
        latent = [
            encoder(physics.index_select(1, getattr(self, f"view_index_{position}")))
            for position, encoder in enumerate(self.encoders)
        ]
        stacked = torch.stack(latent, dim=1)
        weights = torch.softmax(self.reliability(stacked).squeeze(-1), dim=1)
        gated = (weights.unsqueeze(-1) * stacked).sum(1)
        complementary = self.complementary_fusion(torch.cat(latent, dim=1))
        if RESIDUAL_ALPHA == 0.0:
            physical = complementary
        else:
            physical = complementary + RESIDUAL_ALPHA * gated
        physical = F.normalize(physical, dim=1)
        projected = [projector(value) for projector, value in zip(self.projectors, latent)]
        return physical, projected, weights

    def forward_parts(self, deep, physics):
        physical, projected, weights = self.encode_physics(physics)
        logits = self.classifier(torch.cat([F.normalize(deep, dim=1), physical], dim=1))
        return logits, [*projected, self.zp_projector(physical)], weights

    def forward(self, deep, physics):
        return self.forward_parts(deep, physics)[0]

    def deployed_parameters(self):
        modules = [self.encoders, self.complementary_fusion, self.classifier]
        if RESIDUAL_ALPHA > 0:
            modules.append(self.reliability)
        return sum(parameter.numel() for module in modules for parameter in module.parameters())


def configure(candidate: str) -> None:
    global ACTIVE_GROUPS, ACTIVE_NAMES, RESIDUAL_ALPHA
    views, alpha_name = candidate.split("_alpha")
    keep_by_views = {
        "four": (0, 1, 2, 3),
        "apst": (0, 1, 2, 3),
        "nophase": (0, 2, 3),
        "ast": (0, 2, 3),
        "st": (2, 3),
        "at": (0, 3),
        "as": (0, 2),
    }
    keep = keep_by_views[views]
    ACTIVE_GROUPS = tuple(PRUNED_GROUPS[position] for position in keep)
    ACTIVE_NAMES = tuple(VIEW_NAMES[position] for position in keep)
    RESIDUAL_ALPHA = ALPHAS[alpha_name]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate", required=True, choices=CANDIDATES)
    custom, remaining = parser.parse_known_args()
    if "--fit-only" not in remaining:
        raise RuntimeError("inner candidate training requires --fit-only")
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output-dir", required=True)
    output, _ = output_parser.parse_known_args(remaining)
    configure(custom.candidate)

    original_extract = base.extract_all
    base.extract_all = lambda values, batch: original_extract(values, batch)[:, KEEP_INDICES]
    base.EXPECTED_PHYSICS_DIM = PRUNED_DIMENSION
    base.VIEW_NAMES = ACTIVE_NAMES
    base.PhysicsMultiViewHead = InnerPhysicalCandidateHead
    base.__file__ = __file__
    sys.argv = [sys.argv[0], *remaining]
    output_dir = Path(output.output_dir)
    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        base.main()

    checkpoint = torch.load(best_path, map_location="cpu")
    checkpoint.update(
        {
            "head_arch": "inner_physical_arch_selection_v1",
            "candidate": custom.candidate,
            "descriptor": "pruned304",
            "descriptor_dimension": PRUNED_DIMENSION,
            "descriptor_keep_indices": KEEP_INDICES,
            "descriptor_manifest": manifest(),
            "view_indices": ACTIVE_GROUPS,
            "view_names": ACTIVE_NAMES,
            "residual_alpha": RESIDUAL_ALPHA,
            "physical_path": "complementary_only" if RESIDUAL_ALPHA == 0 else "complementary_plus_scaled_gated",
            "contrast_mode": "remaining_views_plus_q_z",
            "contrastive_views": [*ACTIVE_NAMES, "q_z"],
            "ablation_scope": "learned_physical_branch_only",
            "explicit_raw_rf_evidence": "unchanged_full_pruned304",
        }
    )
    torch.save(checkpoint, best_path)
    fit_result = output_dir / "fit_result.json"
    if fit_result.exists():
        payload = json.loads(fit_result.read_text(encoding="utf-8"))
        payload.update(
            {
                "candidate": custom.candidate,
                "view_names": list(ACTIVE_NAMES),
                "residual_alpha": RESIDUAL_ALPHA,
                "outer_role2_access": False,
            }
        )
        fit_result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
