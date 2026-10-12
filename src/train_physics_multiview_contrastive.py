"""Physics-guided multi-view contrastive identification.

The handcrafted RF descriptors are treated as four physical views rather than
one flat vector. Contrastive learning acts inside the physical representation
formation, before reliability-gated fusion with the frozen deep embedding.
Only development splits and known samples are accessible in this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rf_physics_feature_probe import extract_all
from spectral_conformal_benchmark import deterministic
from train_causal_ablation import build_model, extract, seed_everything
from train_dual_head_multiknown import predict
from train_multiknown17_baseline import make_loader
from train_representation import dataset_fingerprint, supervised_contrastive_loss


VIEW_INDICES = (
    np.r_[0:24, 83:115],   # global statistics and amplitude distribution
    np.r_[24:83, 115:147], # phase, complex moments, correlation, phase histogram
    np.arange(147, 243),   # centered spectral envelope
    np.arange(243, 339),   # blockwise temporal envelope mean/std
)
VIEW_NAMES = ("amplitude_statistics", "phase_correlation", "spectral", "temporal")
EXPECTED_PHYSICS_DIM = 339


class ViewEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, value):
        return self.net(value)


class PhysicsMultiViewHead(nn.Module):
    """Group encoding -> shared contrastive space -> reliability fusion."""

    def __init__(self, deep_dim: int, classes: int):
        super().__init__()
        self.encoders = nn.ModuleList([ViewEncoder(len(index)) for index in VIEW_INDICES])
        self.projectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32)) for _ in VIEW_INDICES]
        )
        self.reliability = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.complementary_fusion = nn.Sequential(
            nn.Linear(64 * len(VIEW_INDICES), 64), nn.LayerNorm(64), nn.GELU()
        )
        self.classifier = nn.Linear(deep_dim + 64, classes)
        for position, index in enumerate(VIEW_INDICES):
            self.register_buffer(f"view_index_{position}", torch.from_numpy(index).long())

    def encode_physics(self, physics):
        latent = [
            encoder(physics.index_select(1, getattr(self, f"view_index_{position}")))
            for position, encoder in enumerate(self.encoders)
        ]
        stacked = torch.stack(latent, dim=1)
        weights = torch.softmax(self.reliability(stacked).squeeze(-1), dim=1)
        gated = (weights.unsqueeze(-1) * stacked).sum(1)
        complementary = self.complementary_fusion(torch.cat(latent, dim=1))
        physical = F.normalize(gated + complementary, dim=1)
        projected = [projector(value) for projector, value in zip(self.projectors, latent)]
        return physical, projected, weights

    def forward_parts(self, deep, physics):
        physical, projected, weights = self.encode_physics(physics)
        logits = self.classifier(torch.cat([F.normalize(deep, dim=1), physical], dim=1))
        return logits, projected, weights

    def forward(self, deep, physics):
        return self.forward_parts(deep, physics)[0]

    def deployed_parameters(self):
        modules = [self.encoders, self.reliability, self.complementary_fusion, self.classifier]
        return sum(parameter.numel() for module in modules for parameter in module.parameters())


@torch.no_grad()
def evaluate(head, deep, physics, target, batch_size, device):
    logits = predict(head, deep, physics, batch_size, device)
    loss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(target).long())
    return float(loss.item()), float((logits.argmax(1) == target).mean())


@torch.no_grad()
def mean_gate(head, deep, physics, batch_size, device):
    head.eval()
    values = []
    for start in range(0, len(deep), batch_size):
        d = torch.from_numpy(deep[start : start + batch_size]).to(device)
        p = torch.from_numpy(physics[start : start + batch_size]).to(device)
        _, _, weights = head.forward_parts(d, p)
        values.append(weights.cpu().numpy())
    return np.concatenate(values).mean(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-contrast", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="train/select the head without constructing any role==2 indices",
    )
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    root = Path(model_args.dataset_root)
    manifest = json.loads((root / f"split_{model_args.split}.json").read_text())
    allowed_status = {"development", "FROZEN_BEFORE_OPEN_SET_TEST"}
    if manifest["status"] not in allowed_status or model_args.split not in (1, 2, 3):
        raise ValueError("physics multi-view training requires a pre-registered split")
    known_classes = list(map(int, checkpoint.get("active_known_classes", manifest["known_classes"])))
    if not set(known_classes).issubset(set(map(int, manifest["known_classes"]))):
        raise ValueError("checkpoint active known classes do not match outer split")
    mapping = {class_id: position for position, class_id in enumerate(known_classes)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(root / "Y_global.npy")
    role = np.load(root / "sample_role.npy")
    indices = {
        "train": np.flatnonzero(np.isin(y, known_classes) & (role == 0)),
        "validation": np.flatnonzero(np.isin(y, known_classes) & (role == 1)),
    }
    if not args.fit_only:
        indices["known_test"] = np.flatnonzero(np.isin(y, known_classes) & (role == 2))

    seed_everything(model_args.seed + 20_000)
    train_loader = make_loader(x, y, indices["train"], mapping, model_args.batch_size, True)
    val_loader = make_loader(x, y, indices["validation"], mapping, args.batch_size, False)
    fingerprints = {
        "known_train": dataset_fingerprint(train_loader),
        "known_validation": dataset_fingerprint(val_loader),
    }
    known_loader = None
    if not args.fit_only:
        known_loader = make_loader(x, y, indices["known_test"], mapping, args.batch_size, False)
        fingerprints["known_test"] = dataset_fingerprint(known_loader)
    checkpoint_fingerprints = checkpoint.get("split_fingerprints", {})
    if any(checkpoint_fingerprints.get(name) != value for name, value in fingerprints.items()):
        raise RuntimeError("checkpoint/data split fingerprint mismatch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(model_args.seed)
    backbone = build_model(model_args, device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    train_e, _, train_y = extract(backbone, deterministic(train_loader, args.batch_size), device)
    val_e, _, val_y = extract(backbone, val_loader, device)
    known_e = known_y = None
    if not args.fit_only:
        known_e, _, known_y = extract(backbone, known_loader, device)

    physics_train = extract_all(x[indices["train"]], 64)
    physics_val = extract_all(x[indices["validation"]], 64)
    physics_known = None if args.fit_only else extract_all(x[indices["known_test"]], 64)
    if physics_train.shape[1] != EXPECTED_PHYSICS_DIM:
        raise RuntimeError("unexpected physics feature dimension")
    physics_mean = physics_train.mean(0, keepdims=True)
    physics_std = np.maximum(physics_train.std(0, keepdims=True), 1e-5)
    standardize = lambda value: ((value - physics_mean) / physics_std).astype(np.float32)
    physics_train, physics_val = map(standardize, (physics_train, physics_val))
    if physics_known is not None:
        physics_known = standardize(physics_known)
    train_e = train_e.astype(np.float32)
    val_e = val_e.astype(np.float32)
    if not args.fit_only:
        known_e = known_e.astype(np.float32)

    seed_everything(model_args.seed + 70_000)
    head = PhysicsMultiViewHead(train_e.shape[1], len(known_classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )
    generator = torch.Generator().manual_seed(model_args.seed + 80_000)
    feature_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_e),
            torch.from_numpy(physics_train),
            torch.from_numpy(train_y).long(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_loss = float("inf")
    stale = 0
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        head.train()
        totals = {"loss": 0.0, "ce": 0.0, "contrast": 0.0, "correct": 0, "n": 0}
        for deep_batch, physics_batch, target in feature_loader:
            deep_batch = deep_batch.to(device)
            physics_batch = physics_batch.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, projected, _ = head.forward_parts(deep_batch, physics_batch)
            contrastive_embedding = torch.cat(projected, dim=0)
            contrastive_target = target.repeat(len(projected))
            contrast = supervised_contrastive_loss(
                contrastive_embedding, contrastive_target, args.temperature
            )
            ce = F.cross_entropy(logits, target, label_smoothing=0.05)
            loss = ce + args.lambda_contrast * contrast
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            n = len(target)
            totals["loss"] += float(loss.item()) * n
            totals["ce"] += float(ce.item()) * n
            totals["contrast"] += float(contrast.item()) * n
            totals["correct"] += int((logits.argmax(1) == target).sum().item())
            totals["n"] += n

        val_loss, val_acc = evaluate(head, val_e, physics_val, val_y, args.batch_size, device)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / totals["n"],
            "train_ce": totals["ce"] / totals["n"],
            "train_contrast": totals["contrast"] / totals["n"],
            "train_acc": totals["correct"] / totals["n"],
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            stale = 0
            torch.save(
                {
                    "head": head.state_dict(),
                    "physics_mean": physics_mean,
                    "physics_std": physics_std,
                    "base_checkpoint": str(Path(args.checkpoint).resolve()),
                    "split_fingerprints": fingerprints,
                    "runner_sha256": runner_sha256,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "view_names": VIEW_NAMES,
                    "lambda_contrast": args.lambda_contrast,
                    "temperature": args.temperature,
                    "active_known_classes": known_classes,
                    "fit_only": bool(args.fit_only),
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    selected = torch.load(best_path, map_location=device)
    head.load_state_dict(selected["head"])
    if args.fit_only:
        fit_result = {
            "method": "physics_guided_multiview_contrastive_loeo_fit_only",
            "split": int(model_args.split),
            "seed": int(model_args.seed),
            "active_known_classes": known_classes,
            "split_fingerprints": fingerprints,
            "selected_epoch": int(selected["epoch"]),
            "selected_val_loss": float(selected["val_loss"]),
            "selected_val_acc": float(selected["val_acc"]),
            "role2_indices_constructed": False,
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "train_seconds": time.time() - start,
            "protocol": {
                "training": "active known train only; frozen active-known deep encoder",
                "checkpoint_selection": "active known validation CE only",
                "target_rogue_materialized": False,
            },
        }
        (output_dir / "fit_result.json").write_text(
            json.dumps(fit_result, indent=2), encoding="utf-8"
        )
        print("FINAL_FIT_ONLY", json.dumps(fit_result, indent=2), flush=True)
        return
    test_loss, test_acc = evaluate(head, known_e, physics_known, known_y, args.batch_size, device)
    gate = mean_gate(head, known_e, physics_known, args.batch_size, device)
    output = {
        "method": "physics_guided_multiview_contrastive_identification_v1",
        "split": model_args.split,
        "seed": model_args.seed,
        "known_classes": known_classes,
        "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256,
        "selected_epoch": selected["epoch"],
        "selected_val_loss": selected["val_loss"],
        "selected_val_acc": selected["val_acc"],
        "known_test_loss": test_loss,
        "known_closed_acc": test_acc,
        "mean_view_reliability": dict(zip(VIEW_NAMES, map(float, gate))),
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "deployed_parameters": head.deployed_parameters(),
        "train_seconds": time.time() - start,
        "protocol": {
            "training": "known train role only; frozen deep encoder",
            "checkpoint_selection": "known validation CE only",
            "target_rogue_materialized": False,
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "rejection_spaces_modified": False,
            "projection_heads_deployed": False,
            "contrastive_location": "inside physical multi-view representation formation",
        },
        "history": history,
    }
    (output_dir / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
