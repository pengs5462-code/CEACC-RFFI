"""Train a small physics-aware identification head with frozen OSR embedding.

Unknown rejection remains the known-only calibrated deep Mondrian score.  The
new head is trained solely to improve known-device identification from frozen
deep embeddings and fixed RF statistics.  Development splits only.
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

from conformal_benchmark import MondrianMahalanobis
from recalibrate_multiknown17 import leave_one_out_validation_p
from rf_physics_feature_probe import extract_all
from score_benchmark import metric_row
from spectral_conformal_benchmark import deterministic
from train_causal_ablation import build_model, extract, seed_everything
from train_multiknown17_baseline import make_loader
from train_representation import dataset_fingerprint


class PhysicsIdentificationHead(nn.Module):
    def __init__(self, deep_dim: int, physics_dim: int, classes: int) -> None:
        super().__init__()
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.classifier = nn.Linear(deep_dim + 64, classes)

    def forward(self, deep: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        deep = F.normalize(deep, dim=1)
        physical = F.normalize(self.physics_encoder(physics), dim=1)
        return self.classifier(torch.cat([deep, physical], dim=1))


@torch.no_grad()
def predict(head, deep, physics, batch_size, device):
    head.eval()
    output = []
    for start in range(0, len(deep), batch_size):
        d = torch.from_numpy(deep[start : start + batch_size]).to(device)
        p = torch.from_numpy(physics[start : start + batch_size]).to(device)
        output.append(head(d, p).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--frozen-manifest", default=None)
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    base_checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**base_checkpoint["args"])
    root = Path(model_args.dataset_root)
    manifest = json.loads((root / f"split_{model_args.split}.json").read_text())
    if manifest["status"] != "development":
        if manifest["status"] != "reserved_untouched" or not args.frozen_manifest:
            raise ValueError("reserved evaluation requires an explicit frozen manifest")
        frozen = json.loads(Path(args.frozen_manifest).read_text())
        if frozen.get("method_id") != "dual_head_physics_fisher_v1" or frozen.get("status") != "frozen_before_reserved_evaluation":
            raise ValueError("invalid frozen method manifest")
    known_classes = list(map(int, manifest["known_classes"]))
    rogue_classes = list(map(int, manifest["rogue_classes"]))
    mapping = {class_id: position for position, class_id in enumerate(known_classes)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(root / "Y_global.npy")
    role = np.load(root / "sample_role.npy")
    indices = {
        "train": np.flatnonzero(np.isin(y, known_classes) & (role == 0)),
        "validation": np.flatnonzero(np.isin(y, known_classes) & (role == 1)),
        "known_test": np.flatnonzero(np.isin(y, known_classes) & (role == 2)),
        "rogue_test": np.flatnonzero(np.isin(y, rogue_classes) & (role == 2)),
    }

    seed_everything(model_args.seed + 20_000)
    train_loader = make_loader(x, y, indices["train"], mapping, model_args.batch_size, True)
    val_loader = make_loader(x, y, indices["validation"], mapping, args.batch_size, False)
    known_loader = make_loader(x, y, indices["known_test"], mapping, args.batch_size, False)
    fingerprints = {
        "known_train": dataset_fingerprint(train_loader),
        "known_validation": dataset_fingerprint(val_loader),
        "known_test": dataset_fingerprint(known_loader),
    }
    if base_checkpoint.get("split_fingerprints") != fingerprints:
        raise RuntimeError("checkpoint/data split fingerprint mismatch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(model_args.seed)
    backbone = build_model(model_args, device)
    backbone.load_state_dict(base_checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    train_e, _, train_y = extract(backbone, deterministic(train_loader, args.batch_size), device)
    val_e, _, val_y = extract(backbone, val_loader, device)
    known_e, _, known_y = extract(backbone, known_loader, device)

    physics_train = extract_all(x[indices["train"]], 64)
    physics_val = extract_all(x[indices["validation"]], 64)
    physics_known = extract_all(x[indices["known_test"]], 64)
    physics_mean = physics_train.mean(0, keepdims=True)
    physics_std = np.maximum(physics_train.std(0, keepdims=True), 1e-5)

    def standardize(value):
        return ((value - physics_mean) / physics_std).astype(np.float32)

    physics_train = standardize(physics_train)
    physics_val = standardize(physics_val)
    physics_known = standardize(physics_known)
    train_e = train_e.astype(np.float32)
    val_e = val_e.astype(np.float32)
    known_e = known_e.astype(np.float32)

    seed_everything(model_args.seed + 50_000)
    head = PhysicsIdentificationHead(
        train_e.shape[1], physics_train.shape[1], len(known_classes)
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )
    generator = torch.Generator().manual_seed(model_args.seed + 60_000)
    mlp_loader = DataLoader(
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
    start_time = time.time()
    for epoch in range(args.epochs):
        head.train()
        total_loss = total_correct = total = 0
        for deep_batch, physics_batch, target in mlp_loader:
            deep_batch = deep_batch.to(device)
            physics_batch = physics_batch.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(deep_batch, physics_batch)
            loss = F.cross_entropy(logits, target, label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
            total_correct += int((logits.argmax(1) == target).sum())
            total += len(target)
        val_logits = predict(head, val_e, physics_val, args.batch_size, device)
        val_loss = float(
            F.cross_entropy(
                torch.from_numpy(val_logits), torch.from_numpy(val_y).long()
            ).item()
        )
        val_acc = float((val_logits.argmax(1) == val_y).mean())
        scheduler.step(val_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "train_acc": total_correct / total,
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
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    selected = torch.load(best_path, map_location=device)
    head.load_state_dict(selected["head"])
    known_logits = predict(head, known_e, physics_known, args.batch_size, device)
    known_correct = known_logits.argmax(1) == known_y

    dummy_logits = np.zeros((len(train_e), len(known_classes)), dtype=np.float32)
    conformal = MondrianMahalanobis(train_e, dummy_logits, train_y, val_e, val_y)
    val_p_loo = leave_one_out_validation_p(conformal, val_e, val_y)
    val_score = 1.0 - val_p_loo.max(1)
    known_score = 1.0 - conformal.p_values(known_e).max(1)

    # Unknown data is materialized only after training, model selection,
    # feature normalization, covariance fitting, and threshold references freeze.
    rogue_global = y[indices["rogue_test"]].astype(np.int64)
    rogue_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(x[indices["rogue_test"]])),
            torch.from_numpy(rogue_global),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    rogue_e, _, extracted_rogue = extract(backbone, rogue_loader, device)
    physics_rogue = standardize(extract_all(x[indices["rogue_test"]], 64))
    rogue_logits = predict(
        head, rogue_e.astype(np.float32), physics_rogue, args.batch_size, device
    )
    rogue_score = 1.0 - conformal.p_values(rogue_e).max(1)
    metrics = metric_row(
        "deep_mondrian_rejection_plus_learned_physics_identification",
        val_score,
        known_score,
        rogue_score,
        known_correct,
        args.known_reject_rate,
    )
    metrics["known_closed_acc"] = float(known_correct.mean())
    output = {
        "method": "frozen_deep_osr_plus_learned_physics_identification_head_v1",
        "split": int(model_args.split),
        "seed": int(model_args.seed),
        "known_classes": known_classes,
        "rogue_classes": rogue_classes,
        "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256,
        "selected_epoch": int(selected["epoch"]),
        "selected_val_loss": float(selected["val_loss"]),
        "selected_val_acc": float(selected["val_acc"]),
        "head_trainable_parameters": int(
            sum(parameter.numel() for parameter in head.parameters())
        ),
        "train_seconds": time.time() - start_time,
        "protocol": {
            "class_split_status": manifest["status"],
            "reserved_splits_accessed": manifest["status"] != "development",
            "frozen_method_manifest": args.frozen_manifest,
            "backbone": "frozen known-only checkpoint",
            "identification_head_training_and_selection": "known train/validation only",
            "rejection_score": "unchanged deep Mondrian nonconformity",
            "threshold": "known validation leave-one-out 90th percentile",
            "target_rogue_materialized": "after all training, selection, fitting, and calibration froze",
            "target_rogue_used_for_fit_selection_or_threshold": False,
        },
        "metrics": metrics,
        "history": history,
        "rogue_class_counts": {
            str(int(class_id)): int((extracted_rogue == class_id).sum())
            for class_id in np.unique(extracted_rogue)
        },
        "rogue_predictions_not_used_for_selection": np.bincount(
            rogue_logits.argmax(1), minlength=len(known_classes)
        ).astype(int).tolist(),
    }
    (output_dir / "result.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("FINAL", json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
