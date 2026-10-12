"""Known-only ARPL baseline for open-set specific-emitter identification.

The reciprocal-point head and loss follow the authors' public ARPL reference
implementation.  Only known training/validation samples are used for fitting,
checkpoint selection, and the rejection threshold.  Target rogue emitters are
loaded once, after training, for final reporting only.
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

import train_emitter_based as legacy
from posthoc_ablation import auc_rank, eer, oscr, roc_points
from score_benchmark import full_rogue_loader
from train_causal_ablation import build_model, move_batch, seed_everything
from train_representation import dataset_fingerprint


class ARPLHead(nn.Module):
    """Exact reciprocal-point scoring/loss used by the official ARPL code."""

    def __init__(
        self,
        classes: int = 9,
        dimension: int = 64,
        temperature: float = 1.0,
        weight_pl: float = 0.1,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.temperature = temperature
        self.weight_pl = weight_pl
        self.points = nn.Parameter(0.1 * torch.randn(classes, dimension))
        self.radius = nn.Parameter(torch.zeros(1))

    def logits(self, embedding: torch.Tensor) -> torch.Tensor:
        feature_square = embedding.square().sum(dim=1, keepdim=True)
        point_square = self.points.square().sum(dim=1)[None, :]
        distance_l2 = (
            feature_square
            - 2.0 * embedding @ self.points.T
            + point_square
        ) / float(self.dimension)
        distance_dot = embedding @ self.points.T
        return distance_l2 - distance_dot

    def forward(
        self, embedding: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(embedding)
        if target is None:
            return logits, logits.new_zeros(())
        classification = F.cross_entropy(logits / self.temperature, target)
        selected_points = self.points[target]
        distance_known = (embedding - selected_points).square().mean(dim=1)
        ranking_target = torch.ones_like(distance_known)
        radius_loss = F.margin_ranking_loss(
            self.radius.expand_as(distance_known),
            distance_known,
            ranking_target,
            margin=1.0,
        )
        return logits, classification + self.weight_pl * radius_loss


def evaluate_known(model, head, loader, device, temperature):
    model.eval()
    head.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = move_batch(data, target, device)
            embedding, _ = model(data)
            logits = head.logits(embedding)
            batch = target.numel()
            total_loss += F.cross_entropy(logits / temperature, target).item() * batch
            total_correct += (logits.argmax(1) == target).sum().item()
            total_samples += batch
    return total_loss / total_samples, total_correct / total_samples


def extract(model, head, loader, device):
    model.eval()
    head.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data, _ = move_batch(data, target, device)
            embedding, _ = model(data)
            all_logits.append(head.logits(embedding).cpu().numpy())
            all_labels.append(target.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def metric_row(known_logits, known_labels, rogue_logits, threshold):
    # The official ARPL evaluation uses maximum reciprocal-point logit as its
    # confidence. Negation makes larger values mean "more unknown" here.
    known_score = -known_logits.max(axis=1)
    rogue_score = -rogue_logits.max(axis=1)
    known_correct = known_logits.argmax(axis=1) == known_labels
    known_reject = known_score > threshold
    rogue_reject = rogue_score > threshold
    tp = int(rogue_reject.sum())
    fp = int(known_reject.sum())
    fn = int((~rogue_reject).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr, tpr, _ = roc_points(known_score, rogue_score)
    reaches = np.flatnonzero(tpr >= 0.95)
    labels = np.r_[
        np.zeros(len(known_score), dtype=int),
        np.ones(len(rogue_score), dtype=int),
    ]
    scores = np.r_[known_score, rogue_score]
    return {
        "score": "official_arpl_negative_max_logit",
        "auc": auc_rank(labels, scores),
        "eer": eer(known_score, rogue_score),
        "fpr95": float(np.min(fpr[reaches])) if len(reaches) else 1.0,
        "known_closed_acc": float(known_correct.mean()),
        "known_accept_rate": float((~known_reject).mean()),
        "unknown_recall": float(recall),
        "unknown_f1": float(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
        "oscr": oscr(known_score, rogue_score, known_correct),
        "tau_known_only": float(threshold),
        "n_known_test": int(len(known_score)),
        "n_rogue_test": int(len(rogue_score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--eval-R", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-2,
        help="learning rate for the newly initialized reciprocal-point head",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--weight-pl", type=float, default=0.1)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--input-normalization", choices=["none", "rms", "dc_rms"], default="none")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.variant = "backbone"
    args.topk = 2
    args.fixed_rho = 0.25
    if not 0.0 < args.known_reject_rate < 1.0:
        raise ValueError("known-reject-rate must lie strictly between zero and one")

    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    seed_everything(args.data_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = legacy.EmitterBasedDataset(R=args.R)
    loaders = legacy.create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        rogue_val_split=0.5,
        seed=args.data_seed,
    )
    split_fingerprints = {
        name: dataset_fingerprint(loader)
        for name, loader in zip(
            ("known_train", "known_validation", "known_test"), loaders[:3]
        )
    }

    seed_everything(args.seed)
    model = build_model(args, device)
    head = ARPLHead(
        classes=9,
        dimension=64,
        temperature=args.temperature,
        weight_pl=args.weight_pl,
    ).to(device)
    seed_everything(args.seed + 20_000)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": head.parameters(),
                "lr": args.head_lr,
                "weight_decay": 0.0,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    best_loss = float("inf")
    stale = 0
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        head.train()
        loss_total = correct = samples = 0
        for data, target in loaders[0]:
            data, target = move_batch(data, target, device)
            optimizer.zero_grad(set_to_none=True)
            embedding, _ = model(data)
            logits, loss = head(embedding, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), args.grad_clip
            )
            optimizer.step()
            batch = target.numel()
            loss_total += loss.item() * batch
            correct += (logits.argmax(1) == target).sum().item()
            samples += batch
        val_loss, val_acc = evaluate_known(
            model, head, loaders[1], device, args.temperature
        )
        scheduler.step(val_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_total / samples,
            "train_acc": correct / samples,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "radius": float(head.radius.detach().cpu()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arpl_head": head.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "split_fingerprints": split_fingerprints,
                    "runner_sha256": runner_sha256,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    head.load_state_dict(checkpoint["arpl_head"])
    validation_logits, _ = extract(model, head, loaders[1], device)
    known_logits, known_labels = extract(model, head, loaders[2], device)
    rogue_logits, rogue_labels = extract(
        model,
        head,
        full_rogue_loader(args.eval_R, args.eval_batch_size),
        device,
    )
    validation_score = -validation_logits.max(axis=1)
    threshold = float(
        np.quantile(validation_score, 1.0 - args.known_reject_rate)
    )
    metrics = metric_row(known_logits, known_labels, rogue_logits, threshold)
    per_rogue_class = {
        str(int(class_id)): metric_row(
            known_logits,
            known_labels,
            rogue_logits[rogue_labels == class_id],
            threshold,
        )
        for class_id in sorted(np.unique(rogue_labels))
    }
    result = {
        "method": "ARPL",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "R": args.R,
        "evaluation_R": args.eval_R,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "best_val_acc": checkpoint["val_acc"],
        "train_seconds": time.time() - start,
        "deployed_trainable_parameters": sum(
            parameter.numel()
            for parameter in list(model.parameters()) + list(head.parameters())
            if parameter.requires_grad
        ),
        "split_fingerprints": split_fingerprints,
        "runner_sha256": runner_sha256,
        "protocol": {
            "implementation": "official ARPL reciprocal-point equations",
            "hyperparameters": "public defaults temperature=1.0, weight_pl=0.1",
            "optimization": (
                f"AdamW encoder_lr={args.lr}, reciprocal_head_lr={args.head_lr}; "
                "head learning rate validated using known validation only"
            ),
            "training": "known training only",
            "checkpoint_selection": "known validation classification loss",
            "threshold": "known validation 90th percentile of negative max logit",
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "target_rogue_role": "final evaluation only",
        },
        "metrics": metrics,
        "per_rogue_class": per_rogue_class,
        "history": history,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
