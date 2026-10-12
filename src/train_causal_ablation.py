"""Clean retraining runner for causal FRAME component ablations.

Only known-class data participate in training, checkpoint selection, temperature
selection (fixed), and threshold calibration. Rogue validation samples returned by
the legacy dataset loader are deliberately ignored.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model.MRAN import NetTime

from correctness_modules import CorrectFRAE, NetFRAEAdapter
from posthoc_ablation import auc_rank, eer, oscr, centers, unknownness
from spectral_fusion_model import SpectralAuxNet, SpectralFusionNet, SpectralOnlyNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(data, target, device):
    if data.ndim == 3 and data.shape[-1] == 2:
        data = data.permute(0, 2, 1)
    return data.to(device, dtype=torch.float32), target.to(device)


def build_model(args, device):
    classes = int(getattr(args, "num_classes", 9))
    if args.variant == "spectral_only":
        model = SpectralOnlyNet(classes=classes)
    elif args.variant == "spectral_fusion":
        model = SpectralFusionNet(classes=classes)
    elif args.variant == "spectral_aux":
        model = SpectralAuxNet(classes=classes)
    else:
        model = NetTime(c_in=2, c_out=classes, nf=16)
    if args.variant in {"backbone", "spectral_only", "spectral_fusion", "spectral_aux"}:
        if hasattr(model, "fexp"):
            model.fexp = nn.Identity()
    else:
        mode = {
            "mee_only": "mee_only",
            "fd_only": "fd_only",
            "full": "full",
            "fixed_split": "full",
            "no_fusion": "no_fusion",
        }[args.variant]
        fixed_rho = args.fixed_rho if args.variant == "fixed_split" else None
        model.fexp = NetFRAEAdapter(
            CorrectFRAE(
                channels=512,
                mode=mode,
                num_experts=4,
                topk=args.topk,
                fixed_rho=fixed_rho,
            )
        )
    input_normalization = getattr(args, "input_normalization", "none")
    if input_normalization not in {"none", "rms", "dc_rms"}:
        raise ValueError(f"unknown input normalization {input_normalization!r}")
    if input_normalization != "none":
        original_forward = model.forward

        def normalized_forward(self, x):
            if input_normalization == "dc_rms":
                x = x - x.mean(dim=-1, keepdim=True)
            # One complex-signal scale per sample: preserve relative I/Q imbalance,
            # which may carry emitter information, while removing receiver gain.
            rms = x.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
            return original_forward(x / rms)

        model.forward = types.MethodType(normalized_forward, model)
    return model.to(device)


def backbone_fingerprint(model):
    values = [
        parameter.detach().double().reshape(-1)
        for name, parameter in model.named_parameters()
        if not name.startswith("fexp.")
    ]
    return {
        "sum": float(sum(value.sum().item() for value in values)),
        "sum_squares": float(sum(value.square().sum().item() for value in values)),
        "count": int(sum(value.numel() for value in values)),
    }


def parameter_groups(model, base_lr, router_lr):
    router, regular = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (router if ".router." in name else regular).append(parameter)
    groups = [{"params": regular, "lr": base_lr}]
    if router:
        groups.append({"params": router, "lr": router_lr})
    return groups


def router_aux(model, device):
    if isinstance(model.fexp, NetFRAEAdapter):
        diagnostics = model.fexp.last_diagnostics
        if diagnostics is not None and diagnostics.aux_loss is not None:
            return diagnostics.aux_loss
    return torch.zeros((), device=device)


def run_epoch(model, loader, optimizer, arcface, args, epoch, device):
    model.train()
    total_loss = total_correct = total_samples = 0
    aux_weight = args.lambda_aux * min(1.0, (epoch + 1) / max(args.aux_warmup, 1))
    completed = epoch + 1
    arc_progress = max(0.0, (completed - args.arc_warmup) / max(args.arc_ramp, 1))
    arc_weight = args.lambda_arc * min(1.0, arc_progress)
    for data, target in loader:
        data, target = move_batch(data, target, device)
        optimizer.zero_grad(set_to_none=True)
        embedding, logits = model(data)
        ce = F.cross_entropy(logits, target)
        angular = (
            F.cross_entropy(arcface(embedding, target), target)
            if arcface is not None
            else torch.zeros((), device=device)
        )
        auxiliary = router_aux(model, device)
        loss = ce + arc_weight * angular + aux_weight * auxiliary
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if arcface is not None:
            torch.nn.utils.clip_grad_norm_(arcface.parameters(), args.grad_clip)
        optimizer.step()
        batch = target.numel()
        total_loss += loss.item() * batch
        total_correct += (logits.argmax(1) == target).sum().item()
        total_samples += batch
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_known(model, loader, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    for data, target in loader:
        data, target = move_batch(data, target, device)
        _, logits = model(data)
        batch = target.numel()
        total_loss += F.cross_entropy(logits, target).item() * batch
        total_correct += (logits.argmax(1) == target).sum().item()
        total_samples += batch
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    embeddings, logits, labels = [], [], []
    for data, target in loader:
        data, target_device = move_batch(data, target, device)
        embedding, output = model(data)
        embeddings.append(embedding.cpu().numpy())
        logits.append(output.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(embeddings), np.concatenate(logits), np.concatenate(labels)


def final_metrics(model, loaders, args, device):
    train_loader, val_loader, known_loader, rogue_val_loader, rogue_test_loader = loaders
    train_embedding, _, train_labels = extract(model, train_loader, device)
    val_embedding, _, _ = extract(model, val_loader, device)
    known_embedding, known_logits, known_labels = extract(model, known_loader, device)
    # No target-rogue sample participates in training, model selection, score
    # selection, or threshold calibration.  Therefore the legacy rogue-val/test
    # partition has no statistical role here; merge both halves for one final
    # evaluation rather than silently discarding half of the unknown samples.
    rogue_val_embedding, _, _ = extract(model, rogue_val_loader, device)
    rogue_test_embedding, _, _ = extract(model, rogue_test_loader, device)
    rogue_embedding = np.concatenate([rogue_val_embedding, rogue_test_embedding])
    prototypes = centers(train_embedding, train_labels)
    val_scores = unknownness(val_embedding, prototypes, args.temperature)
    known_scores = unknownness(known_embedding, prototypes, args.temperature)
    rogue_scores = unknownness(rogue_embedding, prototypes, args.temperature)
    tau = float(np.quantile(val_scores, 1.0 - args.known_reject_rate))
    known_predictions = known_logits.argmax(1)
    known_correct = known_predictions == known_labels
    known_reject = known_scores > tau
    rogue_reject = rogue_scores > tau
    tp = int(rogue_reject.sum())
    fp = int(known_reject.sum())
    fn = len(rogue_reject) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    labels = np.r_[np.zeros(len(known_scores), dtype=int), np.ones(len(rogue_scores), dtype=int)]
    scores = np.r_[known_scores, rogue_scores]
    return {
        "auc": auc_rank(labels, scores),
        "eer": eer(known_scores, rogue_scores),
        "known_closed_acc": float(known_correct.mean()),
        "known_accept_rate": float((~known_reject).mean()),
        "unknown_recall": float(recall),
        "unknown_f1": float(f1),
        "oscr": oscr(known_scores, rogue_scores, known_correct),
        "tau_known_only": tau,
        "n_known_test": len(known_scores),
        "n_rogue_test": len(rogue_scores),
    }


def main():
    # Only the historical single-split runner needs these legacy modules.
    # The public multi-known pipeline can import the shared helpers without them.
    import train_emitter_based as legacy
    from model.arcface_loss import ArcFace
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["backbone", "mee_only", "fd_only", "full", "fixed_split", "no_fusion", "spectral_only", "spectral_fusion", "spectral_aux"], required=True)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-arc", type=float, default=1.0)
    parser.add_argument("--arc-warmup", type=int, default=5)
    parser.add_argument("--arc-ramp", type=int, default=20)
    parser.add_argument("--lambda-aux", type=float, default=0.01)
    parser.add_argument("--aux-warmup", type=int, default=10)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--fixed-rho", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = legacy.EmitterBasedDataset(R=args.R)
    loaders = legacy.create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        rogue_val_split=0.5,
        seed=args.seed,
    )
    model = build_model(args, device)
    initial_backbone = backbone_fingerprint(model)
    # Variant-specific module construction consumes a different number of random
    # draws. Reset here so ArcFace initialization, DataLoader shuffling, and
    # dropout streams are paired across all variants.
    seed_everything(args.seed + 10_000)
    arcface = ArcFace(64, 9, scale=30.0, margin=0.3).to(device) if args.lambda_arc else None
    groups = parameter_groups(model, args.lr, args.router_lr)
    if arcface is not None:
        groups.append({"params": arcface.parameters(), "lr": args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history = []
    best_loss = float("inf")
    stale = 0
    start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model, loaders[0], optimizer, arcface, args, epoch, device
        )
        val_loss, val_acc = evaluate_known(model, loaders[1], device)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
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
                    "model": model.state_dict(),
                    "arcface": arcface.state_dict() if arcface is not None else None,
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics = final_metrics(model, loaders, args, device)
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "R": args.R,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "best_val_acc": checkpoint["val_acc"],
        "train_seconds": time.time() - start,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_backbone_fingerprint": initial_backbone,
        "protocol": {
            "model_selection": "known validation loss only",
            "threshold": f"known validation {1.0-args.known_reject_rate:.3f} quantile",
            "softmin_temperature": args.temperature,
            "rogue_validation_used": False,
            "target_rogue_final_evaluation": "legacy rogue-val and rogue-test merged",
        },
        "metrics": metrics,
        "history": history,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
