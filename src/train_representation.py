"""Known-only representation-learning experiments for open-set SEI.

This runner deliberately keeps the deployed encoder equal to the audited backbone
and varies only the representation objective.  It tests whether directly shaping
the normalized embedding geometry is more useful than adding a wide frequency/MoE
feature block.  Target rogue emitters never participate in training or selection.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import train_emitter_based as legacy
from model.arcface_loss import ArcFace

from train_causal_ablation import (
    backbone_fingerprint,
    build_model,
    evaluate_known,
    final_metrics,
    move_batch,
    seed_everything,
)


class PrototypeGeometryLoss(nn.Module):
    """CIDER-style hyperspherical compactness and dispersion regularizer."""

    def __init__(self, classes: int, dimension: int, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.empty(classes, dimension))
        nn.init.orthogonal_(self.prototypes)

    def forward(self, embedding, target):
        z = F.normalize(embedding, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        compactness = F.cross_entropy(z @ prototypes.T / self.temperature, target)

        similarities = prototypes @ prototypes.T / self.temperature
        diagonal = torch.eye(
            len(prototypes), device=similarities.device, dtype=torch.bool
        )
        similarities = similarities.masked_fill(diagonal, -torch.inf)
        dispersion = (
            torch.logsumexp(similarities, dim=1) - math.log(len(prototypes) - 1)
        ).mean()
        return compactness, dispersion


class ProxyAnchorLoss(nn.Module):
    """Proxy-Anchor metric loss with numerically stable log-sum-exp terms."""

    def __init__(
        self, classes: int, dimension: int, margin: float = 0.1, alpha: float = 32.0
    ) -> None:
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.proxies = nn.Parameter(torch.empty(classes, dimension))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    @staticmethod
    def log_one_plus_sum_exp(values: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(
            torch.cat([values, values.new_zeros(1)]), dim=0
        )

    def forward(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = F.normalize(embedding, dim=1) @ F.normalize(self.proxies, dim=1).T
        positive_terms = []
        negative_terms = []
        for class_id in range(self.proxies.shape[0]):
            is_positive = target == class_id
            if is_positive.any():
                positive_terms.append(
                    self.log_one_plus_sum_exp(
                        -self.alpha * (cosine[is_positive, class_id] - self.margin)
                    )
                )
            if (~is_positive).any():
                negative_terms.append(
                    self.log_one_plus_sum_exp(
                        self.alpha * (cosine[~is_positive, class_id] + self.margin)
                    )
                )
        positive = torch.stack(positive_terms).mean()
        negative = torch.stack(negative_terms).mean()
        return positive + negative


def supervised_contrastive_loss(
    embedding: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Supervised contrastive loss with safe handling of singleton-class anchors."""
    z = F.normalize(embedding, dim=1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    denominator = torch.exp(logits).masked_fill(self_mask, 0.0).sum(dim=1)
    log_probability = logits - torch.log(denominator.clamp_min(1e-12))[:, None]
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embedding.sum() * 0.0
    per_anchor = -(log_probability * positive).sum(dim=1) / positive_count.clamp_min(1)
    return per_anchor[valid].mean()


def dataset_fingerprint(loader) -> str:
    """Hash the exact materialized split so checkpoints prove split provenance."""
    digest = hashlib.sha256()
    tensors = getattr(loader.dataset, "tensors", None)
    if tensors is None:
        raise TypeError("expected the legacy loader to materialize a TensorDataset")
    for value in tensors:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def augment_iq(x: torch.Tensor, args) -> torch.Tensor:
    """RF-valid nuisance augmentation applied independently per sample."""
    batch, _, length = x.shape
    phase = torch.empty(batch, 1, 1, device=x.device).uniform_(
        -args.max_phase, args.max_phase
    )
    cosine, sine = phase.cos(), phase.sin()
    i, q = x[:, 0:1], x[:, 1:2]
    rotated = torch.cat([cosine * i - sine * q, sine * i + cosine * q], dim=1)

    gain = torch.empty(batch, 1, 1, device=x.device).uniform_(
        1.0 - args.gain_jitter, 1.0 + args.gain_jitter
    )
    augmented = gain * rotated

    if args.max_shift:
        shifts = torch.randint(
            -args.max_shift, args.max_shift + 1, (batch, 1), device=x.device
        )
        base = torch.arange(length, device=x.device)[None, :]
        indices = (base - shifts) % length
        augmented = augmented.gather(2, indices[:, None, :].expand(-1, 2, -1))

    if args.snr_min < args.snr_max:
        snr_db = torch.empty(batch, 1, 1, device=x.device).uniform_(
            args.snr_min, args.snr_max
        )
        signal_power = augmented.square().mean(dim=(1, 2), keepdim=True)
        noise_power = signal_power / torch.pow(10.0, snr_db / 10.0)
        augmented = augmented + torch.randn_like(augmented) * noise_power.sqrt()
    return augmented


class ModelEMA:
    """Parameter/buffer EMA without keeping a second network on the GPU."""

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.shadow = None
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if self.shadow is None:
            self.shadow = {name: value.detach().clone() for name, value in current.items()}
        else:
            for name, value in current.items():
                if value.is_floating_point():
                    self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[name].copy_(value)
        self.updates += 1

    @contextlib.contextmanager
    def applied(self, model: nn.Module):
        if self.shadow is None:
            yield model
            return
        backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield model
        finally:
            model.load_state_dict(backup)


def run_epoch(model, loader, optimizer, objective, arcface, geometry, ema, epoch, args, device):
    model.train()
    totals = {name: 0.0 for name in ("loss", "ce", "metric", "consistency")}
    total_correct = total_samples = 0
    use_views = objective in {"supcon_aug", "cider_consistency"}
    for data, target in loader:
        data, target = move_batch(data, target, device)
        optimizer.zero_grad(set_to_none=True)

        if use_views:
            model_input = torch.cat(
                [augment_iq(data, args), augment_iq(data, args)], dim=0
            )
            repeated_target = target.repeat(2)
        else:
            model_input, repeated_target = data, target

        embedding, logits = model(model_input)
        ce = F.cross_entropy(
            logits, repeated_target, label_smoothing=args.label_smoothing
        )
        metric = torch.zeros((), device=device)
        consistency = torch.zeros((), device=device)
        if objective == "arcface":
            metric = F.cross_entropy(arcface(embedding, repeated_target), repeated_target)
        elif objective in {"cider", "cider_consistency"}:
            compactness, dispersion = geometry(embedding, repeated_target)
            metric = compactness + args.dispersion_weight * dispersion
        elif objective in {"supcon", "supcon_aug"}:
            metric = supervised_contrastive_loss(
                embedding, repeated_target, args.metric_temperature
            )
        elif objective == "proxy_anchor":
            metric = geometry(embedding, repeated_target)

        if objective == "cider_consistency":
            first, second = embedding.chunk(2)
            consistency = (
                1.0
                - F.cosine_similarity(
                    F.normalize(first, dim=1), F.normalize(second, dim=1)
                )
            ).mean()

        loss = ce + args.lambda_metric * metric + args.lambda_consistency * consistency
        loss.backward()
        parameters = list(model.parameters())
        if arcface is not None:
            parameters += list(arcface.parameters())
        if geometry is not None:
            parameters += list(geometry.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        if ema is not None and epoch + 1 >= args.ema_start_epoch:
            ema.update(model)

        batch = repeated_target.numel()
        totals["loss"] += loss.item() * batch
        totals["ce"] += ce.item() * batch
        totals["metric"] += metric.item() * batch
        totals["consistency"] += consistency.item() * batch
        total_correct += (logits.argmax(1) == repeated_target).sum().item()
        total_samples += batch

    return {
        name: value / total_samples for name, value in totals.items()
    } | {"acc": total_correct / total_samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        choices=[
            "ce",
            "arcface",
            "cider",
            "supcon",
            "supcon_aug",
            "proxy_anchor",
            "cider_consistency",
        ],
        required=True,
    )
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="known train/validation/test split seed; defaults to --seed for legacy runs",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="checkpoint/evaluate an exponential moving average; zero disables EMA",
    )
    parser.add_argument("--ema-start-epoch", type=int, default=5)
    parser.add_argument("--lambda-metric", type=float, default=0.5)
    parser.add_argument("--lambda-consistency", type=float, default=0.25)
    parser.add_argument("--dispersion-weight", type=float, default=1.0)
    parser.add_argument("--metric-temperature", type=float, default=0.1)
    parser.add_argument("--proxy-margin", type=float, default=0.1)
    parser.add_argument("--proxy-alpha", type=float, default=32.0)
    parser.add_argument("--max-phase", type=float, default=3.141592653589793)
    parser.add_argument("--gain-jitter", type=float, default=0.1)
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--snr-min", type=float, default=20.0)
    parser.add_argument("--snr-max", type=float, default=40.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--input-normalization",
        choices=["none", "rms", "dc_rms"],
        default="none",
    )
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.data_seed is None:
        args.data_seed = args.seed
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label smoothing must lie in [0, 1)")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("EMA decay must lie in [0, 1)")

    # Compatibility fields used by build_model and the score benchmark.
    args.variant = "backbone"
    args.topk = 2
    args.fixed_rho = 0.25

    # The legacy loader ignores its ``seed`` argument and consumes the global
    # torch RNG.  Seed it explicitly for split construction, then independently
    # reseed model initialization/training below.
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
    initial_backbone = backbone_fingerprint(model)
    seed_everything(args.seed + 10_000)

    arcface = (
        ArcFace(64, 9, scale=30.0, margin=0.3).to(device)
        if args.objective == "arcface"
        else None
    )
    if args.objective in {"cider", "cider_consistency"}:
        geometry = PrototypeGeometryLoss(9, 64, args.metric_temperature).to(device)
    elif args.objective == "proxy_anchor":
        geometry = ProxyAnchorLoss(
            9, 64, margin=args.proxy_margin, alpha=args.proxy_alpha
        ).to(device)
    else:
        geometry = None
    # Objective-specific heads consume different random draws.  Pair the
    # DataLoader shuffle/dropout streams after constructing them so CE-vs-metric
    # comparisons differ only through the intended objective.
    seed_everything(args.seed + 20_000)
    parameters = list(model.parameters())
    if arcface is not None:
        parameters += list(arcface.parameters())
    if geometry is not None:
        parameters += list(geometry.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    ema = ModelEMA(args.ema_decay) if args.ema_decay else None
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
        train = run_epoch(
            model,
            loaders[0],
            optimizer,
            args.objective,
            arcface,
            geometry,
            ema,
            epoch,
            args,
            device,
        )
        if ema is None:
            val_loss, val_acc = evaluate_known(model, loaders[1], device)
        else:
            with ema.applied(model) as selection_model:
                val_loss, val_acc = evaluate_known(
                    selection_model, loaders[1], device
                )
        scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train": train, "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            stale = 0
            selected_state = (
                ema.shadow if ema is not None and ema.shadow is not None else model.state_dict()
            )
            torch.save(
                {
                    "model": selected_state,
                    "arcface": arcface.state_dict() if arcface is not None else None,
                    "geometry": geometry.state_dict() if geometry is not None else None,
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
    result = {
        "objective": args.objective,
        "variant": "backbone",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "R": args.R,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "best_val_acc": checkpoint["val_acc"],
        "train_seconds": time.time() - start,
        "deployed_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "initial_backbone_fingerprint": initial_backbone,
        "split_fingerprints": split_fingerprints,
        "runner_sha256": runner_sha256,
        "protocol": {
            "training_and_selection": "known only",
            "target_rogue_used_for_selection": False,
            "checkpoint_selection": "known validation loss",
            "initialization_seed": args.seed,
            "known_split_seed": args.data_seed,
            "label_smoothing": args.label_smoothing,
            "checkpoint_weights": (
                f"EMA decay={args.ema_decay} from epoch {args.ema_start_epoch}"
                if ema is not None
                else "instantaneous"
            ),
        },
        "metrics": final_metrics(model, loaders, args, device),
        "history": history,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
