"""Official GCPL/RPL/PROSER objectives on the frozen multiknown17 NetTime representation.

The losses and native scores follow the public author repositories:
  ARPL: https://github.com/iCGY96/ARPL
  PROSER: https://github.com/LAMDA-CL/CVPR21-Proser
Only the input adapter and evaluation protocol are dataset-specific.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate_openmax_multiknown17 import (
    NormalizedNetTime,
    extract,
    load_nettime,
    make_loader,
    metric_row,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Dist(nn.Module):
    """Verbatim geometry of ARPL loss/Dist.py (one center per class)."""

    def __init__(self, classes: int, dimension: int):
        super().__init__()
        self.classes = classes
        self.dimension = dimension
        self.centers = nn.Parameter(0.1 * torch.randn(classes, dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value_norm = value.square().sum(1, keepdim=True)
        center_norm = self.centers.square().sum(1, keepdim=True).T
        return (value_norm - 2 * value @ self.centers.T + center_norm) / float(self.dimension)


class PrototypeHead(nn.Module):
    """GCPLoss.py or RPLoss.py from the official ARPL repository."""

    def __init__(self, method: str, classes: int, dimension: int, weight_pl: float = 0.1):
        super().__init__()
        self.method = method
        self.weight_pl = weight_pl
        self.dist = Dist(classes, dimension)
        if method == "rpl":
            self.radius = nn.Parameter(torch.zeros(1))

    def logits(self, value: torch.Tensor) -> torch.Tensor:
        distance = self.dist(value)
        return -distance if self.method == "gcpl" else distance

    def forward(self, value: torch.Tensor, target: torch.Tensor):
        distance = self.dist(value)
        logits = -distance if self.method == "gcpl" else distance
        loss = F.cross_entropy(logits, target)
        center = self.dist.centers[target]
        if self.method == "gcpl":
            loss_r = F.mse_loss(value, center) / 2
        else:
            sample_distance = (value - center).square().mean(1)
            loss_r = F.mse_loss(sample_distance, self.radius)
        return logits, loss + self.weight_pl * loss_r


class ProserHead(nn.Module):
    """One official classifier placeholder, attached to frozen 64-D features."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dummy = nn.Linear(dimension, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dummy(value)


def load_data(args):
    root = Path(args.dataset_root)
    manifest = json.loads((root / f"split_{args.split}.json").read_text())
    known = list(map(int, manifest["known_classes"]))
    rogue = list(map(int, manifest["rogue_classes"]))
    mapping = {value: index for index, value in enumerate(known)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(root / "Y_global.npy")
    role = np.load(root / "sample_role.npy")
    known_mask = np.isin(y, known)
    indices = {
        "train": np.flatnonzero(known_mask & (role == 0)),
        "val": np.flatnonzero(known_mask & (role == 1)),
        "known_test": np.flatnonzero(known_mask & (role == 2)),
        "rogue_test": np.flatnonzero(np.isin(y, rogue) & (role == 2)),
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    NetTime = load_nettime(Path(args.model_source_root))
    base = NetTime(c_in=2, c_out=len(known), nf=16)
    base.fexp = nn.Identity()
    base.load_state_dict(checkpoint["model"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = NormalizedNetTime(base).to(device)
    train = extract(model, make_loader(x, y, indices["train"], mapping), device)
    val = extract(model, make_loader(x, y, indices["val"], mapping), device)
    known_test = extract(model, make_loader(x, y, indices["known_test"], mapping), device)
    rogue_test = extract(model, make_loader(x, y, indices["rogue_test"], {}), device)
    return known, rogue, train, val, known_test, rogue_test, device


def train_prototype(args, train, val, classes, device, output):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train[0]).float(), torch.from_numpy(train[2]).long()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    head = PrototypeHead(args.method, classes, train[0].shape[1]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best, stale, history = float("inf"), 0, []
    for epoch in range(args.epochs):
        head.train()
        total_loss = correct = count = 0
        for value, target in loader:
            value, target = value.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = head(value, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
            correct += (logits.argmax(1) == target).sum().item()
            count += len(target)
        head.eval()
        with torch.no_grad():
            value = torch.from_numpy(val[0]).float().to(device)
            target = torch.from_numpy(val[2]).long().to(device)
            logits = head.logits(value)
            val_loss = F.cross_entropy(logits, target).item()
            val_acc = (logits.argmax(1) == target).float().mean().item()
        scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train_loss": total_loss / count, "train_acc": correct / count,
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            torch.save({"head": head.state_dict(), "epoch": epoch + 1, "val_loss": val_loss}, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    saved = torch.load(output / "best.pt", map_location=device)
    head.load_state_dict(saved["head"])
    head.eval()

    def infer(values):
        with torch.no_grad():
            logits = head.logits(torch.from_numpy(values).float().to(device))
            probability = F.softmax(logits, dim=1)
        return logits.cpu().numpy(), probability.cpu().numpy()

    return infer, saved, history


def train_proser(args, train, val, classes, device, output):
    """Official 10-epoch placeholder fine-tuning, adapted at the frozen embedding layer."""
    dataset = TensorDataset(
        torch.from_numpy(train[0]).float(), torch.from_numpy(train[1]).float(), torch.from_numpy(train[2]).long()
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    head = ProserHead(train[0].shape[1]).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=args.proser_lr, momentum=0.9, weight_decay=5e-4)
    history = []
    for epoch in range(args.proser_epochs):
        head.train()
        total_loss = correct = count = 0
        for feature, known_logits, target in loader:
            feature, known_logits, target = feature.to(device), known_logits.to(device), target.to(device)
            half = len(target) // 2
            first_feature, first_target = feature[:half], target[:half]
            later_feature, later_logits, later_target = feature[half:], known_logits[half:], target[half:]
            beta = torch.distributions.beta.Beta(args.alpha, args.alpha).sample().to(device)
            permutation = torch.randperm(half, device=device)
            mixed_feature = beta * first_feature + (1 - beta) * first_feature[permutation]
            mixed_known_logits = beta * known_logits[:half] + (1 - beta) * known_logits[:half][permutation]
            mixed_total = torch.cat((mixed_known_logits, head(mixed_feature)), 1)
            later_dummy = head(later_feature)
            later_total = torch.cat((later_logits, later_dummy), 1)
            dummy_competition = torch.cat((later_logits.clone(), later_dummy), 1)
            dummy_competition[torch.arange(len(later_target), device=device), later_target] = -1e9
            dummy_target = torch.full_like(later_target, classes)
            loss1 = F.cross_entropy(mixed_total, torch.full_like(first_target, classes))
            loss2 = F.cross_entropy(later_total, later_target)
            loss3 = F.cross_entropy(dummy_competition, dummy_target)
            loss = 0.01 * loss1 + loss2 + loss3
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
            correct += (later_total.argmax(1) == later_target).sum().item()
            count += len(target)
        head.eval()
        with torch.no_grad():
            val_feature = torch.from_numpy(val[0]).float().to(device)
            val_known_logits = torch.from_numpy(val[1]).float().to(device)
            val_target = torch.from_numpy(val[2]).long().to(device)
            val_total = torch.cat((val_known_logits, head(val_feature)), 1)
            val_loss = F.cross_entropy(val_total, val_target).item()
            val_acc = (val_total[:, :classes].argmax(1) == val_target).float().mean().item()
        row = {"epoch": epoch + 1, "train_loss": total_loss / count, "train_acc_later_half": correct / (count / 2),
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
    saved = {"head": head.state_dict(), "epoch": args.proser_epochs, "val_loss": history[-1]["val_loss"]}
    torch.save(saved, output / "best.pt")

    def infer(values, known_logits):
        with torch.no_grad():
            value = torch.from_numpy(values).float().to(device)
            logits = torch.from_numpy(known_logits).float().to(device)
            total = torch.cat((logits, head(value)), 1)
            probability = F.softmax(total / args.proser_temperature, dim=1)
            score = probability[:, -1] - probability[:, :-1].max(1).values
        return total.cpu().numpy(), score.cpu().numpy()

    return infer, saved, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gcpl", "rpl", "proser"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-source-root",
        default=str(Path(__file__).resolve().parent / "model"),
    )
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--proser-lr", type=float, default=1e-3)
    parser.add_argument("--proser-epochs", type=int, default=10)
    parser.add_argument("--proser-temperature", type=float, default=1024.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    known, rogue, train, val, known_test, rogue_test, device = load_data(args)
    start = time.time()

    if args.method in ("gcpl", "rpl"):
        infer, saved, history = train_prototype(args, train, val, len(known), device, output)
        val_logits, val_probability = infer(val[0])
        known_logits, known_probability = infer(known_test[0])
        rogue_logits, rogue_probability = infer(rogue_test[0])
        val_score = -val_probability.max(1)
        known_score = -known_probability.max(1)
        rogue_score = -rogue_probability.max(1)
        known_prediction = known_logits.argmax(1)
        source = "https://github.com/iCGY96/ARPL"
        objective = f"official {args.method.upper()} distance and prototype regularizer"
        native_score = "negative maximum official-head probability"
    else:
        infer, saved, history = train_proser(args, train, val, len(known), device, output)
        _, val_score = infer(val[0], val[1])
        known_total, known_score = infer(known_test[0], known_test[1])
        _, rogue_score = infer(rogue_test[0], rogue_test[1])
        known_prediction = known_total[:, :len(known)].argmax(1)
        source = "https://github.com/LAMDA-CL/CVPR21-Proser"
        objective = "official PROSER placeholder losses and manifold mixup"
        native_score = "dummy probability minus maximum known probability at T=1024"

    metrics = metric_row(val_score, known_score, rogue_score, known_prediction == known_test[2])
    np.savez_compressed(
        output / "per_sample_scores.npz",
        validation_score=val_score,
        known_score=known_score,
        known_label=known_test[2],
        known_prediction=known_prediction,
        rogue_score=rogue_score,
        rogue_label=rogue_test[2],
    )
    result = {
        "method": args.method.upper(),
        "implementation": f"official_{args.method}_objective_on_frozen_nettime_v1",
        "official_source": source,
        "split": args.split,
        "seed": args.seed,
        "known_classes": known,
        "rogue_classes": rogue,
        "best_epoch": saved["epoch"],
        "train_seconds": time.time() - start,
        "metrics": metrics,
        "history": history,
        "protocol": {
            "backbone": "exact frozen NetTime checkpoint used by Ours",
            "trainable_scope": "official method-specific head only",
            "objective": objective,
            "score": native_score,
            "checkpoint_selection": "known-validation CE" if args.method != "proser" else "official fixed 10-epoch fine-tuning",
            "operating_threshold": "known-validation Q90",
            "target_rogue_used_for_fit_selection_or_threshold": False,
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
