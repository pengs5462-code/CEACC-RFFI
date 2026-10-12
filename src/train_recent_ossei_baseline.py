"""Protocol-adapted recent OS-SEI baselines on multiknown17_v1.

This preserves the audited IQ input, NetTime encoder, data roles, optimizer
budget, known-validation checkpoint rule and final metrics while changing only
each paper's defining head and auxiliary objectives. Target rogue samples are
materialized only after training, model selection and score calibration.
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

from posthoc_ablation import auc_rank, eer, oscr, roc_points
from train_arpl import ARPLHead
from train_causal_ablation import build_model, move_batch, seed_everything
from train_multiknown17_baseline import make_loader
from train_representation import dataset_fingerprint


METHODS = (
    "psgrnn", "medae", "rpdrl", "dacc", "tifs_vae_supcl", "dual_level",
    "arpl_repaired",
)
EPS = 1e-8


def pooled_target(data, bins=96):
    return F.adaptive_avg_pool1d(data, bins).flatten(1)


def add_awgn(data, low_db=8.0, high_db=25.0):
    snr = torch.empty(len(data), 1, 1, device=data.device).uniform_(low_db, high_db)
    power = data.square().mean((1, 2), keepdim=True).clamp_min(EPS)
    return data + torch.randn_like(data) * (power / torch.pow(10.0, snr / 10.0)).sqrt()


def mask_temporal(data, fraction=0.2):
    result = data.clone()
    width = max(1, int(data.shape[-1] * fraction))
    starts = torch.randint(0, data.shape[-1] - width + 1, (len(data),), device=data.device)
    for row, start in enumerate(starts.tolist()):
        result[row, :, start : start + width] = 0.0
    return result


def center_penalty(embedding, target, centers):
    return (embedding - centers[target]).square().mean()


def separation_penalty(centers, margin=1.0):
    distance = torch.cdist(centers, centers)
    mask = ~torch.eye(len(centers), dtype=torch.bool, device=centers.device)
    return F.relu(margin - distance[mask]).mean()


def supervised_contrastive(embedding, target, temperature=0.1):
    z = F.normalize(embedding, dim=1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(1, keepdim=True).values.detach()
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = target[:, None].eq(target[None, :]) & ~self_mask
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(EPS))
    count = positive.sum(1)
    valid = count > 0
    if not valid.any():
        return logits.new_zeros(())
    return -(log_prob * positive).sum(1)[valid].div(count[valid]).mean()


class LinearCenterHead(nn.Module):
    def __init__(self, classes, dimension=64):
        super().__init__()
        self.classifier = nn.Linear(dimension, classes)
        self.centers = nn.Parameter(0.1 * torch.randn(classes, dimension))

    def logits(self, embedding):
        return self.classifier(embedding)


class Decoder(nn.Module):
    def __init__(self, dimension=64, output=192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, output))

    def forward(self, embedding):
        return self.net(embedding)


class DACCHead(nn.Module):
    def __init__(self, classes, anchors=3, dimension=64):
        super().__init__()
        self.anchors = nn.Parameter(0.1 * torch.randn(classes, anchors, dimension))
        self.log_scale = nn.Parameter(torch.tensor(2.0))

    def distances(self, embedding):
        squared = (embedding[:, None, None, :] - self.anchors[None]).square().mean(3)
        return squared.min(2).values

    def logits(self, embedding):
        return -self.distances(embedding) * self.log_scale.exp().clamp(max=50.0)

    def clustering_loss(self, embedding, target):
        own = self.anchors[target]
        compact = (embedding[:, None] - own).square().mean(2).min(1).values.mean()
        coverage = embedding.new_zeros(())
        used = 0
        for class_id in target.unique():
            values = embedding[target == class_id]
            per_anchor = (values[:, None] - self.anchors[class_id][None]).square().mean(2)
            coverage = coverage + per_anchor.min(0).values.mean()
            used += 1
        return compact + coverage / max(used, 1) + 0.05 * separation_penalty(self.anchors.mean(1))


class VAEHead(nn.Module):
    def __init__(self, classes, dimension=64):
        super().__init__()
        self.mu = nn.Linear(dimension, dimension)
        self.logvar = nn.Linear(dimension, dimension)
        self.classifier = nn.Linear(dimension, classes)
        self.centers = nn.Parameter(0.1 * torch.randn(classes, dimension))

    def latent(self, embedding, sample):
        mu = self.mu(embedding)
        logvar = self.logvar(embedding).clamp(-8.0, 8.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return z, mu, logvar

    def logits(self, embedding):
        return self.classifier(self.mu(embedding))


class MethodModules(nn.Module):
    def __init__(self, method, classes):
        super().__init__()
        self.method = method
        if method == "psgrnn":
            self.head = LinearCenterHead(classes + 1)
        elif method in {"rpdrl", "arpl_repaired"}:
            self.head = ARPLHead(classes=classes, dimension=64, temperature=1.0, weight_pl=0.1)
        elif method == "dacc":
            self.head = DACCHead(classes)
        elif method == "tifs_vae_supcl":
            self.head = VAEHead(classes)
        else:
            self.head = LinearCenterHead(classes)
        self.decoder = Decoder() if method in {"medae", "rpdrl", "tifs_vae_supcl", "dual_level"} else None


def train_loss(method, encoder, modules, data, target, classes):
    clean_target = pooled_target(data)
    if method == "psgrnn":
        embedding, _ = encoder(data)
        logits = modules.head.logits(embedding)
        base_order = torch.arange(len(data), device=data.device)
        order = torch.roll(base_order, 1)
        for shift in range(2, len(data) + 1):
            same = target[order].eq(target)
            if not same.any():
                break
            candidate = torch.roll(base_order, shift)
            order = torch.where(same & target[candidate].ne(target), candidate, order)
        pseudo_embedding, _ = encoder(0.5 * data + 0.5 * data[order])
        pseudo_logits = modules.head.logits(pseudo_embedding)
        pseudo_target = torch.full_like(target, classes)
        all_embedding = torch.cat([embedding, pseudo_embedding])
        all_target = torch.cat([target, pseudo_target])
        loss = (
            F.cross_entropy(logits, target)
            + 0.5 * F.cross_entropy(pseudo_logits, pseudo_target)
            + 0.05 * center_penalty(all_embedding, all_target, modules.head.centers)
            + 0.01 * separation_penalty(modules.head.centers)
        )
        return loss, logits[:, :classes]

    encoder_input = add_awgn(data) if method == "medae" else data
    if method == "dual_level":
        encoder_input = mask_temporal(data)
    embedding, _ = encoder(encoder_input)
    if method in {"rpdrl", "arpl_repaired"}:
        logits, reciprocal = modules.head(embedding, target)
        reconstruction = (
            0.5 * F.mse_loss(modules.decoder(embedding), clean_target)
            if modules.decoder is not None else embedding.new_zeros(())
        )
        return reciprocal + reconstruction, logits
    if method == "dacc":
        logits = modules.head.logits(embedding)
        return F.cross_entropy(logits, target) + 0.1 * modules.head.clustering_loss(embedding, target), logits
    if method == "tifs_vae_supcl":
        z, mu, logvar = modules.head.latent(embedding, sample=True)
        logits = modules.head.classifier(z)
        reconstruction = F.mse_loss(modules.decoder(z), clean_target)
        kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean()
        loss = (
            F.cross_entropy(logits, target) + 0.5 * reconstruction + 1e-3 * kl
            + 0.05 * center_penalty(mu, target, modules.head.centers)
            + 0.05 * supervised_contrastive(mu, target)
        )
        return loss, logits

    logits = modules.head.logits(embedding)
    reconstruction_each = (modules.decoder(embedding) - clean_target).square().mean(1)
    metric = center_penalty(embedding, target, modules.head.centers)
    if method == "medae":
        loss = F.cross_entropy(logits, target) + reconstruction_each.mean() + 0.05 * metric
    else:
        uncertainty = (reconstruction_each.detach() / reconstruction_each.detach().mean().clamp_min(EPS)).clamp(0.5, 2.0)
        classification = F.cross_entropy(logits, target, reduction="none")
        proxy_logits = 16.0 * F.normalize(embedding, dim=1) @ F.normalize(modules.head.centers, dim=1).T
        loss = (
            (classification * uncertainty).mean() + 0.5 * reconstruction_each.mean()
            + 0.05 * metric + 0.1 * F.cross_entropy(proxy_logits, target)
        )
    return loss, logits


@torch.no_grad()
def infer(method, encoder, modules, loader, device):
    encoder.eval(); modules.eval()
    output = {"embedding": [], "logits": [], "labels": [], "reconstruction": []}
    for data, target in loader:
        data, _ = move_batch(data, target, device)
        embedding, _ = encoder(data)
        if method == "tifs_vae_supcl":
            representation = modules.head.mu(embedding)
            logits = modules.head.classifier(representation)
        else:
            representation = embedding
            logits = modules.head.logits(embedding)
        output["embedding"].append(representation.cpu().numpy())
        output["logits"].append(logits.cpu().numpy())
        output["labels"].append(target.numpy())
        if modules.decoder is not None:
            reconstruction = (modules.decoder(representation) - pooled_target(data)).square().mean(1)
            output["reconstruction"].append(reconstruction.cpu().numpy())
    return {key: np.concatenate(value) if value else None for key, value in output.items()}


def empirical_cdf(reference, values):
    return np.searchsorted(np.sort(reference), values, side="right") / (len(reference) + 1.0)


def scores_for(method, train, validation, value):
    logits = value["logits"]
    shifted = logits - logits.max(1, keepdims=True)
    probability = np.exp(shifted); probability /= probability.sum(1, keepdims=True)
    if method == "psgrnn":
        return probability[:, -1]
    if method == "dacc":
        return -logits.max(1)
    primary_reference = -validation["logits"].max(1)
    primary = empirical_cdf(primary_reference, -logits.max(1))
    if value["reconstruction"] is None:
        return primary
    reconstruct = empirical_cdf(validation["reconstruction"], value["reconstruction"])
    if method in {"medae", "tifs_vae_supcl"}:
        classes = np.unique(train["labels"])
        centers = np.stack([train["embedding"][train["labels"] == c].mean(0) for c in classes])
        reference_distance = ((validation["embedding"][:, None] - centers[None]) ** 2).mean(2).min(1)
        distance = ((value["embedding"][:, None] - centers[None]) ** 2).mean(2).min(1)
        return (primary + reconstruct + empirical_cdf(reference_distance, distance)) / 3.0
    return 0.5 * (primary + reconstruct)


def metric_row(name, validation_score, known_score, rogue_score, known_correct, reject_rate):
    threshold = float(np.quantile(validation_score, 1.0 - reject_rate))
    known_reject = known_score > threshold; rogue_reject = rogue_score > threshold
    tp = int(rogue_reject.sum()); fp = int(known_reject.sum()); fn = int((~rogue_reject).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    fpr, tpr, _ = roc_points(known_score, rogue_score); reaches = np.flatnonzero(tpr >= 0.95)
    labels = np.r_[np.zeros(len(known_score), int), np.ones(len(rogue_score), int)]
    scores = np.r_[known_score, rogue_score]
    return {
        "score": name, "auc": auc_rank(labels, scores), "eer": eer(known_score, rogue_score),
        "fpr95": float(np.min(fpr[reaches])) if len(reaches) else 1.0,
        "known_closed_acc": float(known_correct.mean()), "known_accept_rate": float((~known_reject).mean()),
        "unknown_recall": float(recall), "unknown_f1": float(2 * precision * recall / max(precision + recall, EPS)),
        "oscr": oscr(known_score, rogue_score, known_correct), "tau_known_only": threshold,
        "n_known_test": int(len(known_score)), "n_rogue_test": int(len(rogue_score)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-checkpoint")
    args = parser.parse_args()
    args.variant = "backbone"; args.input_normalization = "dc_rms"; args.topk = 2; args.fixed_rho = 0.25; args.R = 8
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    root = Path(args.dataset_root); manifest = json.loads((root / f"split_{args.split}.json").read_text())
    if manifest["status"] != "development": raise ValueError("only development splits 1-3 are permitted")
    known_classes = [int(v) for v in manifest["known_classes"]]
    mapping = {value: index for index, value in enumerate(known_classes)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r"); y = np.load(root / "Y_global.npy"); role = np.load(root / "sample_role.npy")
    is_known = np.isin(y, known_classes)
    train_index = np.flatnonzero(is_known & (role == 0)); val_index = np.flatnonzero(is_known & (role == 1)); known_index = np.flatnonzero(is_known & (role == 2))
    expected = (manifest["known_train_samples"], manifest["known_validation_samples"], manifest["known_test_samples"])
    if tuple(map(len, (train_index, val_index, known_index))) != expected: raise ValueError("known role counts do not match manifest")
    seed_everything(args.seed + 20_000)
    train_loader = make_loader(x, y, train_index, mapping, args.batch_size, True)
    val_loader = make_loader(x, y, val_index, mapping, args.eval_batch_size, False)
    known_loader = make_loader(x, y, known_index, mapping, args.eval_batch_size, False)
    fingerprints = {name: dataset_fingerprint(loader) for name, loader in zip(("known_train", "known_validation", "known_test"), (train_loader, val_loader, known_loader))}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed); encoder = build_model(args, device)
    initialization = None
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location=device)
        initial_manifest = initial.get("split_manifest", {})
        if list(map(int, initial_manifest.get("known_classes", []))) != known_classes:
            raise ValueError("initial checkpoint known classes do not match requested split")
        encoder.load_state_dict(initial["model"])
        initialization = {
            "path": str(Path(args.initial_checkpoint).resolve()),
            "epoch": int(initial.get("epoch", -1)),
            "runner_sha256": initial.get("runner_sha256"),
        }
    modules = MethodModules(args.method, len(known_classes)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": modules.parameters(), "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True); checkpoint_path = output_dir / "best.pt"
    best_loss = float("inf"); stale = 0; history = []; start = time.time()
    for epoch in range(args.epochs):
        encoder.train(); modules.train(); total_loss = total_correct = total = 0
        for data, target in train_loader:
            data, target = move_batch(data, target, device); optimizer.zero_grad(set_to_none=True)
            loss, logits = train_loss(args.method, encoder, modules, data, target, len(known_classes))
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(modules.parameters()), 5.0); optimizer.step()
            batch = len(target); total_loss += loss.item() * batch; total_correct += (logits.argmax(1) == target).sum().item(); total += batch
        encoder.eval(); modules.eval(); val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = move_batch(data, target, device); embedding, _ = encoder(data)
                logits = modules.head.logits(embedding)
                batch = len(target); val_loss += F.cross_entropy(logits, target).item() * batch; val_correct += (logits.argmax(1) == target).sum().item(); val_total += batch
        val_loss /= val_total; scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train_loss": total_loss / total, "train_acc": total_correct / total, "val_loss": val_loss, "val_acc": val_correct / val_total}
        history.append(row); print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss; stale = 0
            torch.save({"encoder": encoder.state_dict(), "modules": modules.state_dict(), "args": vars(args), "epoch": epoch + 1, "val_loss": val_loss, "split_manifest": manifest, "split_fingerprints": fingerprints, "runner_sha256": runner_sha256}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience: break

    checkpoint = torch.load(checkpoint_path, map_location=device); encoder.load_state_dict(checkpoint["encoder"]); modules.load_state_dict(checkpoint["modules"])
    train_eval = DataLoader(train_loader.dataset, batch_size=args.eval_batch_size, shuffle=False)
    train_result = infer(args.method, encoder, modules, train_eval, device); val_result = infer(args.method, encoder, modules, val_loader, device); known_result = infer(args.method, encoder, modules, known_loader, device)
    validation_score = scores_for(args.method, train_result, val_result, val_result); known_score = scores_for(args.method, train_result, val_result, known_result)

    rogue_classes = [int(v) for v in manifest["rogue_classes"]]; rogue_index = np.flatnonzero(np.isin(y, rogue_classes) & (role == 2))
    if len(rogue_index) != manifest["rogue_test_samples"]: raise ValueError("rogue role count does not match manifest")
    rogue_loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(x[rogue_index])), torch.from_numpy(y[rogue_index].astype(np.int64))), batch_size=args.eval_batch_size, shuffle=False)
    rogue_result = infer(args.method, encoder, modules, rogue_loader, device); rogue_score = scores_for(args.method, train_result, val_result, rogue_result)
    known_correct = known_result["logits"].argmax(1) == known_result["labels"]
    result_metrics = metric_row(f"{args.method}_known_only_score", validation_score, known_score, rogue_score, known_correct, args.known_reject_rate)
    result = {
        "method": args.method, "implementation": "protocol_adapted_reimplementation_v1", "benchmark": "multiknown17_v1_low_sample",
        "split": args.split, "seed": args.seed, "known_classes": known_classes, "rogue_classes": rogue_classes,
        "best_epoch": checkpoint["epoch"], "train_seconds": time.time() - start, "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256, "initialization": initialization,
        "metrics": result_metrics, "history": history,
        "protocol": {"encoder": "same NetTime nf=16, 64D", "preprocessing": "per-sample DC removal and joint IQ RMS", "training_and_selection": "known only", "checkpoint_selection": "known validation CE", "threshold": "known validation 90th percentile", "target_rogue_used_for_fit_selection_score_or_threshold": False},
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
