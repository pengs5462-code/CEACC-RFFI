"""Official-code protocol adaptation of HyperRSI on multiknown17_v1.

The complex CNN, CosFace objective (s=8, m=0.2), hypersphere embedding,
prototype similarity, and 17-sample GPD tail follow the authors' released
ADS-B implementation.  Only dataset plumbing, checkpoint selection, metrics,
and the deployment threshold are adapted to the common known-only protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import genpareto
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-8


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def dataset_fingerprint(data_loader):
    digest = hashlib.sha256(); count = 0
    for value, target in DataLoader(data_loader.dataset, batch_size=128, shuffle=False):
        digest.update(np.ascontiguousarray(value.numpy()).tobytes())
        digest.update(np.ascontiguousarray(target.numpy()).tobytes()); count += len(target)
    return {"sha256": digest.hexdigest(), "samples": count}


def auc_rank(labels, scores):
    labels = labels.astype(np.int64); order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]; ranks = np.empty(len(scores), dtype=np.float64); start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]: end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0; start = end
    positives = labels.sum(); negatives = len(labels) - positives
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def roc_points(known, rogue):
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[known, rogue]))[::-1], -np.inf]
    return (np.asarray([np.mean(known >= t) for t in thresholds]),
            np.asarray([np.mean(rogue >= t) for t in thresholds]), thresholds)


def eer(known, rogue):
    fpr, tpr, _ = roc_points(known, rogue); index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    return float(0.5 * (fpr[index] + 1.0 - tpr[index]))


def oscr(known_scores, rogue_scores, known_correct):
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[known_scores, rogue_scores])), -np.inf]
    fpr = np.asarray([np.mean(rogue_scores <= t) for t in thresholds])
    ccr = np.asarray([np.mean(known_correct & (known_scores <= t)) for t in thresholds])
    order = np.argsort(fpr); return float(np.trapz(ccr[order], fpr[order]))


class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, value):
        real, imag = value.chunk(2, dim=1)
        return torch.cat((self.conv_re(real) - self.conv_im(imag),
                          self.conv_re(imag) + self.conv_im(real)), dim=1)


class HyperRSIEncoder(nn.Module):
    """The nine-layer complex CNN released for 4,800-sample ADS-B records."""

    def __init__(self):
        super().__init__()
        layers = []
        paddings = (0, 0, 1, 0, 0, 0, 0, 0, 0)
        in_channels = 1
        for padding in paddings:
            layers.append(nn.Sequential(
                ComplexConv(in_channels, 64, 3, padding=padding),
                nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2)))
            in_channels = 64
        self.layers = nn.ModuleList(layers)
        self.projection = nn.Linear(896, 512)

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return self.projection(value.reshape(len(value), -1))


class CosFace(nn.Module):
    def __init__(self, classes, scale=8.0, margin=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, 512))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.scale = scale
        self.margin = margin

    def logits(self, embedding):
        return F.linear(F.normalize(embedding, dim=1), F.normalize(self.weight, dim=1))

    def forward(self, embedding, target):
        cosine = self.logits(embedding)
        numerator = self.scale * (cosine.gather(1, target[:, None]).squeeze(1) - self.margin)
        mask = F.one_hot(target, cosine.shape[1]).bool()
        other = (self.scale * cosine).masked_fill(mask, float("-inf"))
        denominator = torch.logsumexp(torch.cat((numerator[:, None], other), dim=1), dim=1)
        return (denominator - numerator).mean(), cosine


class HyperRSI(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = HyperRSIEncoder()
        self.cosface = CosFace(classes)

    def forward(self, value, target=None):
        embedding = self.encoder(value)
        if target is None:
            return embedding, self.cosface.logits(embedding)
        loss, logits = self.cosface(embedding, target)
        return embedding, logits, loss


def loader(x, y, indices, mapping, batch, shuffle):
    values = np.asarray(x[indices], dtype=np.float32)
    # Stored samples are [N, 4800, 2]; the official complex CNN expects [N, 2, 4800].
    values = np.transpose(values, (0, 2, 1))
    # Use the common per-sample DC/RMS preprocessing applied by the paper protocol.
    values = values - values.mean(axis=2, keepdims=True)
    rms = np.sqrt(np.mean(values * values, axis=(1, 2), keepdims=True) + 1e-8)
    values = values / rms
    labels = np.asarray([mapping.get(int(v), int(v)) for v in y[indices]], dtype=np.int64)
    return DataLoader(TensorDataset(torch.from_numpy(values), torch.from_numpy(labels)),
                      batch_size=batch, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def extract(model, data_loader, device):
    model.eval(); embeddings, logits, labels = [], [], []
    for value, target in data_loader:
        value = value.to(device); embedding, score = model(value)
        embeddings.append(F.normalize(embedding, dim=1).cpu().numpy())
        logits.append(score.cpu().numpy()); labels.append(target.numpy())
    return np.concatenate(embeddings), np.concatenate(logits), np.concatenate(labels)


def fit_gpd(train_embedding, train_label, classes, tail_size):
    prototypes = np.stack([train_embedding[train_label == c].mean(0) for c in range(classes)])
    prototypes /= np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), EPS, None)
    true_similarity = np.sum(train_embedding * prototypes[train_label], axis=1)
    native = -true_similarity
    sorted_native = np.sort(native)[::-1]
    count = min(tail_size, len(sorted_native) - 1)
    threshold = float(sorted_native[count - 1])
    excess = np.clip(sorted_native[:count] - threshold, 0.0, None)
    shape, _, scale = genpareto.fit(excess, floc=0.0)
    return prototypes, {"shape": float(shape), "scale": float(scale),
                        "tail_threshold": threshold, "tail_size": int(count)}


def score(embedding, prototypes, gpd):
    similarity = embedding @ prototypes.T
    native = -similarity.max(1)
    excess = np.clip(native - gpd["tail_threshold"], 0.0, None)
    unknown = genpareto.cdf(excess, gpd["shape"], loc=0.0,
                            scale=max(gpd["scale"], EPS))
    unknown[native <= gpd["tail_threshold"]] = 0.0
    return unknown, similarity.argmax(1), native


def metrics(validation_score, known_score, rogue_score, known_correct, reject_rate):
    threshold = float(np.quantile(validation_score, 1.0 - reject_rate))
    known_reject = known_score > threshold; rogue_reject = rogue_score > threshold
    tp = int(rogue_reject.sum()); fp = int(known_reject.sum()); fn = int((~rogue_reject).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    fpr, tpr, _ = roc_points(known_score, rogue_score)
    reaches = np.flatnonzero(tpr >= 0.95)
    labels = np.r_[np.zeros(len(known_score), int), np.ones(len(rogue_score), int)]
    scores = np.r_[known_score, rogue_score]
    return {"auc": auc_rank(labels, scores), "eer": eer(known_score, rogue_score),
            "fpr95": float(np.min(fpr[reaches])) if len(reaches) else 1.0,
            "known_closed_acc": float(known_correct.mean()),
            "known_accept_rate": float((~known_reject).mean()),
            "unknown_recall": float(recall),
            "unknown_f1": float(2 * precision * recall / max(precision + recall, EPS)),
            "oscr": oscr(known_score, rogue_score, known_correct),
            "tau_known_only": threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tail-size", type=int, default=17)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    args = parser.parse_args()
    seed_everything(args.seed)
    root = Path(args.dataset_root); manifest = json.loads((root / f"split_{args.split}.json").read_text())
    known = list(map(int, manifest["known_classes"])); rogue = list(map(int, manifest["rogue_classes"]))
    mapping = {value: index for index, value in enumerate(known)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(root / "Y_global.npy"); role = np.load(root / "sample_role.npy")
    known_mask = np.isin(y, known)
    indices = {"train": np.flatnonzero(known_mask & (role == 0)),
               "validation": np.flatnonzero(known_mask & (role == 1)),
               "known": np.flatnonzero(known_mask & (role == 2)),
               "rogue": np.flatnonzero(np.isin(y, rogue) & (role == 2))}
    train = loader(x, y, indices["train"], mapping, args.batch_size, True)
    validation = loader(x, y, indices["validation"], mapping, 128, False)
    known_test = loader(x, y, indices["known"], mapping, 128, False)
    rogue_test = loader(x, y, indices["rogue"], {}, 128, False)
    fingerprints = {name: dataset_fingerprint(item) for name, item in
                    (("known_train", train), ("known_validation", validation), ("known_test", known_test))}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HyperRSI(len(known)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "best.pt"; best = float("inf"); stale = 0; history = []; start = time.time()
    for epoch in range(args.epochs):
        model.train(); total_loss = total_correct = total = 0
        for value, target in train:
            value, target = value.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits, loss = model(value, target); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(target); total_correct += (logits.argmax(1) == target).sum().item(); total += len(target)
        model.eval(); val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for value, target in validation:
                value, target = value.to(device), target.to(device)
                _, logits, loss = model(value, target)
                val_loss += loss.item() * len(target); val_correct += (logits.argmax(1) == target).sum().item(); val_total += len(target)
        val_loss /= val_total
        row = {"epoch": epoch + 1, "train_loss": total_loss / total, "train_acc": total_correct / total,
               "val_loss": val_loss, "val_acc": val_correct / val_total}
        history.append(row); print(json.dumps(row), flush=True)
        if val_loss < best - 1e-5:
            best = val_loss; stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss,
                        "manifest": manifest, "fingerprints": fingerprints, "args": vars(args)}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience: break
    saved = torch.load(checkpoint, map_location=device); model.load_state_dict(saved["model"])
    train_e, _, train_y = extract(model, DataLoader(train.dataset, batch_size=128), device)
    val_e, _, _ = extract(model, validation, device)
    known_e, _, known_y = extract(model, known_test, device)
    rogue_e, _, rogue_y = extract(model, rogue_test, device)
    prototypes, gpd = fit_gpd(train_e, train_y, len(known), args.tail_size)
    val_score, _, val_native = score(val_e, prototypes, gpd)
    known_score, known_pred, known_native = score(known_e, prototypes, gpd)
    rogue_score, _, rogue_native = score(rogue_e, prototypes, gpd)
    known_correct = known_pred == known_y
    result_metrics = metrics(val_score, known_score, rogue_score, known_correct, args.known_reject_rate)
    np.savez_compressed(out / "per_sample_scores.npz", validation_score=val_score,
                        validation_native=val_native, known_score=known_score, known_native=known_native,
                        known_label=known_y, known_prediction=known_pred, rogue_score=rogue_score,
                        rogue_native=rogue_native, rogue_label=rogue_y)
    result = {"method": "HyperRSI", "implementation": "official_code_protocol_adaptation_v1",
              "split": args.split, "seed": args.seed, "known_classes": known, "rogue_classes": rogue,
              "best_epoch": saved["epoch"], "train_seconds": time.time() - start,
              "split_fingerprints": fingerprints, "gpd": gpd, "metrics": result_metrics, "history": history,
              "protocol": {"architecture": "official nine-layer complex CNN, 512D",
                           "objective": "official CosFace s=8 m=0.2", "tail_size": args.tail_size,
                           "checkpoint_selection": "minimum known-validation CosFace loss",
                           "operating_threshold": "known-validation Q90",
                           "target_rogue_used_for_fit_selection_or_threshold": False},
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "result.json").write_text(json.dumps(result, indent=2)); print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
