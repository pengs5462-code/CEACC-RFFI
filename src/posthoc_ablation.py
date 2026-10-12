"""Inference-time dependency diagnosis for a legacy FRAME checkpoint.

This is not a replacement for retrained ablations.  It answers the narrower question:
which trained FRAE paths does the existing checkpoint rely on at inference time?
All variants use the same known-only threshold calibration and the same held-out test set.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path

import numpy as np
import torch

import train_emitter_based as legacy
from model.MRAN import NetTime


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    positives = labels.sum()
    negatives = len(labels) - positives
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def roc_points(known: np.ndarray, rogue: np.ndarray):
    labels = np.r_[np.zeros(len(known), dtype=int), np.ones(len(rogue), dtype=int)]
    scores = np.r_[known, rogue]
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    fpr, tpr = [], []
    for tau in thresholds:
        fpr.append(float(np.mean(known >= tau)))
        tpr.append(float(np.mean(rogue >= tau)))
    return np.asarray(fpr), np.asarray(tpr), thresholds


def eer(known: np.ndarray, rogue: np.ndarray) -> float:
    fpr, tpr, _ = roc_points(known, rogue)
    index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    return float(0.5 * (fpr[index] + 1.0 - tpr[index]))


def oscr(
    known_scores: np.ndarray,
    rogue_scores: np.ndarray,
    known_correct: np.ndarray,
) -> float:
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[known_scores, rogue_scores])), -np.inf]
    fpr, ccr = [], []
    for tau in thresholds:
        fpr.append(float(np.mean(rogue_scores <= tau)))
        ccr.append(float(np.mean(known_correct & (known_scores <= tau))))
    order = np.argsort(fpr)
    return float(np.trapz(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def extract(model, loader, device):
    embeddings, logits, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            if data.ndim == 3 and data.shape[-1] == 2:
                data = data.permute(0, 2, 1)
            embedding, output_logits = model(data.to(device, dtype=torch.float32))
            embeddings.append(embedding.cpu().numpy())
            logits.append(output_logits.cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(embeddings), np.concatenate(logits), np.concatenate(labels)


def centers(embedding: np.ndarray, labels: np.ndarray) -> np.ndarray:
    embedding = embedding / np.clip(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12, None)
    result = []
    for class_id in sorted(np.unique(labels)):
        center = embedding[labels == class_id].mean(axis=0)
        result.append(center / max(np.linalg.norm(center), 1e-12))
    return np.asarray(result)


def unknownness(embedding: np.ndarray, prototypes: np.ndarray, temperature: float):
    embedding = embedding / np.clip(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12, None)
    distances = np.linalg.norm(embedding[:, None] - prototypes[None], axis=-1)
    logits = -distances / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return np.sum(probability * distances, axis=1)


def install_variant(model, variant: str):
    fexp = model.fexp
    if variant == "as_trained":
        return

    def bypass(self, x):
        return x

    def no_expert(self, x):
        high, low = self.spilit(x)
        return self.fusion_gate(x, low, high)

    def equal_fusion(self, x):
        high, low = self.spilit(x)
        high = self.high_block(high)
        low = self.low_block(low)
        return (x + low + high) / 3.0

    def high_only(self, x):
        high, _ = self.spilit(x)
        return self.high_block(high)

    def low_only(self, x):
        _, low = self.spilit(x)
        return self.low_block(low)

    implementations = {
        "bypass_frae": bypass,
        "no_expert": no_expert,
        "equal_fusion": equal_fusion,
        "high_only": high_only,
        "low_only": low_only,
    }
    fexp.forward = types.MethodType(implementations[variant], fexp)


def evaluate_variant(args, variant, loaders, checkpoint, device):
    train_loader, val_loader, known_loader, _, rogue_loader = loaders
    model = NetTime(c_in=2, c_out=checkpoint.get("n_classes", 9), nf=16).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    install_variant(model, variant)

    train_embedding, _, train_labels = extract(model, train_loader, device)
    val_embedding, val_logits, val_labels = extract(model, val_loader, device)
    known_embedding, known_logits, known_labels = extract(model, known_loader, device)
    rogue_embedding, _, _ = extract(model, rogue_loader, device)
    prototypes = centers(train_embedding, train_labels)

    val_scores = unknownness(val_embedding, prototypes, args.temperature)
    known_scores = unknownness(known_embedding, prototypes, args.temperature)
    rogue_scores = unknownness(rogue_embedding, prototypes, args.temperature)
    tau = float(np.quantile(val_scores, 1.0 - args.known_reject_rate))
    known_predictions = known_logits.argmax(axis=1)
    known_correct = known_predictions == known_labels
    unknown_predictions = rogue_scores > tau
    known_false_reject = known_scores > tau
    tp = int(unknown_predictions.sum())
    fp = int(known_false_reject.sum())
    fn = len(rogue_scores) - tp
    unknown_precision = tp / max(tp + fp, 1)
    unknown_recall = tp / max(tp + fn, 1)
    unknown_f1 = 2 * unknown_precision * unknown_recall / max(
        unknown_precision + unknown_recall, 1e-12
    )
    labels = np.r_[np.zeros(len(known_scores), dtype=int), np.ones(len(rogue_scores), dtype=int)]
    scores = np.r_[known_scores, rogue_scores]
    return {
        "variant": variant,
        "auc": auc_rank(labels, scores),
        "eer": eer(known_scores, rogue_scores),
        "known_closed_acc": float(known_correct.mean()),
        "known_accept_rate": float((~known_false_reject).mean()),
        "unknown_recall": float(unknown_recall),
        "unknown_f1": float(unknown_f1),
        "oscr": oscr(known_scores, rogue_scores, known_correct),
        "tau_known_only": tau,
        "n_known_test": len(known_scores),
        "n_rogue_test": len(rogue_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    legacy.setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = legacy.EmitterBasedDataset(R=args.R)
    loaders = legacy.create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        rogue_val_split=0.5,
        seed=args.seed,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variants = [
        "as_trained",
        "bypass_frae",
        "no_expert",
        "equal_fusion",
        "high_only",
        "low_only",
    ]
    results = []
    for variant in variants:
        print(f"\nEvaluating {variant}", flush=True)
        result = evaluate_variant(args, variant, loaders, checkpoint, device)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    baseline = results[0]
    for result in results:
        result["delta_auc_vs_as_trained"] = result["auc"] - baseline["auc"]
        result["delta_oscr_vs_as_trained"] = result["oscr"] - baseline["oscr"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

