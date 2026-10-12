"""Audited SciPy-EVT implementation of the classic OpenMax baseline.

The implementation follows Bendale & Boult's released algorithm: class mean
activation vectors are formed from correctly classified known-training samples,
the largest 20 eucos distances fit a Weibull tail, and the top three activations
are recalibrated.  The deployment threshold is calibrated only on known
validation samples; target rogue samples are final evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import weibull_min

import train_emitter_based as legacy
from score_benchmark import full_rogue_loader, metric_row, resolve_data_seed
from train_causal_ablation import build_model, extract, seed_everything
from train_representation import dataset_fingerprint


EPS = 1e-12


def eucos(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    euclidean = np.linalg.norm(x - mean[None, :], axis=1) / 200.0
    denominator = np.linalg.norm(x, axis=1) * max(np.linalg.norm(mean), EPS)
    cosine = 1.0 - (x @ mean) / np.clip(denominator, EPS, None)
    return euclidean + cosine


class OpenMax:
    def __init__(self, train_logits, train_labels, tail_size=20, alpha_rank=3):
        classes = int(train_logits.shape[1])
        if alpha_rank < 1 or alpha_rank > classes:
            raise ValueError("alpha_rank must be in [1, number of classes]")
        train_prediction = train_logits.argmax(axis=1)
        self.means = []
        self.weibull = []
        self.correct_counts = []
        self.tail_counts = []
        self.alpha_rank = alpha_rank
        for class_id in range(classes):
            mask = (train_labels == class_id) & (train_prediction == class_id)
            values = train_logits[mask]
            if len(values) < 3:
                raise ValueError(f"class {class_id} has too few correctly classified samples")
            mean = values.mean(axis=0)
            distance = np.sort(eucos(values, mean))
            count = min(tail_size, len(distance))
            tail = np.clip(distance[-count:], 1e-8, None)
            shape, location, scale = weibull_min.fit(tail, floc=0.0)
            self.means.append(mean)
            self.weibull.append((float(shape), float(location), float(scale)))
            self.correct_counts.append(int(len(values)))
            self.tail_counts.append(int(count))
        self.means = np.asarray(self.means)

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        n, classes = logits.shape
        distance = np.column_stack([
            eucos(logits, self.means[class_id]) for class_id in range(classes)
        ])
        weibull_score = np.column_stack([
            weibull_min.cdf(distance[:, class_id], *self.weibull[class_id])
            for class_id in range(classes)
        ])
        rank = np.argsort(logits, axis=1)[:, ::-1]
        alpha = np.zeros_like(logits, dtype=np.float64)
        for position in range(self.alpha_rank):
            weight = (self.alpha_rank - position) / float(self.alpha_rank)
            alpha[np.arange(n), rank[:, position]] = weight
        removed = logits * weibull_score * alpha
        revised = logits - removed
        unknown = removed.sum(axis=1, keepdims=True)
        joined = np.concatenate([revised, unknown], axis=1)
        joined -= joined.max(axis=1, keepdims=True)
        probability = np.exp(joined)
        return probability / np.clip(probability.sum(axis=1, keepdims=True), EPS, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-R", type=int, default=8)
    parser.add_argument("--tail-size", type=int, default=20)
    parser.add_argument("--alpha-rank", type=int, default=3)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    data_seed = resolve_data_seed(model_args, None)
    seed_everything(data_seed)
    dataset = legacy.EmitterBasedDataset(R=model_args.R)
    loaders = legacy.create_dataloaders(
        dataset, batch_size=args.batch_size, val_split=0.2,
        rogue_val_split=0.5, seed=data_seed
    )
    fingerprints = {
        name: dataset_fingerprint(loader)
        for name, loader in zip(
            ("known_train", "known_validation", "known_test"), loaders[:3]
        )
    }
    if checkpoint.get("split_fingerprints") != fingerprints:
        raise RuntimeError("checkpoint/data split fingerprint mismatch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(model_args.seed)
    model = build_model(model_args, device)
    model.load_state_dict(checkpoint["model"])
    _, train_logits, train_y = extract(model, loaders[0], device)
    _, val_logits, _ = extract(model, loaders[1], device)
    _, known_logits, known_y = extract(model, loaders[2], device)
    _, rogue_logits, rogue_y = extract(
        model, full_rogue_loader(args.eval_R, args.batch_size), device
    )

    openmax = OpenMax(train_logits, train_y, args.tail_size, args.alpha_rank)
    val_probability = openmax.probabilities(val_logits)
    known_probability = openmax.probabilities(known_logits)
    rogue_probability = openmax.probabilities(rogue_logits)
    val_score = val_probability[:, -1]
    known_score = known_probability[:, -1]
    rogue_score = rogue_probability[:, -1]
    known_correct = known_probability[:, :-1].argmax(axis=1) == known_y
    result = metric_row(
        "openmax_unknown_probability", val_score, known_score, rogue_score,
        known_correct, args.known_reject_rate
    )
    result["known_closed_acc_openmax"] = float(known_correct.mean())
    per_rogue_class = {
        str(int(class_id)): metric_row(
            "openmax_unknown_probability", val_score, known_score,
            rogue_score[rogue_y == class_id], known_correct,
            args.known_reject_rate
        )
        for class_id in sorted(np.unique(rogue_y))
    }
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": int(model_args.seed),
        "data_seed": int(data_seed),
        "evaluation_R": args.eval_R,
        "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256,
        "correct_train_counts": openmax.correct_counts,
        "weibull_tail_counts": openmax.tail_counts,
        "protocol": {
            "mav_and_weibull_fit": "correctly classified known training only",
            "distance": "released OpenMax eucos: euclidean/200 + cosine",
            "tail_size": args.tail_size,
            "alpha_rank": args.alpha_rank,
            "operating_threshold": "known-validation 90th percentile",
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "implementation_note": "SciPy MLE Weibull replaces LibMR",
            "additional_trainable_parameters": 0,
        },
        "result": result,
        "per_rogue_class": per_rogue_class,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
