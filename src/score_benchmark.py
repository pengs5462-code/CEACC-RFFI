"""Leakage-free open-set score benchmark for a trained causal-ablation model.

Every score is fitted (when necessary) on known training data and every operating
threshold is calibrated on known validation data.  Target rogue samples are used
only once for final reporting; they never determine score direction, temperature,
hyperparameters, ensemble weights, or thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, TensorDataset

import train_emitter_based as legacy

from posthoc_ablation import auc_rank, eer, oscr, roc_points, centers, unknownness
from train_causal_ablation import build_model, extract, seed_everything


EPS = 1e-12


def resolve_data_seed(model_args, override: int | None = None) -> int:
    """Require explicit split provenance instead of silently guessing it."""
    recorded = getattr(model_args, "data_seed", None)
    if recorded is None and override is None:
        raise ValueError(
            "checkpoint lacks data_seed; make a provenance-preserving metadata "
            "copy or pass --data-seed-override with documented evidence"
        )
    if recorded is not None and override is not None and int(recorded) != override:
        raise ValueError(
            f"data-seed override {override} conflicts with checkpoint value {recorded}"
        )
    return int(recorded if recorded is not None else override)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), EPS, None)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)


def robust_z(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    median = np.median(reference)
    q25, q75 = np.quantile(reference, [0.25, 0.75])
    # IQR / 1.349 estimates the standard deviation for a Gaussian while being
    # substantially less sensitive to hard known validation examples.
    scale = max((q75 - q25) / 1.349, 1e-6)
    return (value - median) / scale


class KnownOnlyScorer:
    def __init__(
        self,
        train_embedding: np.ndarray,
        train_logits: np.ndarray,
        train_labels: np.ndarray,
        temperature: float,
        knn_k: int,
    ) -> None:
        del train_logits
        self.train = normalize(train_embedding)
        self.labels = train_labels.astype(int)
        self.prototypes = centers(train_embedding, train_labels)
        self.temperature = temperature
        self.knn_k = min(knn_k, len(self.train))

        residuals = self.train - self.prototypes[self.labels]
        covariance = LedoitWolf(assume_centered=True).fit(residuals)
        self.precision = covariance.precision_.astype(np.float64)

        train_proto_distance = 1.0 - np.sum(
            self.train * self.prototypes[self.labels], axis=1
        )
        self.class_distance = {
            int(class_id): np.sort(train_proto_distance[self.labels == class_id])
            for class_id in np.unique(self.labels)
        }
        self.class_distance_location = np.asarray(
            [np.median(self.class_distance[class_id]) for class_id in range(len(self.prototypes))]
        )
        self.class_distance_scale = np.asarray(
            [
                max(
                    (
                        np.quantile(self.class_distance[class_id], 0.75)
                        - np.quantile(self.class_distance[class_id], 0.25)
                    )
                    / 1.349,
                    1e-6,
                )
                for class_id in range(len(self.prototypes))
            ]
        )

        self.class_precisions = []
        self.class_mahalanobis_location = []
        self.class_mahalanobis_scale = []
        for class_id in range(len(self.prototypes)):
            class_embedding = self.train[self.labels == class_id]
            class_residual = class_embedding - self.prototypes[class_id]
            precision = LedoitWolf(assume_centered=True).fit(class_residual).precision_
            reference = np.einsum(
                "nd,de,ne->n", class_residual, precision, class_residual, optimize=True
            )
            self.class_precisions.append(precision.astype(np.float64))
            self.class_mahalanobis_location.append(float(np.median(reference)))
            self.class_mahalanobis_scale.append(
                max(
                    float((np.quantile(reference, 0.75) - np.quantile(reference, 0.25)) / 1.349),
                    1e-6,
                )
            )
        self.class_mahalanobis_location = np.asarray(self.class_mahalanobis_location)
        self.class_mahalanobis_scale = np.asarray(self.class_mahalanobis_scale)

    def score(self, embedding: np.ndarray, logits: np.ndarray) -> dict[str, np.ndarray]:
        z = normalize(embedding)
        cosine_distance = 1.0 - z @ self.prototypes.T
        nearest_class = cosine_distance.argmin(axis=1)

        train_distance = 1.0 - z @ self.train.T
        knn = np.partition(train_distance, self.knn_k - 1, axis=1)[:, : self.knn_k]

        delta = z[:, None, :] - self.prototypes[None, :, :]
        mahalanobis_all = np.einsum(
            "ncd,de,nce->nc", delta, self.precision, delta, optimize=True
        )

        class_percentile = np.empty(len(z), dtype=np.float64)
        all_class_percentile = np.empty_like(cosine_distance, dtype=np.float64)
        nearest_distance = cosine_distance[np.arange(len(z)), nearest_class]
        for index, (class_id, distance) in enumerate(
            zip(nearest_class, nearest_distance)
        ):
            reference = self.class_distance[int(class_id)]
            class_percentile[index] = np.searchsorted(
                reference, distance, side="right"
            ) / len(reference)
        for class_id in range(cosine_distance.shape[1]):
            reference = self.class_distance[class_id]
            all_class_percentile[:, class_id] = np.searchsorted(
                reference, cosine_distance[:, class_id], side="right"
            ) / len(reference)

        class_mahalanobis = np.empty_like(cosine_distance, dtype=np.float64)
        for class_id, precision in enumerate(self.class_precisions):
            class_delta = z - self.prototypes[class_id]
            class_mahalanobis[:, class_id] = np.einsum(
                "nd,de,ne->n", class_delta, precision, class_delta, optimize=True
            )
        cosine_robust_z = (
            cosine_distance - self.class_distance_location[None, :]
        ) / self.class_distance_scale[None, :]
        mahalanobis_robust_z = (
            class_mahalanobis - self.class_mahalanobis_location[None, :]
        ) / self.class_mahalanobis_scale[None, :]

        probability = softmax(logits)
        sorted_logits = np.sort(logits, axis=1)
        return {
            "softmin_expected_distance": unknownness(
                embedding, self.prototypes, self.temperature
            ),
            "min_cosine_distance": cosine_distance.min(axis=1),
            f"knn_cosine_k{self.knn_k}": knn.mean(axis=1),
            "shared_mahalanobis": mahalanobis_all.min(axis=1),
            "class_distance_percentile": class_percentile,
            "min_class_cosine_percentile": all_class_percentile.min(axis=1),
            "min_class_cosine_robust_z": cosine_robust_z.min(axis=1),
            "class_conditional_mahalanobis": class_mahalanobis.min(axis=1),
            "min_class_mahalanobis_robust_z": mahalanobis_robust_z.min(axis=1),
            "msp": 1.0 - probability.max(axis=1),
            "negative_max_logit": -logits.max(axis=1),
            "energy": -np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1))
            - logits.max(axis=1),
            "logit_negative_margin": sorted_logits[:, -2] - sorted_logits[:, -1],
            "predictive_entropy": -np.sum(
                probability * np.log(np.clip(probability, EPS, None)), axis=1
            ),
        }


def metric_row(
    name: str,
    val: np.ndarray,
    known: np.ndarray,
    rogue: np.ndarray,
    known_correct: np.ndarray,
    reject_rate: float,
) -> dict[str, float | str]:
    tau = float(np.quantile(val, 1.0 - reject_rate))
    known_reject = known > tau
    rogue_reject = rogue > tau
    tp = int(rogue_reject.sum())
    fp = int(known_reject.sum())
    fn = len(rogue_reject) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    labels = np.r_[np.zeros(len(known), dtype=int), np.ones(len(rogue), dtype=int)]
    scores = np.r_[known, rogue]
    fpr_curve, tpr_curve, _ = roc_points(known, rogue)
    reaches_95 = np.flatnonzero(tpr_curve >= 0.95)
    fpr95 = float(np.min(fpr_curve[reaches_95])) if len(reaches_95) else 1.0
    return {
        "score": name,
        "auc": auc_rank(labels, scores),
        "eer": eer(known, rogue),
        "fpr95": fpr95,
        "known_accept_rate": float((~known_reject).mean()),
        "unknown_recall": float(recall),
        "unknown_f1": float(f1),
        "oscr": oscr(known, rogue, known_correct),
        "tau_known_only": tau,
    }


def full_rogue_loader(eval_r: int, batch_size: int) -> DataLoader:
    """Load every target-rogue sample in deterministic on-disk order."""
    root = Path("dataset") / "emitter_based_splits_balanced" / f"9_{eval_r}rogue"
    x = np.load(root / "X_test_rogue.npy")
    global_labels = root / "Y_test_rogue_global.npy"
    y = np.load(global_labels if global_labels.exists() else root / "Y_test_rogue.npy")
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument(
        "--primary-score",
        default="min_class_mahalanobis_robust_z",
        help="score fixed before target-rogue evaluation; never selected by test AUC",
    )
    parser.add_argument(
        "--eval-R",
        type=int,
        default=None,
        help="rogue set used only for final evaluation; defaults to checkpoint R",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-seed-override", type=int, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    data_seed = resolve_data_seed(model_args, args.data_seed_override)
    # Legacy create_dataloaders consumes global RNG and ignores its seed kwarg.
    seed_everything(data_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = legacy.EmitterBasedDataset(R=model_args.R)
    loaders = legacy.create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        rogue_val_split=0.5,
        seed=data_seed,
    )
    seed_everything(model_args.seed)
    model = build_model(model_args, device)
    model.load_state_dict(checkpoint["model"])

    train_embedding, train_logits, train_labels = extract(model, loaders[0], device)
    val_embedding, val_logits, _ = extract(model, loaders[1], device)
    known_embedding, known_logits, known_labels = extract(model, loaders[2], device)
    eval_r = args.eval_R if args.eval_R is not None else model_args.R
    rogue_loader = full_rogue_loader(eval_r, args.batch_size)
    rogue_embedding, rogue_logits, rogue_labels = extract(model, rogue_loader, device)

    scorer = KnownOnlyScorer(
        train_embedding,
        train_logits,
        train_labels,
        args.temperature,
        args.knn_k,
    )
    val_scores = scorer.score(val_embedding, val_logits)
    known_scores = scorer.score(known_embedding, known_logits)
    rogue_scores = scorer.score(rogue_embedding, rogue_logits)

    ensemble_members = {
        "embedding_consensus": [
            "min_cosine_distance",
            f"knn_cosine_k{scorer.knn_k}",
            "shared_mahalanobis",
        ],
        "embedding_logit_consensus": [
            "min_cosine_distance",
            f"knn_cosine_k{scorer.knn_k}",
            "shared_mahalanobis",
            "msp",
            "energy",
        ],
    }
    for ensemble_name, members in ensemble_members.items():
        val_scores[ensemble_name] = np.mean(
            [robust_z(val_scores[name], val_scores[name]) for name in members], axis=0
        )
        known_scores[ensemble_name] = np.mean(
            [robust_z(val_scores[name], known_scores[name]) for name in members], axis=0
        )
        rogue_scores[ensemble_name] = np.mean(
            [robust_z(val_scores[name], rogue_scores[name]) for name in members], axis=0
        )

    known_predictions = known_logits.argmax(axis=1)
    known_correct = known_predictions == known_labels
    rows = [
        metric_row(
            name,
            val_scores[name],
            known_scores[name],
            rogue_scores[name],
            known_correct,
            args.known_reject_rate,
        )
        for name in val_scores
    ]
    primary_matches = [row for row in rows if row["score"] == args.primary_score]
    if len(primary_matches) != 1:
        raise ValueError(f"unknown or duplicate primary score: {args.primary_score!r}")
    per_rogue_class = {}
    per_rogue_class_counts = {}
    for class_id in sorted(np.unique(rogue_labels)):
        class_mask = rogue_labels == class_id
        per_rogue_class_counts[str(int(class_id))] = int(class_mask.sum())
        per_rogue_class[str(int(class_id))] = [
            metric_row(
                name,
                val_scores[name],
                known_scores[name],
                rogue_scores[name][class_mask],
                known_correct,
                args.known_reject_rate,
            )
            for name in val_scores
        ]
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "variant": model_args.variant,
        "seed": model_args.seed,
        "data_seed": data_seed,
        "training_R_metadata_only": model_args.R,
        "evaluation_R": eval_r,
        "known_closed_acc": float(known_correct.mean()),
        "n_known_test": int(len(known_labels)),
        "n_rogue_test": int(len(rogue_labels)),
        "per_rogue_class_counts": per_rogue_class_counts,
        "primary_score": args.primary_score,
        "primary_result": primary_matches[0],
        "protocol": {
            "fit": "known training only",
            "threshold": f"known validation {1.0-args.known_reject_rate:.3f} quantile",
            "target_rogue_used_for_selection": False,
            "target_rogue_final_evaluation": "legacy rogue-val and rogue-test merged",
            "score_direction_fixed_a_priori": "larger means more unknown",
            "score_selection": "primary score fixed before target-rogue evaluation",
            "results_order": "predefined diagnostic order; not sorted by target AUC",
            "knn_k": scorer.knn_k,
            "temperature": args.temperature,
        },
        "results": rows,
        "per_rogue_class": per_rogue_class,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
