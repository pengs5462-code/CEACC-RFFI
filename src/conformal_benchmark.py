"""Class-conditional conformal calibration for known-only open-set RF rejection.

The encoder and classwise shrinkage covariance are fit on known training data.
Each known class then receives its own finite-sample calibration distribution from
known validation data. A sample is accepted if at least one class has conformal
p-value greater than alpha; no target-rogue example sets the operating threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_emitter_based as legacy
from posthoc_ablation import auc_rank, eer, oscr, roc_points
from score_benchmark import (
    KnownOnlyScorer,
    full_rogue_loader,
    normalize,
    resolve_data_seed,
)
from train_causal_ablation import build_model, extract, seed_everything


class MondrianMahalanobis:
    def __init__(
        self,
        train_embedding: np.ndarray,
        train_logits: np.ndarray,
        train_labels: np.ndarray,
        calibration_embedding: np.ndarray,
        calibration_labels: np.ndarray,
    ) -> None:
        scorer = KnownOnlyScorer(
            train_embedding, train_logits, train_labels, temperature=0.2, knn_k=5
        )
        self.prototypes = scorer.prototypes
        self.precisions = scorer.class_precisions
        calibration_distance = self.distance(calibration_embedding)
        self.reference = []
        for class_id in range(len(self.prototypes)):
            values = calibration_distance[calibration_labels == class_id, class_id]
            if len(values) < 2:
                raise ValueError(
                    f"class {class_id} has only {len(values)} calibration samples"
                )
            self.reference.append(np.sort(values))

    def distance(self, embedding: np.ndarray) -> np.ndarray:
        z = normalize(embedding)
        result = np.empty((len(z), len(self.prototypes)), dtype=np.float64)
        for class_id, precision in enumerate(self.precisions):
            delta = z - self.prototypes[class_id]
            result[:, class_id] = np.einsum(
                "nd,de,ne->n", delta, precision, delta, optimize=True
            )
        return result

    def p_values(self, embedding: np.ndarray) -> np.ndarray:
        distances = self.distance(embedding)
        p_values = np.empty_like(distances)
        for class_id, reference in enumerate(self.reference):
            # Large nonconformity distance is unlikely under the class. The +1
            # correction gives a valid finite-sample inductive conformal p-value.
            greater_or_equal = len(reference) - np.searchsorted(
                reference, distances[:, class_id], side="left"
            )
            p_values[:, class_id] = (greater_or_equal + 1.0) / (
                len(reference) + 1.0
            )
        return p_values


def result_row(
    known_p: np.ndarray,
    rogue_p: np.ndarray,
    known_labels: np.ndarray,
    alpha: float,
) -> dict:
    known_max = known_p.max(axis=1)
    rogue_max = rogue_p.max(axis=1)
    known_score = 1.0 - known_max
    rogue_score = 1.0 - rogue_max
    known_predictions = known_p.argmax(axis=1)
    known_correct = known_predictions == known_labels
    known_reject = known_max <= alpha
    rogue_reject = rogue_max <= alpha
    tp = int(rogue_reject.sum())
    fp = int(known_reject.sum())
    fn = int((~rogue_reject).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr, tpr, _ = roc_points(known_score, rogue_score)
    reaches = np.flatnonzero(tpr >= 0.95)
    return {
        "score": "mondrian_conformal_mahalanobis",
        "auc": auc_rank(
            np.r_[np.zeros(len(known_score), dtype=int), np.ones(len(rogue_score), dtype=int)],
            np.r_[known_score, rogue_score],
        ),
        "eer": eer(known_score, rogue_score),
        "fpr95": float(np.min(fpr[reaches])) if len(reaches) else 1.0,
        "known_closed_acc_conformal": float(known_correct.mean()),
        "known_accept_rate": float((~known_reject).mean()),
        "unknown_recall": float(recall),
        "unknown_f1": float(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
        "oscr": oscr(known_score, rogue_score, known_correct),
        "alpha_fixed_known_only": alpha,
        "threshold_unknown_score": 1.0 - alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-R", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--curve-alphas",
        type=float,
        nargs="*",
        default=[0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25],
        help="known-only operating points reported without selecting on rogue data",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-seed-override", type=int, default=None)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    data_seed = resolve_data_seed(model_args, args.data_seed_override)
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
    calibration_embedding, _, calibration_labels = extract(model, loaders[1], device)
    known_embedding, known_logits, known_labels = extract(model, loaders[2], device)
    rogue_embedding, _, rogue_labels = extract(
        model, full_rogue_loader(args.eval_R, args.batch_size), device
    )

    calibrator = MondrianMahalanobis(
        train_embedding,
        train_logits,
        train_labels,
        calibration_embedding,
        calibration_labels,
    )
    known_p = calibrator.p_values(known_embedding)
    rogue_p = calibrator.p_values(rogue_embedding)
    row = result_row(known_p, rogue_p, known_labels, args.alpha)
    operating_curve = [
        result_row(known_p, rogue_p, known_labels, alpha)
        for alpha in args.curve_alphas
    ]
    per_class = {}
    rogue_confusions = {}
    for class_id in sorted(np.unique(rogue_labels)):
        mask = rogue_labels == class_id
        key = str(int(class_id))
        per_class[key] = result_row(
            known_p, rogue_p[mask], known_labels, args.alpha
        )
        predicted_known = rogue_p[mask].argmax(axis=1)
        counts = np.bincount(predicted_known, minlength=rogue_p.shape[1])
        rogue_confusions[key] = {
            "most_likely_known_class_histogram": counts.astype(int).tolist(),
            "median_max_p_value": float(np.median(rogue_p[mask].max(axis=1))),
            "q90_max_p_value": float(np.quantile(rogue_p[mask].max(axis=1), 0.9)),
        }

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": int(model_args.seed),
        "data_seed": int(data_seed),
        "evaluation_R": args.eval_R,
        "network_known_closed_acc": float(
            (known_logits.argmax(axis=1) == known_labels).mean()
        ),
        "n_known_test": int(len(known_labels)),
        "n_rogue_test": int(len(rogue_labels)),
        "per_rogue_class_counts": {
            str(int(class_id)): int((rogue_labels == class_id).sum())
            for class_id in sorted(np.unique(rogue_labels))
        },
        "calibration_counts": [int(len(values)) for values in calibrator.reference],
        "protocol": {
            "encoder_and_covariance_fit": "known training only",
            "conformal_calibration": "classwise known validation only",
            "threshold": "fixed alpha; no target-rogue threshold fitting",
            "target_rogue_used_for_fit_selection_or_threshold": False,
        },
        "result": row,
        "known_only_operating_curve": operating_curve,
        "per_rogue_class": per_class,
        "rogue_to_known_diagnostics": rogue_confusions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
