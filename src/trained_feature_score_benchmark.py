"""Known-only OSR score benchmark on the actually trained backbone feature.

The legacy CE checkpoints return a 64-D contrast-head embedding that is not
connected to the training loss.  This runner instead captures the 512-D tensor
feeding that head, fits a fixed 64-D PCA on known training samples only, and
calibrates every operating threshold on known validation samples only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

import train_emitter_based as legacy
from score_benchmark import (
    KnownOnlyScorer,
    full_rogue_loader,
    metric_row,
    normalize,
    resolve_data_seed,
    robust_z,
)
from train_causal_ablation import build_model, seed_everything
from train_representation import dataset_fingerprint
from trained_feature_conformal import extract_trained_feature


EPS = 1e-12


class ClassLocalKNN:
    """Minimum class-standardized local kNN distance, fitted on known train."""

    def __init__(self, train_embedding: np.ndarray, labels: np.ndarray, k: int = 5):
        self.train = normalize(train_embedding)
        self.labels = labels.astype(int)
        self.classes = np.unique(self.labels)
        self.k = int(k)
        self.by_class: dict[int, np.ndarray] = {}
        self.location: dict[int, float] = {}
        self.scale: dict[int, float] = {}
        for class_id in self.classes:
            x = self.train[self.labels == class_id]
            if len(x) <= self.k:
                raise ValueError(f"class {class_id} has only {len(x)} samples for k={self.k}")
            self.by_class[int(class_id)] = x
            distance = 1.0 - x @ x.T
            np.fill_diagonal(distance, np.inf)
            reference = np.partition(distance, self.k - 1, axis=1)[:, : self.k].mean(1)
            median = float(np.median(reference))
            q25, q75 = np.quantile(reference, [0.25, 0.75])
            self.location[int(class_id)] = median
            self.scale[int(class_id)] = max(float((q75 - q25) / 1.349), 1e-6)

    def score(self, embedding: np.ndarray) -> np.ndarray:
        z = normalize(embedding)
        class_scores = []
        for class_id in self.classes:
            reference = self.by_class[int(class_id)]
            distance = 1.0 - z @ reference.T
            local = np.partition(distance, self.k - 1, axis=1)[:, : self.k].mean(1)
            standardized = (
                local - self.location[int(class_id)]
            ) / self.scale[int(class_id)]
            class_scores.append(standardized)
        return np.stack(class_scores, axis=1).min(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-R", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
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
        dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        rogue_val_split=0.5,
        seed=data_seed,
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

    train_f, train_logits, train_y = extract_trained_feature(model, loaders[0], device)
    val_f, val_logits, _ = extract_trained_feature(model, loaders[1], device)
    known_f, known_logits, known_y = extract_trained_feature(model, loaders[2], device)

    pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=0)
    train_e = pca.fit_transform(train_f)
    val_e = pca.transform(val_f)
    known_e = pca.transform(known_f)

    scorer = KnownOnlyScorer(
        train_e, train_logits, train_y, args.temperature, args.knn_k
    )
    local_knn = ClassLocalKNN(train_e, train_y, args.knn_k)
    val_scores = scorer.score(val_e, val_logits)
    known_scores = scorer.score(known_e, known_logits)

    # Materialize target rogue only after PCA/scorers/threshold references are frozen.
    rogue_loader = full_rogue_loader(args.eval_R, args.batch_size)
    rogue_f, rogue_logits, rogue_y = extract_trained_feature(model, rogue_loader, device)
    rogue_e = pca.transform(rogue_f)
    rogue_scores = scorer.score(rogue_e, rogue_logits)
    local_name = f"min_class_local_knn_z_k{args.knn_k}"
    val_scores[local_name] = local_knn.score(val_e)
    known_scores[local_name] = local_knn.score(known_e)
    rogue_scores[local_name] = local_knn.score(rogue_e)

    # Fixed, equal-weight diagnostic combinations; validation statistics alone
    # define normalization.  They are reported, never target-selected here.
    members = [
        "min_cosine_distance",
        f"knn_cosine_k{scorer.knn_k}",
        "shared_mahalanobis",
        local_name,
    ]
    for split_scores, reference in (
        (val_scores, val_scores),
        (known_scores, val_scores),
        (rogue_scores, val_scores),
    ):
        split_scores["trained_feature_embedding_consensus"] = np.mean(
            [robust_z(reference[name], split_scores[name]) for name in members], axis=0
        )

    known_correct = known_logits.argmax(1) == known_y
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
    per_class = {}
    for class_id in sorted(np.unique(rogue_y)):
        mask = rogue_y == class_id
        per_class[str(int(class_id))] = [
            metric_row(
                name,
                val_scores[name],
                known_scores[name],
                rogue_scores[name][mask],
                known_correct,
                args.known_reject_rate,
            )
            for name in val_scores
        ]

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "variant": model_args.variant,
        "seed": int(model_args.seed),
        "data_seed": int(data_seed),
        "evaluation_R": int(args.eval_R),
        "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256,
        "known_closed_acc": float(known_correct.mean()),
        "feature": {
            "source": "trained 512-D tensor feeding contrast_head",
            "legacy_untrained_contrast_head_used": False,
            "pca_dim": int(args.pca_dim),
            "pca_fit": "known training only",
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        },
        "protocol": {
            "encoder_training_and_selection": "known only",
            "score_fit": "known training only",
            "threshold": f"known validation {1.0-args.known_reject_rate:.3f} quantile",
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "target_rogue_materialized": "after feature transforms and scorer fits were frozen",
            "pca_dim_and_knn_k": "fixed a priori at 64 and 5",
            "results_order": "predefined diagnostic order; not sorted by target AUC",
        },
        "results": rows,
        "per_rogue_class": per_class,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
