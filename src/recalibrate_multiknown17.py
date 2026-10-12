"""Independent known-only LOO operating-threshold calibration for conformal OSR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from conformal_benchmark import MondrianMahalanobis
from posthoc_ablation import auc_rank, eer, oscr, roc_points
from spectral_conformal_benchmark import deterministic
from train_causal_ablation import build_model, extract, seed_everything
from train_multiknown17_baseline import make_loader
from train_representation import dataset_fingerprint


def leave_one_out_validation_p(calibrator, feature, label):
    distance = calibrator.distance(feature)
    p_values = np.empty_like(distance)
    for i in range(len(feature)):
        for class_id, reference in enumerate(calibrator.reference):
            count_ge = len(reference) - np.searchsorted(
                reference, distance[i, class_id], side="left"
            )
            if class_id == int(label[i]):
                # The current validation observation is present exactly once in
                # its true-class reference. Remove it, then apply the +1 rule.
                p_values[i, class_id] = count_ge / len(reference)
            else:
                p_values[i, class_id] = (count_ge + 1.0) / (len(reference) + 1.0)
    return p_values


def metric_row(known_p, rogue_p, known_labels, threshold):
    known_score = 1.0 - known_p.max(axis=1)
    rogue_score = 1.0 - rogue_p.max(axis=1)
    known_prediction = known_p.argmax(axis=1)
    known_correct = known_prediction == known_labels
    known_reject = known_score > threshold
    rogue_reject = rogue_score > threshold
    tp, fp = int(rogue_reject.sum()), int(known_reject.sum())
    fn = int((~rogue_reject).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr, tpr, _ = roc_points(known_score, rogue_score)
    reaches = np.flatnonzero(tpr >= 0.95)
    labels = np.r_[np.zeros(len(known_score), int), np.ones(len(rogue_score), int)]
    scores = np.r_[known_score, rogue_score]
    return {
        "score": "mondrian_conformal_loo_known_operating_threshold",
        "auc": auc_rank(labels, scores),
        "eer": eer(known_score, rogue_score),
        "fpr95": float(np.min(fpr[reaches])) if len(reaches) else 1.0,
        "known_closed_acc_conformal": float(known_correct.mean()),
        "known_accept_rate": float((~known_reject).mean()),
        "unknown_recall": float(recall),
        "unknown_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "oscr": oscr(known_score, rogue_score, known_correct),
        "tau_known_only_loo": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    manifest = checkpoint["split_manifest"]
    root = Path(args.dataset_root)
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(root / "Y_global.npy")
    role = np.load(root / "sample_role.npy")
    known_classes = manifest["known_classes"]
    mapping = {int(class_id): i for i, class_id in enumerate(known_classes)}
    is_known = np.isin(y, known_classes)
    index = [
        np.flatnonzero(is_known & (role == code)) for code in (0, 1, 2)
    ]
    loaders = [
        make_loader(x, y, index[0], mapping, args.batch_size, False),
        make_loader(x, y, index[1], mapping, args.batch_size, False),
        make_loader(x, y, index[2], mapping, args.batch_size, False),
    ]
    fingerprints = {
        name: dataset_fingerprint(loader)
        for name, loader in zip(
            ("known_train", "known_validation", "known_test"), loaders
        )
    }
    if checkpoint["split_fingerprints"] != fingerprints:
        raise ValueError("known split fingerprints do not match checkpoint")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(model_args.seed)
    model = build_model(model_args, device)
    model.load_state_dict(checkpoint["model"])
    tr_e, tr_l, tr_y = extract(model, loaders[0], device)
    va_e, _, va_y = extract(model, loaders[1], device)
    kn_e, _, kn_y = extract(model, loaders[2], device)
    calibrator = MondrianMahalanobis(tr_e, tr_l, tr_y, va_e, va_y)
    validation_p_loo = leave_one_out_validation_p(calibrator, va_e, va_y)
    validation_score = 1.0 - validation_p_loo.max(axis=1)
    threshold = float(np.quantile(validation_score, 1.0 - args.known_reject_rate))
    known_p = calibrator.p_values(kn_e)

    # Materialize rogue data only after the known-only threshold is frozen.
    rogue_index = np.flatnonzero(
        np.isin(y, manifest["rogue_classes"]) & (role == 2)
    )
    rogue_y = y[rogue_index].astype(np.int64)
    rogue_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(x[rogue_index])),
            torch.from_numpy(rogue_y),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    rogue_e, _, extracted_rogue_y = extract(model, rogue_loader, device)
    rogue_p = calibrator.p_values(rogue_e)
    metrics = metric_row(known_p, rogue_p, kn_y, threshold)
    per_rogue_class = {
        str(int(class_id)): metric_row(
            known_p, rogue_p[extracted_rogue_y == class_id], kn_y, threshold
        )
        for class_id in sorted(np.unique(extracted_rogue_y))
    }
    result = {
        "checkpoint": args.checkpoint,
        "benchmark": "multiknown17_v1_low_sample",
        "split": manifest["split_index"],
        "seed": int(model_args.seed),
        "known_classes": known_classes,
        "rogue_classes": manifest["rogue_classes"],
        "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256,
        "protocol": {
            "score_fit": "known training only",
            "conformal_reference": "known validation by true class",
            "operating_threshold": "90th percentile leave-one-out known-validation nonconformity",
            "target_rogue_materialized": "after threshold frozen",
            "target_rogue_used_for_fit_selection_or_threshold": False,
        },
        "validation_loo_score_quantiles": {
            str(q): float(np.quantile(validation_score, q))
            for q in (0.5, 0.8, 0.9, 0.95, 1.0)
        },
        "metrics": metrics,
        "per_rogue_class": per_rogue_class,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
