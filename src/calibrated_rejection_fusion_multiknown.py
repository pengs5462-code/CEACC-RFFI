"""Known-validation calibrated rejection fusion with the learned dual head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from conformal_benchmark import MondrianMahalanobis
from recalibrate_multiknown17 import leave_one_out_validation_p
from rf_physics_feature_probe import extract_all
from score_benchmark import metric_row
from spectral_conformal_benchmark import deterministic
from train_causal_ablation import build_model, extract, seed_everything
from train_dual_head_multiknown import PhysicsIdentificationHead, predict
from train_multiknown17_baseline import make_loader
from train_representation import dataset_fingerprint


EPS = 1e-12


def loo_cdf(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        reference = np.sort(np.delete(values, index))
        result[index] = (
            np.searchsorted(reference, value, side="right") + 1.0
        ) / (len(reference) + 1.0)
    return result


def reference_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    return (
        np.searchsorted(ordered, values, side="right") + 1.0
    ) / (len(ordered) + 1.0)


def combine(
    cdf_a: np.ndarray, cdf_b: np.ndarray, calibration_size: int
) -> dict[str, np.ndarray]:
    floor = 1.0 / (calibration_size + 1.0)
    tail_a = np.clip(1.0 - cdf_a + floor, EPS, 1.0)
    tail_b = np.clip(1.0 - cdf_b + floor, EPS, 1.0)
    return {
        "mean_known_cdf_rank": 0.5 * (cdf_a + cdf_b),
        "max_known_cdf_rank": np.maximum(cdf_a, cdf_b),
        "fisher_known_tail_p": -2.0 * (np.log(tail_a) + np.log(tail_b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--physics-pca-dim", type=int, default=32)
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--frozen-manifest", default=None)
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    head_checkpoint = torch.load(args.head_checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**base_checkpoint["args"])
    root = Path(model_args.dataset_root)
    manifest = json.loads((root / f"split_{model_args.split}.json").read_text())
    if manifest["status"] != "development":
        if manifest["status"] != "reserved_untouched" or not args.frozen_manifest:
            raise ValueError("reserved evaluation requires an explicit frozen manifest")
        frozen = json.loads(Path(args.frozen_manifest).read_text())
        if frozen.get("method_id") != "dual_head_physics_fisher_v1" or frozen.get("status") != "frozen_before_reserved_evaluation":
            raise ValueError("invalid frozen method manifest")
    known_classes = list(map(int, manifest["known_classes"]))
    rogue_classes = list(map(int, manifest["rogue_classes"]))
    mapping = {class_id: position for position, class_id in enumerate(known_classes)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(root / "Y_global.npy")
    role = np.load(root / "sample_role.npy")
    indices = {
        "train": np.flatnonzero(np.isin(y, known_classes) & (role == 0)),
        "validation": np.flatnonzero(np.isin(y, known_classes) & (role == 1)),
        "known_test": np.flatnonzero(np.isin(y, known_classes) & (role == 2)),
        "rogue_test": np.flatnonzero(np.isin(y, rogue_classes) & (role == 2)),
    }
    seed_everything(model_args.seed + 20_000)
    train_loader = make_loader(x, y, indices["train"], mapping, model_args.batch_size, True)
    val_loader = make_loader(x, y, indices["validation"], mapping, args.batch_size, False)
    known_loader = make_loader(x, y, indices["known_test"], mapping, args.batch_size, False)
    fingerprints = {
        "known_train": dataset_fingerprint(train_loader),
        "known_validation": dataset_fingerprint(val_loader),
        "known_test": dataset_fingerprint(known_loader),
    }
    if base_checkpoint.get("split_fingerprints") != fingerprints:
        raise RuntimeError("base checkpoint split fingerprint mismatch")
    if head_checkpoint.get("split_fingerprints") != fingerprints:
        raise RuntimeError("head checkpoint split fingerprint mismatch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(model_args.seed)
    backbone = build_model(model_args, device)
    backbone.load_state_dict(base_checkpoint["model"])
    backbone.eval()
    train_e, _, train_y = extract(backbone, deterministic(train_loader, args.batch_size), device)
    val_e, _, val_y = extract(backbone, val_loader, device)
    known_e, _, known_y = extract(backbone, known_loader, device)

    physics_train_raw = extract_all(x[indices["train"]], 64)
    physics_val_raw = extract_all(x[indices["validation"]], 64)
    physics_known_raw = extract_all(x[indices["known_test"]], 64)
    scaler = StandardScaler().fit(physics_train_raw)
    pca = PCA(n_components=args.physics_pca_dim, whiten=True, random_state=0).fit(
        scaler.transform(physics_train_raw)
    )
    transform = lambda value: pca.transform(scaler.transform(value))
    p_train, p_val, p_known = map(
        transform, (physics_train_raw, physics_val_raw, physics_known_raw)
    )
    physics_svm = SVC(
        C=args.svm_c, kernel="rbf", gamma="scale", decision_function_shape="ovr"
    ).fit(p_train, train_y)
    physics_val_score = -physics_svm.decision_function(p_val).max(1)
    physics_known_score = -physics_svm.decision_function(p_known).max(1)

    dummy_logits = np.zeros((len(train_e), len(known_classes)), dtype=np.float32)
    deep = MondrianMahalanobis(train_e, dummy_logits, train_y, val_e, val_y)
    deep_val_score = 1.0 - leave_one_out_validation_p(deep, val_e, val_y).max(1)
    deep_known_score = 1.0 - deep.p_values(known_e).max(1)

    physics_mean = head_checkpoint["physics_mean"]
    physics_std = head_checkpoint["physics_std"]
    standardize = lambda value: ((value - physics_mean) / physics_std).astype(np.float32)
    head = PhysicsIdentificationHead(
        train_e.shape[1], physics_train_raw.shape[1], len(known_classes)
    ).to(device)
    head.load_state_dict(head_checkpoint["head"])
    known_logits = predict(
        head,
        known_e.astype(np.float32),
        standardize(physics_known_raw),
        args.batch_size,
        device,
    )
    known_correct = known_logits.argmax(1) == known_y

    val_deep_cdf = loo_cdf(deep_val_score)
    val_physics_cdf = loo_cdf(physics_val_score)
    known_deep_cdf = reference_cdf(deep_val_score, deep_known_score)
    known_physics_cdf = reference_cdf(physics_val_score, physics_known_score)
    calibration_size = len(deep_val_score)
    val_combined = combine(val_deep_cdf, val_physics_cdf, calibration_size)
    known_combined = combine(
        known_deep_cdf, known_physics_cdf, calibration_size
    )

    # Target rogue samples are loaded only after all fit/calibration state freezes.
    rogue_global = y[indices["rogue_test"]].astype(np.int64)
    rogue_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(x[indices["rogue_test"]])),
            torch.from_numpy(rogue_global),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    rogue_e, _, extracted_rogue = extract(backbone, rogue_loader, device)
    physics_rogue_raw = extract_all(x[indices["rogue_test"]], 64)
    p_rogue = transform(physics_rogue_raw)
    physics_rogue_score = -physics_svm.decision_function(p_rogue).max(1)
    deep_rogue_score = 1.0 - deep.p_values(rogue_e).max(1)
    rogue_combined = combine(
        reference_cdf(deep_val_score, deep_rogue_score),
        reference_cdf(physics_val_score, physics_rogue_score),
        calibration_size,
    )

    score_sets = {
        "deep_mondrian_only": (deep_val_score, deep_known_score, deep_rogue_score)
    }
    for name in val_combined:
        score_sets[name] = (
            val_combined[name], known_combined[name], rogue_combined[name]
        )
    rows = []
    for name, (val_score, known_score, rogue_score) in score_sets.items():
        row = metric_row(
            name,
            val_score,
            known_score,
            rogue_score,
            known_correct,
            args.known_reject_rate,
        )
        row["known_closed_acc"] = float(known_correct.mean())
        rows.append(row)

    output = {
        "method": "known_calibrated_rejection_fusion_with_learned_identification_head_v1",
        "split": int(model_args.split),
        "seed": int(model_args.seed),
        "known_classes": known_classes,
        "rogue_classes": rogue_classes,
        "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256,
        "protocol": {
            "class_split_status": manifest["status"],
            "reserved_splits_accessed": manifest["status"] != "development",
            "frozen_method_manifest": args.frozen_manifest,
            "all_fit": "known training only",
            "cdf_and_threshold_calibration": "known validation only with leave-one-out validation ranks",
            "fusion_rules": "fixed mean rank, max rank, and Fisher equal-evidence rules",
            "target_rogue_materialized": "after all fit and calibration state froze",
            "target_rogue_used_for_fit_selection_or_threshold": False,
        },
        "results": rows,
        "rogue_class_counts": {
            str(int(class_id)): int((extracted_rogue == class_id).sum())
            for class_id in np.unique(extracted_rogue)
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
