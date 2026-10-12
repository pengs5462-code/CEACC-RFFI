"""Known-only calibrated rejection benchmark for the learned physics embedding.

The candidate scores and fusion rules are constructed from known train and
known validation data before development rogue samples are materialized. This
runner compares learned z_p rejection with the stable deep and raw-RF spaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from calibrated_rejection_fusion_multiknown import loo_cdf, reference_cdf
from conformal_benchmark import MondrianMahalanobis
from recalibrate_multiknown17 import leave_one_out_validation_p
from rf_physics_feature_probe import extract_all
from score_benchmark import metric_row
from spectral_conformal_benchmark import deterministic
from train_causal_ablation import build_model, extract, seed_everything
from train_dual_head_multiknown import predict
from train_multiknown17_baseline import make_loader
from train_physics_multiview_contrastive import PhysicsMultiViewHead
from train_representation import dataset_fingerprint
from trained_feature_score_benchmark import ClassLocalKNN


EPS = 1e-12


@torch.no_grad()
def extract_physical(head, physics, batch_size, device):
    head.eval()
    output = []
    for start in range(0, len(physics), batch_size):
        value = torch.from_numpy(physics[start : start + batch_size]).to(device)
        latent, _, _ = head.encode_physics(value)
        output.append(latent.cpu().numpy())
    return np.concatenate(output)


def calibrated_views(validation_scores, target_scores):
    validation = {name: loo_cdf(value) for name, value in validation_scores.items()}
    target = {
        name: reference_cdf(validation_scores[name], value)
        for name, value in target_scores.items()
    }
    return validation, target


def fusion_scores(cdf_scores, calibration_size):
    # The final physical evidence is the union of two complementary latent
    # detectors: class-conditional Mahalanobis and MSP.  Form the OR in the
    # calibrated unknownness domain before combining it with deep/raw-RF
    # evidence.  This is the same construction used by the frozen ADS-B
    # three-space pipeline.
    cdf_scores = dict(cdf_scores)
    cdf_scores["latent_or_mondrian_msp"] = np.maximum(
        cdf_scores["latent_mondrian"], cdf_scores["latent_msp"]
    )
    floor = 1.0 / (calibration_size + 1.0)

    def fisher(names):
        tails = [np.clip(1.0 - cdf_scores[name] + floor, EPS, 1.0) for name in names]
        return -2.0 * sum(np.log(value) for value in tails)

    def mean_rank(names):
        return np.mean([cdf_scores[name] for name in names], axis=0)

    result = {}
    for latent in (
        "latent_mondrian",
        "latent_local_knn",
        "latent_msp",
        "latent_or_mondrian_msp",
    ):
        result[f"fisher_deep_plus_{latent}"] = fisher(("deep", latent))
        result[f"fisher_raw_plus_{latent}"] = fisher(("raw_rf", latent))
        result[f"fisher_three_space_{latent}"] = fisher(("deep", "raw_rf", latent))
        result[f"mean_rank_three_space_{latent}"] = mean_rank(("deep", "raw_rf", latent))
    result["fisher_deep_raw_baseline"] = fisher(("deep", "raw_rf"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--physics-pca-dim", type=int, default=32)
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--test-circular-shift", type=int, default=0)
    parser.add_argument(
        "--raw-score-output",
        default=None,
        help="optional NPZ export of one frozen score for exact subset evaluation",
    )
    parser.add_argument(
        "--raw-score-name",
        default="fisher_three_space_latent_or_mondrian_msp",
        help="score exported by --raw-score-output; no selection is performed",
    )
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    head_checkpoint = torch.load(args.head_checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**base_checkpoint["args"])
    root = Path(model_args.dataset_root)
    manifest = json.loads((root / f"split_{model_args.split}.json").read_text())
    allowed_status = {"development", "FROZEN_BEFORE_OPEN_SET_TEST"}
    if manifest["status"] not in allowed_status or model_args.split not in (1, 2, 3):
        raise ValueError("latent rejection requires a pre-registered split")
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
        raise RuntimeError("base checkpoint/data fingerprint mismatch")
    head_fingerprints = head_checkpoint.get("split_fingerprints", {})
    if any(head_fingerprints.get(name) != value for name, value in fingerprints.items() if name in head_fingerprints):
        raise RuntimeError("head checkpoint/data fingerprint mismatch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(model_args.seed)
    backbone = build_model(model_args, device)
    backbone.load_state_dict(base_checkpoint["model"])
    backbone.eval()
    train_e, _, train_y = extract(backbone, deterministic(train_loader, args.batch_size), device)
    val_e, _, val_y = extract(backbone, val_loader, device)
    if args.test_circular_shift:
        known_iq = np.roll(
            np.asarray(x[indices["known_test"]]), args.test_circular_shift, axis=1
        )
        known_target = np.asarray([mapping[int(label)] for label in y[indices["known_test"]]])
        shifted_known_loader = DataLoader(
            TensorDataset(torch.from_numpy(known_iq), torch.from_numpy(known_target).long()),
            batch_size=args.batch_size,
            shuffle=False,
        )
        known_e, _, known_y = extract(backbone, shifted_known_loader, device)
    else:
        known_iq = np.asarray(x[indices["known_test"]])
        known_e, _, known_y = extract(backbone, known_loader, device)

    descriptor_keep = head_checkpoint.get("descriptor_keep_indices")

    def descriptor_extract(values):
        output = extract_all(values, 64)
        if descriptor_keep is not None:
            output = output[:, np.asarray(descriptor_keep, dtype=np.int64)]
        return output

    physics_train_raw = descriptor_extract(x[indices["train"]])
    physics_val_raw = descriptor_extract(x[indices["validation"]])
    physics_known_raw = descriptor_extract(known_iq)
    scaler = StandardScaler().fit(physics_train_raw)
    pca = PCA(n_components=args.physics_pca_dim, whiten=True, random_state=0).fit(
        scaler.transform(physics_train_raw)
    )
    raw_transform = lambda value: pca.transform(scaler.transform(value))
    raw_train, raw_val, raw_known = map(
        raw_transform, (physics_train_raw, physics_val_raw, physics_known_raw)
    )
    raw_svm = SVC(C=args.svm_c, kernel="rbf", gamma="scale", decision_function_shape="ovr")
    raw_svm.fit(raw_train, train_y)

    standardize = lambda value: (
        (value - head_checkpoint["physics_mean"]) / head_checkpoint["physics_std"]
    ).astype(np.float32)
    if head_checkpoint.get("head_arch") == "inner_physical_arch_selection_v1":
        from train_inner_physical_candidate import (
            InnerPhysicalCandidateHead, configure as configure_candidate,
        )

        configure_candidate(head_checkpoint["candidate"])
        head = InnerPhysicalCandidateHead(train_e.shape[1], len(known_classes)).to(device)
    elif head_checkpoint.get("head_arch") in ("flat", "grouped", "random4view"):
        from train_physics_structure_baseline import head_from_checkpoint

        head = head_from_checkpoint(train_e.shape[1], len(known_classes), head_checkpoint).to(device)
    elif any(name.startswith("zp_projector.") for name in head_checkpoint["head"]):
        from train_physics_multiview_zp_contrast import DirectZPContrastHead

        head = DirectZPContrastHead(train_e.shape[1], len(known_classes)).to(device)
    else:
        head = PhysicsMultiViewHead(train_e.shape[1], len(known_classes)).to(device)
    head.load_state_dict(head_checkpoint["head"])
    physics_train = standardize(physics_train_raw)
    physics_val = standardize(physics_val_raw)
    physics_known = standardize(physics_known_raw)
    latent_train = extract_physical(head, physics_train, args.batch_size, device)
    latent_val = extract_physical(head, physics_val, args.batch_size, device)
    latent_known = extract_physical(head, physics_known, args.batch_size, device)
    known_logits = predict(
        head, known_e.astype(np.float32), physics_known, args.batch_size, device
    )
    known_correct = known_logits.argmax(1) == known_y

    dummy_deep = np.zeros((len(train_e), len(known_classes)), dtype=np.float32)
    deep_calibrator = MondrianMahalanobis(train_e, dummy_deep, train_y, val_e, val_y)
    dummy_latent = np.zeros((len(latent_train), len(known_classes)), dtype=np.float32)
    latent_calibrator = MondrianMahalanobis(
        latent_train, dummy_latent, train_y, latent_val, val_y
    )
    local_knn = ClassLocalKNN(latent_train, train_y, args.knn_k)
    val_logits = predict(head, val_e.astype(np.float32), physics_val, args.batch_size, device)

    validation_scores = {
        "deep": 1.0 - leave_one_out_validation_p(deep_calibrator, val_e, val_y).max(1),
        "raw_rf": -raw_svm.decision_function(raw_val).max(1),
        "latent_mondrian": 1.0 - leave_one_out_validation_p(
            latent_calibrator, latent_val, val_y
        ).max(1),
        "latent_local_knn": local_knn.score(latent_val),
        "latent_msp": 1.0 - F.softmax(torch.from_numpy(val_logits), dim=1).max(1).values.numpy(),
    }
    known_scores = {
        "deep": 1.0 - deep_calibrator.p_values(known_e).max(1),
        "raw_rf": -raw_svm.decision_function(raw_known).max(1),
        "latent_mondrian": 1.0 - latent_calibrator.p_values(latent_known).max(1),
        "latent_local_knn": local_knn.score(latent_known),
        "latent_msp": 1.0 - F.softmax(torch.from_numpy(known_logits), dim=1).max(1).values.numpy(),
    }
    val_cdf, known_cdf = calibrated_views(validation_scores, known_scores)
    calibration_size = len(val_y)
    val_fusions = fusion_scores(val_cdf, calibration_size)
    known_fusions = fusion_scores(known_cdf, calibration_size)

    # Rogue data becomes accessible only after all fit objects and candidate
    # fusion rules above have been frozen.
    rogue_global = y[indices["rogue_test"]].astype(np.int64)
    rogue_iq = np.asarray(x[indices["rogue_test"]])
    if args.test_circular_shift:
        rogue_iq = np.roll(rogue_iq, args.test_circular_shift, axis=1)
    rogue_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(rogue_iq),
            torch.from_numpy(rogue_global),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    rogue_e, _, extracted_rogue = extract(backbone, rogue_loader, device)
    physics_rogue_raw = descriptor_extract(rogue_iq)
    raw_rogue = raw_transform(physics_rogue_raw)
    physics_rogue = standardize(physics_rogue_raw)
    latent_rogue = extract_physical(head, physics_rogue, args.batch_size, device)
    rogue_logits = predict(
        head, rogue_e.astype(np.float32), physics_rogue, args.batch_size, device
    )
    rogue_scores = {
        "deep": 1.0 - deep_calibrator.p_values(rogue_e).max(1),
        "raw_rf": -raw_svm.decision_function(raw_rogue).max(1),
        "latent_mondrian": 1.0 - latent_calibrator.p_values(latent_rogue).max(1),
        "latent_local_knn": local_knn.score(latent_rogue),
        "latent_msp": 1.0 - F.softmax(torch.from_numpy(rogue_logits), dim=1).max(1).values.numpy(),
    }
    _, rogue_cdf = calibrated_views(validation_scores, rogue_scores)
    rogue_fusions = fusion_scores(rogue_cdf, calibration_size)

    all_val = validation_scores | val_fusions
    all_known = known_scores | known_fusions
    all_rogue = rogue_scores | rogue_fusions
    rows = []
    for name in all_val:
        row = metric_row(
            name,
            all_val[name],
            all_known[name],
            all_rogue[name],
            known_correct,
            args.known_reject_rate,
        )
        row["known_closed_acc"] = float(known_correct.mean())
        rows.append(row)

    per_class = {}
    for class_id in sorted(np.unique(extracted_rogue)):
        mask = extracted_rogue == class_id
        per_class[str(int(class_id))] = [
            metric_row(
                name,
                all_val[name],
                all_known[name],
                all_rogue[name][mask],
                known_correct,
                args.known_reject_rate,
            )
            for name in all_val
        ]

    output = {
        "method": "physics_multiview_latent_known_only_rejection_benchmark_v1",
        "split": int(model_args.split),
        "seed": int(model_args.seed),
        "known_classes": known_classes,
        "rogue_classes": rogue_classes,
        "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256,
        "descriptor": head_checkpoint.get("descriptor", "full339"),
        "descriptor_dimension": int(physics_train_raw.shape[1]),
        "head_arch": head_checkpoint.get("head_arch", "physical4view"),
        "test_circular_shift": int(args.test_circular_shift),
        "protocol": {
            "all_fit": "known training only",
            "calibration_and_threshold": "known validation only",
            "candidate_latent_scores_fixed_before_rogue": [
                "class-conditional Mondrian Mahalanobis",
                "class-local kNN k=5",
                "maximum-softmax uncertainty",
            ],
            "fusion_rules_fixed_before_rogue": "equal-evidence Fisher and mean rank",
            "target_rogue_materialized": "after all fitting and rule construction froze",
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "test_only_circular_shift": int(args.test_circular_shift),
            "training_validation_calibration_shifted": False,
        },
        "results": rows,
        "per_rogue_class": per_class,
        "rogue_class_counts": {
            str(int(class_id)): int((extracted_rogue == class_id).sum())
            for class_id in np.unique(extracted_rogue)
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.raw_score_output:
        if args.raw_score_name not in all_val:
            raise ValueError(
                f"requested raw score {args.raw_score_name!r} is unavailable; "
                f"choices are {sorted(all_val)}"
            )
        raw_path = Path(args.raw_score_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_path,
            score_name=np.asarray(args.raw_score_name),
            split=np.asarray(int(model_args.split)),
            seed=np.asarray(int(model_args.seed)),
            known_classes=np.asarray(known_classes, dtype=np.int64),
            rogue_classes=np.asarray(rogue_classes, dtype=np.int64),
            validation_score=np.asarray(all_val[args.raw_score_name], dtype=np.float64),
            known_score=np.asarray(all_known[args.raw_score_name], dtype=np.float64),
            rogue_score=np.asarray(all_rogue[args.raw_score_name], dtype=np.float64),
            rogue_label=np.asarray(extracted_rogue, dtype=np.int64),
            known_correct=np.asarray(known_correct, dtype=np.bool_),
            known_reject_rate=np.asarray(args.known_reject_rate, dtype=np.float64),
            split_fingerprint_json=np.asarray(json.dumps(fingerprints, sort_keys=True)),
        )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
