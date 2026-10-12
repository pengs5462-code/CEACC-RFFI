"""Conformal OSR on trained pooled features instead of an untrained head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

import train_emitter_based as legacy
from conformal_benchmark import MondrianMahalanobis, result_row
from score_benchmark import full_rogue_loader, resolve_data_seed
from train_causal_ablation import build_model, move_batch, seed_everything
from train_representation import dataset_fingerprint


@torch.no_grad()
def extract_trained_feature(model, loader, device):
    feature_slot = {}

    def capture(_module, inputs):
        feature_slot["value"] = inputs[0]

    handle = model.contrast_head.register_forward_pre_hook(capture)
    features, logits, labels = [], [], []
    try:
        model.eval()
        for data, target in loader:
            data, target = move_batch(data, target, device)
            _, output = model(data)
            feature = feature_slot.pop("value")
            if feature.ndim != 3 or feature.shape[1] != 512:
                raise RuntimeError(f"unexpected trained feature shape {tuple(feature.shape)}")
            feature = F.adaptive_avg_pool1d(feature, 1).flatten(1)
            features.append(feature.cpu().numpy())
            logits.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(features), np.concatenate(logits), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-R", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.1)
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
    train_f, train_l, train_y = extract_trained_feature(model, loaders[0], device)
    val_f, _, val_y = extract_trained_feature(model, loaders[1], device)
    known_f, known_l, known_y = extract_trained_feature(model, loaders[2], device)

    pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=0)
    train_e = pca.fit_transform(train_f)
    val_e = pca.transform(val_f)
    known_e = pca.transform(known_f)
    calibrator = MondrianMahalanobis(train_e, train_l, train_y, val_e, val_y)
    known_p = calibrator.p_values(known_e)

    rogue_loader = full_rogue_loader(args.eval_R, args.batch_size)
    rogue_f, _, rogue_y = extract_trained_feature(model, rogue_loader, device)
    rogue_e = pca.transform(rogue_f)
    rogue_p = calibrator.p_values(rogue_e)
    result = result_row(known_p, rogue_p, known_y, args.alpha)
    result["network_known_closed_acc"] = float((known_l.argmax(1) == known_y).mean())
    per_rogue_class = {
        str(int(class_id)): result_row(
            known_p, rogue_p[rogue_y == class_id], known_y, args.alpha
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
        "feature": {
            "source": "trained 512-D tensor feeding contrast_head",
            "legacy_untrained_contrast_head_used": False,
            "pca_dim": args.pca_dim,
            "pca_fit": "known training only",
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "stored_projection_parameters": int(pca.components_.size + pca.mean_.size),
        },
        "protocol": {
            "encoder_training_and_selection": "known only",
            "PCA_and_covariance_fit": "known training only",
            "conformal_calibration": "classwise known validation only",
            "alpha": args.alpha,
            "target_rogue_materialized": "after PCA, covariance, and calibration were frozen",
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "score_configuration": "PCA dimension 64 pre-registered to match legacy embedding width",
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
