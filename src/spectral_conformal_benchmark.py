"""Known-only spectral and embedding+spectrum conformal OSR benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_emitter_based as legacy
from conformal_benchmark import MondrianMahalanobis, result_row
from near_unknown_oracle import spectral_features
from score_benchmark import full_rogue_loader, normalize, resolve_data_seed
from train_causal_ablation import build_model, extract, seed_everything
from train_representation import dataset_fingerprint


def deterministic(loader, batch_size: int) -> DataLoader:
    return DataLoader(loader.dataset, batch_size=batch_size, shuffle=False)


def standardized(train, *others):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True).clip(1e-6)
    return ((train - mean) / scale,) + tuple((x - mean) / scale for x in others)


def equal_block_hybrid(embedding, spectrum):
    return np.concatenate([normalize(embedding), normalize(spectrum)], axis=1)


def evaluate_representation(
    name,
    train_feature,
    train_labels,
    val_feature,
    val_labels,
    known_feature,
    known_labels,
    rogue_feature,
    rogue_labels,
    alpha,
):
    zeros_train = np.zeros((len(train_feature), 1), dtype=np.float32)
    calibrator = MondrianMahalanobis(
        train_feature,
        zeros_train,
        train_labels,
        val_feature,
        val_labels,
    )
    known_p = calibrator.p_values(known_feature)
    rogue_p = calibrator.p_values(rogue_feature)
    row = result_row(known_p, rogue_p, known_labels, alpha)
    row["score"] = name
    per_class = {}
    for class_id in sorted(np.unique(rogue_labels)):
        mask = rogue_labels == class_id
        item = result_row(known_p, rogue_p[mask], known_labels, alpha)
        item["score"] = name
        per_class[str(int(class_id))] = item
    return row, per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-R", type=int, default=8)
    parser.add_argument("--bins", type=int, default=128)
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
    actual_fingerprints = {
        name: dataset_fingerprint(loader)
        for name, loader in zip(
            ("known_train", "known_validation", "known_test"), loaders[:3]
        )
    }
    if checkpoint.get("split_fingerprints") != actual_fingerprints:
        raise ValueError("materialized known split fingerprints do not match checkpoint")

    rogue_loader = full_rogue_loader(args.eval_R, args.batch_size)
    aligned = [deterministic(loader, args.batch_size) for loader in loaders[:3]]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(model_args.seed)
    model = build_model(model_args, device)
    model.load_state_dict(checkpoint["model"])
    train_embedding, _, train_labels = extract(model, aligned[0], device)
    val_embedding, _, val_labels = extract(model, aligned[1], device)
    known_embedding, _, known_labels = extract(model, aligned[2], device)
    rogue_embedding, _, rogue_labels = extract(model, rogue_loader, device)

    raw = [loader.dataset.tensors[0].numpy() for loader in aligned]
    rogue_raw = rogue_loader.dataset.tensors[0].numpy()
    train_spectrum, val_spectrum, known_spectrum, rogue_spectrum = standardized(
        spectral_features(raw[0], bins=args.bins),
        spectral_features(raw[1], bins=args.bins),
        spectral_features(raw[2], bins=args.bins),
        spectral_features(rogue_raw, bins=args.bins),
    )
    representations = {
        "spectral_mondrian_conformal": (
            train_spectrum,
            val_spectrum,
            known_spectrum,
            rogue_spectrum,
        ),
        "equal_block_embedding_spectral_mondrian_conformal": (
            equal_block_hybrid(train_embedding, train_spectrum),
            equal_block_hybrid(val_embedding, val_spectrum),
            equal_block_hybrid(known_embedding, known_spectrum),
            equal_block_hybrid(rogue_embedding, rogue_spectrum),
        ),
    }
    results = []
    per_rogue_class = {}
    for name, features in representations.items():
        row, class_rows = evaluate_representation(
            name,
            features[0],
            train_labels,
            features[1],
            val_labels,
            features[2],
            known_labels,
            features[3],
            rogue_labels,
            args.alpha,
        )
        results.append(row)
        per_rogue_class[name] = class_rows
    result = {
        "checkpoint": args.checkpoint,
        "seed": int(model_args.seed),
        "data_seed": data_seed,
        "evaluation_R": args.eval_R,
        "spectral_bins": args.bins,
        "split_fingerprints": actual_fingerprints,
        "runner_sha256": runner_sha256,
        "protocol": {
            "fit": "known training only",
            "calibration": "classwise known validation only",
            "alpha": args.alpha,
            "fusion": "fixed equal-L2-norm blocks; no learned or tuned fusion weight",
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "target_rogue_role": "final evaluation only",
            "development_disclosure": (
                "spectral branch motivated by diagnostic on current split; final claims "
                "require untouched known/rogue class splits"
            ),
        },
        "results": results,
        "per_rogue_class": per_rogue_class,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
