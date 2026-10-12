"""Diagnostic separability ceiling for one near-known rogue/known pair.

This deliberately uses rogue labels and therefore is never a deployable method
or hyperparameter-selection result.  Both classes come from the network-held-out
test partitions. Repeated splits ask whether the frozen embedding or simple
spectral features contain information that the known-only open-set score misses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import train_emitter_based as legacy
from score_benchmark import full_rogue_loader, resolve_data_seed
from train_causal_ablation import build_model, extract, seed_everything


def spectral_features(x: np.ndarray, bins: int = 128) -> np.ndarray:
    if x.shape[-1] != 2:
        x = np.transpose(x, (0, 2, 1))
    z = x[..., 0].astype(np.float64) + 1j * x[..., 1].astype(np.float64)
    z = z - z.mean(axis=1, keepdims=True)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True)).clip(1e-8)
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(z, axis=1), axes=1)))
    pooled = np.stack(
        [part.mean(axis=1) for part in np.array_split(magnitude, bins, axis=1)], axis=1
    )
    amplitude = np.abs(z)
    summary = np.stack(
        [amplitude.mean(1), amplitude.std(1),
         np.quantile(amplitude, 0.25, axis=1),
         np.quantile(amplitude, 0.75, axis=1)], axis=1
    )
    return np.concatenate([pooled, summary], axis=1)


def repeated_oracle(features: np.ndarray, labels: np.ndarray, repeats: int, seed: int):
    rows = {"linear": [], "rbf": []}
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        rng.shuffle(class0); rng.shuffle(class1)
        n0, n1 = int(0.7 * len(class0)), int(0.7 * len(class1))
        train = np.r_[class0[:n0], class1[:n1]]
        test = np.r_[class0[n0:], class1[n1:]]
        models = {
            "linear": make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)
            ),
            "rbf": make_pipeline(
                StandardScaler(), SVC(C=1.0, gamma="scale", kernel="rbf")
            ),
        }
        for name, model in models.items():
            model.fit(features[train], labels[train])
            rows[name].append(float(roc_auc_score(labels[test], model.decision_function(features[test]))))
    return {
        name: {
            "mean_auc": float(np.mean(values)),
            "std_auc": float(np.std(values, ddof=1)),
            "min_auc": float(np.min(values)),
            "max_auc": float(np.max(values)),
            "repeats": repeats,
        }
        for name, values in rows.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--known-index", type=int, default=1)
    parser.add_argument("--known-global", type=int, default=20)
    parser.add_argument("--rogue-global", type=int, default=6)
    parser.add_argument("--eval-R", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7717)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = argparse.Namespace(**checkpoint["args"])
    data_seed = resolve_data_seed(model_args)
    seed_everything(data_seed)
    dataset = legacy.EmitterBasedDataset(R=model_args.R)
    loaders = legacy.create_dataloaders(
        dataset, batch_size=128, val_split=0.2, rogue_val_split=0.5,
        seed=data_seed
    )
    rogue_loader = full_rogue_loader(args.eval_R, 128)
    seed_everything(model_args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(model_args, device)
    model.load_state_dict(checkpoint["model"])
    known_embedding, _, known_label = extract(model, loaders[2], device)
    rogue_embedding, _, rogue_label = extract(model, rogue_loader, device)

    known_mask = known_label == args.known_index
    rogue_mask = rogue_label == args.rogue_global
    embedding = np.r_[known_embedding[known_mask], rogue_embedding[rogue_mask]]
    labels = np.r_[np.zeros(known_mask.sum(), int), np.ones(rogue_mask.sum(), int)]

    known_x = loaders[2].dataset.tensors[0].numpy()[known_mask]
    rogue_x = rogue_loader.dataset.tensors[0].numpy()[rogue_mask]
    spectrum = spectral_features(np.r_[known_x, rogue_x])
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pair": {"known_index": args.known_index, "known_global": args.known_global,
                 "rogue_global": args.rogue_global},
        "samples_per_class": [int(known_mask.sum()), int(rogue_mask.sum())],
        "embedding_oracle": repeated_oracle(embedding, labels, args.repeats, args.seed),
        "spectral_oracle": repeated_oracle(spectrum, labels, args.repeats, args.seed),
        "protocol": {
            "diagnostic_only": True,
            "rogue_labels_used": True,
            "network_input_samples": "held-out known test and target rogue test",
            "not_eligible_for_deployment_or_model_selection": True,
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
