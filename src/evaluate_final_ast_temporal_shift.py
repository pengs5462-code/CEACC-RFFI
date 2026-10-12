#!/usr/bin/env python3
"""Final-paper evaluation for the fixed-AST complementary-only architecture.

This script never trains a model and never uses outer rogue data for fitting,
calibration, lambda selection, beta selection, or threshold selection.  It
reuses the nine frozen Stage-A and complementary-only Stage-B checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from calibrated_rejection_fusion_multiknown import loo_cdf, reference_cdf
from continuous_gcr_loeo import quantile
from evaluate_new_arch_gcr_matched import common_features, metric_row, oscr, write_csv
from fisher_retaining_cascade import build_retaining_scores


ROOT = Path(os.environ.get("PLSAR_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "results/final_ast_temporal_shift_v1"
SPLITS = (1, 2, 3)
SEEDS = (42, 2023, 3407)
EPS = 1e-12
CANDIDATE = "nophase_alpha0"
SHIFTS = (-200, -100, 0, 100, 200)


def read_selection():
    source = ROOT / "results/architecture_simplification_v1/experiment2_complementary_only"
    selected = {}
    for row in csv.DictReader((source / "selected_lambda_beta.csv").open()):
        selected[(int(row["split"]), int(row["seed"]))] = (float(row["lambda"]), float(row["beta"]))

    # Entropy-only uses lambda=0 and independently selects beta from inner LOEO.
    inner = []
    for path in sorted(source.glob("inner_worker*.csv")):
        inner.extend(csv.DictReader(path.open()))
    entropy_beta = {}
    for split in SPLITS:
        for seed in SEEDS:
            candidates = []
            for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
                rows = [r for r in inner if int(r["split"]) == split and int(r["seed"]) == seed
                        and float(r["lambda"]) == 0.0 and float(r["beta"]) == beta]
                candidates.append((np.mean([float(r["retaining_oscr"]) for r in rows]),
                                   min(float(r["retaining_oscr"]) for r in rows), -beta, beta))
            entropy_beta[(split, seed)] = max(candidates)[3]
    return selected, entropy_beta


def aggregate(cdfs, names, rule, calibration_size=180):
    x = np.column_stack([cdfs[name] for name in names])
    if rule == "fisher":
        # The finite-sample floor is set by the known-validation reference,
        # never by the number of test samples being scored.
        floor = 1 / (calibration_size + 1)
        return -2 * np.log(np.clip(1 - x + floor, EPS, 1)).sum(1)
    if rule == "mean":
        return x.mean(1)
    if rule == "maximum":
        return x.max(1)
    if rule == "stouffer":
        return norm.ppf(np.clip(x, 1e-6, 1 - 1e-6)).sum(1) / np.sqrt(x.shape[1])
    if rule == "cauchy":
        return np.tan(np.pi * (np.clip(x, 1e-6, 1 - 1e-6) - 0.5)).mean(1)
    raise ValueError(rule)


def summarize(rows, group_key, metrics):
    out = []
    groups = sorted({r[group_key] for r in rows})
    for group in groups:
        rr = [r for r in rows if r[group_key] == group]
        item = {group_key: group, "n_runs": len(rr)}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in rr])
            item[f"{metric}_mean"] = values.mean()
            item[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        out.append(item)
    return out


def run_worker(worker, workers, device_name):
    OUT.mkdir(parents=True, exist_ok=True)
    selected, _ = read_selection()
    data = ROOT / "dataset/multiknown17_v1"
    y = np.load(data / "Y_global.npy")
    role = np.load(data / "sample_role.npy")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    rows = []
    tasks = [(shift, split, seed) for shift in SHIFTS for split in SPLITS for seed in SEEDS]
    for task_index, (shift, split, seed) in enumerate(tasks):
        if task_index % workers != worker:
            continue
        manifest = json.loads((data / f"split_{split}.json").read_text())
        known = list(map(int, manifest["known_classes"]))
        rogue = list(map(int, manifest["rogue_classes"]))
        train_idx = np.flatnonzero(np.isin(y, known) & (role == 0))
        val_idx = np.flatnonzero(np.isin(y, known) & (role == 1))
        known_idx = np.flatnonzero(np.isin(y, known) & (role == 2))
        rogue_idx = np.flatnonzero(np.isin(y, rogue) & (role == 2))
        test_idx = np.r_[known_idx, rogue_idx]
        lam, beta = selected[(split, seed)]
        head = (ROOT / "results/architecture_simplification_v1/experiment2_complementary_only/outer_selected"
                / f"split{split}_seed{seed}_{CANDIDATE}/stage_b/best.pt")
        f = common_features(
            ROOT, split, seed, known, train_idx, val_idx, test_idx,
            ROOT / f"results/multiknown17_auxce_backbone_split{split}_seed{seed}/best.pt",
            head, CANDIDATE, 128, device, test_circular_shift=shift,
        )
        nk = len(known_idx)
        is_known = np.r_[np.ones(nk, bool), np.zeros(len(rogue_idx), bool)]
        mapped = np.asarray([known.index(int(v)) for v in y[known_idx]])
        correct = np.r_[f["h_unknown"]["logits"][:nk].argmax(1) == mapped,
                        np.zeros(len(rogue_idx), bool)]
        fisher_reject = f["uf"] > f["tau_f"]
        rows.append({"shift": shift, "split": split, "seed": seed,
                     "method": "Primary Fisher", "lambda": lam, "beta": beta,
                     **metric_row(is_known, correct, f["uf"], fisher_reject)})
        review_val = f["val_sh"] + lam * f["val_sc"]
        review_test = f["unk_sh"] + lam * f["unk_sc"]
        tau_r = quantile(review_val, .95, "higher")
        reject = (~f["ua"]) | (f["ua"] & (review_test > tau_r))
        _, score, _, _ = build_retaining_scores(f, lam, beta)
        rows.append({"shift": shift, "split": split, "seed": seed,
                     "method": "Fisher-retaining final", "lambda": lam, "beta": beta,
                     **metric_row(is_known, correct, score, reject)})
        print(json.dumps({"worker": worker, "completed": [shift, split, seed],
                          "lambda": lam, "beta": beta}), flush=True)
    write_csv(OUT / f"worker{worker}.csv", rows)


def summarize_workers():
    rows = []
    for path in sorted(OUT.glob("worker*.csv")):
        rows.extend(csv.DictReader(path.open()))
    expected = len(SHIFTS) * len(SPLITS) * len(SEEDS) * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    rows.sort(key=lambda r: (int(r["shift"]), r["method"], int(r["split"]), int(r["seed"])))
    write_csv(OUT / "temporal_shift_results.csv", rows)
    metrics = ("auc", "eer", "known_acc", "known_accept", "known_correct_accept",
               "unknown_recall", "unknown_precision", "unknown_f1", "oscr",
               "overall_open_accuracy")
    summary = []
    for shift in SHIFTS:
        for method in ("Primary Fisher", "Fisher-retaining final"):
            rr = [r for r in rows if int(r["shift"]) == shift and r["method"] == method]
            item = {"shift": shift, "method": method, "n_runs": len(rr)}
            for metric in metrics:
                values = np.asarray([float(r[metric]) for r in rr])
                item[f"{metric}_mean"] = values.mean()
                item[f"{metric}_std"] = values.std(ddof=1)
            summary.append(item)
    write_csv(OUT / "temporal_shift_summary.csv", summary)
    (OUT / "PROTOCOL.md").write_text(
        "# Final-architecture temporal-shift evaluation\n\n"
        "All nine Stage-A and Stage-B checkpoints are frozen. Only outer known-test and rogue-test I/Q arrays "
        "are circularly shifted. Training and known-validation data are unchanged. Fisher Q90 and review Q95 "
        "calibration, lambda, and beta remain frozen from known-only validation and strict inner LOEO. "
        "No test result selects a shift, threshold, lambda, beta, checkpoint, or architecture.\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "summarize"))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "run":
        run_worker(args.worker, args.workers, args.device)
    else:
        summarize_workers()


if __name__ == "__main__":
    main()
