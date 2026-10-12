#!/usr/bin/env python3
"""Leakage-free Fisher-retaining continuous cascade audit.

The frozen Fisher/GCR decision is unchanged.  Only the threshold-swept
continuous score is reconstructed.  Beta is selected with the existing strict
inner LOEO folds at the already-selected lambda for each outer split/seed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from calibrated_rejection_fusion_multiknown import loo_cdf, reference_cdf
from continuous_gcr_loeo import quantile
from evaluate_new_arch_gcr_matched import common_features, metric_row, oscr, selected_map, write_csv


SPLITS = (1, 2, 3)
SEEDS = (42, 2023, 3407)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def lambda_map(out: Path):
    selected = {}
    for row in csv.DictReader((out / "lambda_selection_summary.csv").open()):
        if row["selected"].lower() == "true":
            selected[(int(row["split"]), int(row["seed"]))] = float(row["lambda"])
    if len(selected) != 9:
        raise RuntimeError(f"expected 9 lambdas, found {len(selected)}")
    return selected


def build_retaining_scores(features, lam, beta):
    """Return validation/test scores; larger means more unknown-like."""
    va, ua = features["va"], features["ua"]
    vf, uf = features["vf"], features["uf"]
    val_review = features["val_sh"] + lam * features["val_sc"]
    unk_review = features["unk_sh"] + lam * features["unk_sc"]

    # Both accepted-branch sources are calibrated against exactly the same
    # Fisher-accepted known-validation population.
    val_f_rank = loo_cdf(vf[va])
    unk_f_rank = reference_cdf(vf[va], uf[ua])
    val_r_rank = loo_cdf(val_review)
    unk_r_rank = reference_cdf(val_review, unk_review[ua])
    val_mix = (1 - beta) * val_f_rank + beta * val_r_rank
    unk_mix = (1 - beta) * unk_f_rank + beta * unk_r_rank

    # Hierarchy is unchanged: every Fisher-rejected sample ranks above every
    # accepted sample; rejected-region Fisher ordering remains monotone.
    val_score = 1.0 + loo_cdf(vf)
    unk_score = 1.0 + reference_cdf(vf, uf)
    val_score[va] = val_mix
    unk_score[ua] = unk_mix
    return val_score, unk_score, val_mix, unk_mix


def inner_tasks(root: Path):
    data = root / "dataset/multiknown17_v1"
    y = np.load(data / "Y_global.npy")
    tasks = []
    for split in SPLITS:
        known = list(map(int, json.loads((data / f"split_{split}.json").read_text())["known_classes"]))
        for seed in SEEDS:
            for held in known:
                tasks.append((split, seed, held, known))
    return tasks, y


def run_inner(root, worker, workers, batch_size, device):
    data = root / "dataset/multiknown17_v1"
    role = np.load(data / "sample_role.npy")
    tasks, y = inner_tasks(root)
    chosen, lambdas = selected_map(root), lambda_map(root / "results/new_arch_gcr_matched_v1")
    rows = []
    for index, (split, seed, held, known) in enumerate(tasks):
        if index % workers != worker:
            continue
        candidate, lam = chosen[(split, seed)], lambdas[(split, seed)]
        active = [value for value in known if value != held]
        train_idx = np.flatnonzero(np.isin(y, active) & (role == 0))
        val_idx = np.flatnonzero(np.isin(y, active) & (role == 1))
        pseudo_idx = np.flatnonzero(y == held)
        tag = f"split{split}_seed{seed}_heldout{held}"
        features = common_features(
            root, split, seed, active, train_idx, val_idx, pseudo_idx,
            root / "results/continuous_gcr_loeo_v2/inner" / tag / "stage_a/best.pt",
            root / "results/inner_physical_arch_selection_v1/inner" / tag / candidate / "stage_b/best.pt",
            candidate, batch_size, device,
        )
        is_known = np.r_[np.ones(len(val_idx), bool), np.zeros(len(pseudo_idx), bool)]
        correct = np.r_[features["h_val"]["logits"].argmax(1) == features["val_y"], np.zeros(len(pseudo_idx), bool)]
        for beta in BETAS:
            val_score, unk_score, _, _ = build_retaining_scores(features, lam, beta)
            score = np.r_[val_score, unk_score]
            rows.append({
                "split": split, "seed": seed, "held_out_emitter": held,
                "candidate": candidate, "frozen_lambda": lam, "beta": beta,
                "inner_auc": float(roc_auc_score((~is_known).astype(int), score)),
                "inner_oscr": oscr(is_known, correct, score),
                "outer_rogue_accessed": False,
            })
        print(json.dumps({"worker": worker, "completed": [split, seed, held],
                          "candidate": candidate, "lambda": lam}), flush=True)
    out = root / "results/fisher_retaining_cascade_v1"
    write_csv(out / f"inner_worker{worker}.csv", rows)


def summarize_inner(root):
    out = root / "results/fisher_retaining_cascade_v1"
    rows = []
    for path in sorted(out.glob("inner_worker*.csv")):
        rows.extend(csv.DictReader(path.open()))
    rows = list(rows)
    rows.sort(key=lambda r: (int(r["split"]), int(r["seed"]), int(r["held_out_emitter"]), float(r["beta"])))
    write_csv(out / "inner_loeo_fisher_retaining_all_folds.csv", rows)
    summary = []
    for split in SPLITS:
        for seed in SEEDS:
            candidates = []
            for beta in BETAS:
                rr = [r for r in rows if int(r["split"]) == split and int(r["seed"]) == seed and float(r["beta"]) == beta]
                if len(rr) != 9:
                    raise RuntimeError((split, seed, beta, len(rr)))
                candidates.append({
                    "split": split, "seed": seed, "candidate": rr[0]["candidate"],
                    "frozen_lambda": float(rr[0]["frozen_lambda"]), "beta": beta,
                    "mean_inner_auc": float(np.mean([float(r["inner_auc"]) for r in rr])),
                    "mean_inner_oscr": float(np.mean([float(r["inner_oscr"]) for r in rr])),
                    "worst_fold_oscr": float(np.min([float(r["inner_oscr"]) for r in rr])),
                    "selected": False,
                })
            # Select by mean OSCR, then worst fold, then prefer more Fisher retention.
            best = max(candidates, key=lambda r: (r["mean_inner_oscr"], r["worst_fold_oscr"], -r["beta"]))
            best["selected"] = True
            summary.extend(candidates)
    write_csv(out / "beta_selection_summary.csv", summary)


def bootstrap_ci(values, seed=20261006, samples=200000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return float(values.mean()), float(lo), float(hi)


def run_outer(root, batch_size, device):
    source = root / "results/new_arch_gcr_matched_v1"
    out = root / "results/fisher_retaining_cascade_v1"
    chosen, lambdas = selected_map(root), lambda_map(source)
    betas = {}
    for row in csv.DictReader((out / "beta_selection_summary.csv").open()):
        if row["selected"].lower() == "true":
            betas[(int(row["split"]), int(row["seed"]))] = float(row["beta"])
    if len(betas) != 9:
        raise RuntimeError(f"expected 9 betas, found {len(betas)}")
    data = root / "dataset/multiknown17_v1"
    y = np.load(data / "Y_global.npy")
    role = np.load(data / "sample_role.npy")
    rows = []
    for split in SPLITS:
        manifest = json.loads((data / f"split_{split}.json").read_text())
        known = list(map(int, manifest["known_classes"]))
        rogue = list(map(int, manifest["rogue_classes"]))
        for seed in SEEDS:
            candidate, lam, beta = chosen[(split, seed)], lambdas[(split, seed)], betas[(split, seed)]
            train_idx = np.flatnonzero(np.isin(y, known) & (role == 0))
            val_idx = np.flatnonzero(np.isin(y, known) & (role == 1))
            known_idx = np.flatnonzero(np.isin(y, known) & (role == 2))
            rogue_idx = np.flatnonzero(np.isin(y, rogue) & (role == 2))
            test_idx = np.r_[known_idx, rogue_idx]
            features = common_features(
                root, split, seed, known, train_idx, val_idx, test_idx,
                root / f"results/multiknown17_auxce_backbone_split{split}_seed{seed}/best.pt",
                root / "results/inner_physical_arch_selection_v1/outer_selected"
                / f"split{split}_seed{seed}_{candidate}" / "stage_b/best.pt",
                candidate, batch_size, device,
            )
            nk = len(known_idx)
            is_known = np.r_[np.ones(nk, bool), np.zeros(len(rogue_idx), bool)]
            mapped_y = np.asarray([known.index(int(v)) for v in y[known_idx]])
            correct_known = features["h_unknown"]["logits"][:nk].argmax(1) == mapped_y
            correct = np.r_[correct_known, np.zeros(len(rogue_idx), bool)]

            fisher_score = features["uf"]
            fisher_reject = fisher_score > features["tau_f"]
            review = features["unk_sh"] + lam * features["unk_sc"]
            tau_r = quantile(features["val_sh"] + lam * features["val_sc"], .95, "higher")
            gcr_reject = (~features["ua"]) | (features["ua"] & (review > tau_r))
            current = np.where(features["ua"], review, 1 + lam + reference_cdf(features["vf"], features["uf"]))
            _, retaining, _, retaining_accepted = build_retaining_scores(features, lam, beta)

            methods = (
                ("primary_fisher", fisher_score, fisher_reject),
                ("current_review_replacement", current, gcr_reject),
                ("fisher_retaining_review", retaining, gcr_reject),
            )
            labels = (~is_known).astype(int)
            for method, score, reject in methods:
                accepted = features["ua"]
                rows.append({
                    "split": split, "seed": seed, "candidate": candidate,
                    "frozen_lambda": lam, "selected_beta": beta, "method": method,
                    "accepted_region_auc": float(roc_auc_score(labels[accepted], score[accepted])),
                    **metric_row(is_known, correct, score, reject),
                })
            print(json.dumps({"outer_completed": [split, seed], "candidate": candidate,
                              "lambda": lam, "beta": beta}), flush=True)
    write_csv(out / "outer_fisher_retaining_results.csv", rows)

    metrics = ("accepted_region_auc", "auc", "eer", "known_acc", "known_accept",
               "known_correct_accept", "unknown_recall", "unknown_precision",
               "unknown_f1", "oscr", "overall_open_accuracy")
    summary = []
    for method in ("primary_fisher", "current_review_replacement", "fisher_retaining_review"):
        rr = [r for r in rows if r["method"] == method]
        item = {"method": method, "n_runs": len(rr)}
        for metric in metrics:
            values = np.asarray([r[metric] for r in rr], float)
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_sample_std"] = float(values.std(ddof=1))
        summary.append(item)
    write_csv(out / "fisher_retaining_summary.csv", summary)

    paired = []
    lookup = {(r["split"], r["seed"], r["method"]): r for r in rows}
    for comparator in ("primary_fisher", "current_review_replacement"):
        for metric in ("accepted_region_auc", "auc", "eer", "oscr"):
            diffs = []
            for split in SPLITS:
                for seed in SEEDS:
                    a = lookup[(split, seed, "fisher_retaining_review")][metric]
                    b = lookup[(split, seed, comparator)][metric]
                    diffs.append(float(a) - float(b))
            mean, lo, hi = bootstrap_ci(diffs)
            paired.append({"comparison": f"fisher_retaining-minus-{comparator}", "metric": metric,
                           "mean_difference": mean, "bootstrap_ci95_low": lo,
                           "bootstrap_ci95_high": hi, "n_runs": 9})
    write_csv(out / "fisher_retaining_paired_ci.csv", paired)
    (out / "PROTOCOL.md").write_text(
        "# Fisher-retaining cascade protocol\n\n"
        "No checkpoint was retrained. Frozen lambda and Q95 GCR decisions are unchanged. "
        "Beta was selected from {0, 0.25, 0.5, 0.75, 1} by mean inner-LOEO OSCR, then "
        "worst-fold OSCR, then smaller beta. Both Fisher and review accepted-branch scores "
        "were empirical-CDF calibrated only on Fisher-accepted known validation. No outer "
        "known or rogue sample was used for beta selection. Fixed operating-point metrics "
        "for the current and Fisher-retaining GCR must therefore be identical.\n",
        encoding="utf-8",
    )
    (out / "COMPLETE").write_text("complete\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inner", "summarize", "outer"))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.mode == "inner":
        run_inner(root, args.worker, args.workers, args.batch_size, device)
    elif args.mode == "summarize":
        summarize_inner(root)
    else:
        run_outer(root, args.batch_size, device)


if __name__ == "__main__":
    main()
