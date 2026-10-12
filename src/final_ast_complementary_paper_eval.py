#!/usr/bin/env python3
"""Final-paper evaluation for the fixed-AST complementary-only architecture.

This script never trains a model and never uses outer rogue data for fitting,
calibration, lambda selection, beta selection, or threshold selection.  It
reuses the nine frozen Stage-A and complementary-only Stage-B checkpoints.
"""
from __future__ import annotations

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
OUT = ROOT / "results/final_ast_complementary_paper_v1"
SPLITS = (1, 2, 3)
SEEDS = (42, 2023, 3407)
EPS = 1e-12
CANDIDATE = "nophase_alpha0"


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected, entropy_betas = read_selection()
    data = ROOT / "dataset/multiknown17_v1"
    y = np.load(data / "Y_global.npy")
    role = np.load(data / "sample_role.npy")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_rows, space_rows, rule_rows, accepted_rows, openness_rows = [], [], [], [], []

    for split in SPLITS:
        manifest = json.loads((data / f"split_{split}.json").read_text())
        known = list(map(int, manifest["known_classes"]))
        rogue = list(map(int, manifest["rogue_classes"]))
        train_idx = np.flatnonzero(np.isin(y, known) & (role == 0))
        val_idx = np.flatnonzero(np.isin(y, known) & (role == 1))
        known_idx = np.flatnonzero(np.isin(y, known) & (role == 2))
        rogue_idx = np.flatnonzero(np.isin(y, rogue) & (role == 2))
        test_idx = np.r_[known_idx, rogue_idx]
        for seed in SEEDS:
            lam, beta = selected[(split, seed)]
            entropy_beta = entropy_betas[(split, seed)]
            head = (ROOT / "results/architecture_simplification_v1/experiment2_complementary_only/outer_selected"
                    / f"split{split}_seed{seed}_{CANDIDATE}/stage_b/best.pt")
            f = common_features(
                ROOT, split, seed, known, train_idx, val_idx, test_idx,
                ROOT / f"results/multiknown17_auxce_backbone_split{split}_seed{seed}/best.pt",
                head, CANDIDATE, 128, device,
            )
            nk = len(known_idx)
            is_known = np.r_[np.ones(nk, bool), np.zeros(len(rogue_idx), bool)]
            mapped = np.asarray([known.index(int(v)) for v in y[known_idx]])
            correct = np.r_[f["h_unknown"]["logits"][:nk].argmax(1) == mapped,
                            np.zeros(len(rogue_idx), bool)]

            # Preserve/Aggregate: incremental spaces and fixed aggregation rules.
            for label, names in (("Deep", ("deep",)),
                                 ("Deep + explicit RF", ("deep", "raw")),
                                 ("Deep + explicit RF + learned AST", ("deep", "raw", "physical"))):
                vs = aggregate(f["val_cdf"], names, "fisher")
                ts = aggregate(f["unknown_cdf"], names, "fisher")
                reject = ts > quantile(vs, .90, "linear")
                space_rows.append({"split": split, "seed": seed, "evidence_spaces": label,
                                   **metric_row(is_known, correct, ts, reject)})
            for rule in ("fisher", "stouffer", "cauchy", "mean", "maximum"):
                names = ("deep", "raw", "physical")
                vs = aggregate(f["val_cdf"], names, rule)
                ts = aggregate(f["unknown_cdf"], names, rule)
                reject = ts > quantile(vs, .90, "linear")
                rule_rows.append({"split": split, "seed": seed, "rule": rule,
                                  **metric_row(is_known, correct, ts, reject)})

            fisher_reject = f["uf"] > f["tau_f"]
            final_rows.append({"split": split, "seed": seed, "method": "Primary Fisher",
                               "lambda": "", "beta": "", **metric_row(is_known, correct, f["uf"], fisher_reject)})
            for label, current_lambda, current_beta in (
                    ("Entropy-only", 0.0, entropy_beta),
                    ("Geometry-aware Fisher-retaining", lam, beta)):
                review_val = f["val_sh"] + current_lambda * f["val_sc"]
                review_test = f["unk_sh"] + current_lambda * f["unk_sc"]
                tau_r = quantile(review_val, .95, "higher")
                reject = (~f["ua"]) | (f["ua"] & (review_test > tau_r))
                _, score, _, _ = build_retaining_scores(f, current_lambda, current_beta)
                final_rows.append({"split": split, "seed": seed, "method": label,
                                   "lambda": current_lambda, "beta": current_beta,
                                   **metric_row(is_known, correct, score, reject)})

            accepted = f["ua"]
            labels = (~is_known).astype(int)
            _, retained, _, _ = build_retaining_scores(f, lam, beta)
            accepted_scores = {
                "Primary Fisher": f["uf"], "Entropy": f["unk_sh"],
                "Geometry": f["unk_sc"], "Review": f["unk_sh"] + lam * f["unk_sc"],
                "Fisher-retaining": retained,
            }
            for name, score in accepted_scores.items():
                accepted_rows.append({"split": split, "seed": seed, "score": name,
                                      "accepted_region_auc": roc_auc_score(labels[accepted], score[accepted])})

            # Exhaustive openness with the final Fisher-retaining score and frozen decisions.
            review = f["unk_sh"] + lam * f["unk_sc"]
            tau_r = quantile(f["val_sh"] + lam * f["val_sc"], .95, "higher")
            reject = (~f["ua"]) | (f["ua"] & (review > tau_r))
            for cardinality in range(1, 9):
                for subset in itertools.combinations(rogue, cardinality):
                    mask = np.r_[np.ones(nk, bool), np.isin(y[rogue_idx], subset)]
                    row = metric_row(is_known[mask], correct[mask], retained[mask], reject[mask])
                    openness_rows.append({"split": split, "seed": seed, "R": cardinality,
                                          "rogue_subset": ",".join(map(str, subset)), **row})
            print(json.dumps({"completed": [split, seed], "lambda": lam, "beta": beta,
                              "entropy_beta": entropy_beta}), flush=True)

    metrics = ("auc", "eer", "known_acc", "known_accept", "known_correct_accept",
               "unknown_recall", "unknown_precision", "unknown_f1", "oscr", "overall_open_accuracy")
    write_csv(OUT / "final_outer_results.csv", final_rows)
    write_csv(OUT / "final_outer_summary.csv", summarize(final_rows, "method", metrics))
    write_csv(OUT / "space_increment_results.csv", space_rows)
    write_csv(OUT / "space_increment_summary.csv", summarize(space_rows, "evidence_spaces", metrics))
    write_csv(OUT / "aggregation_rule_results.csv", rule_rows)
    write_csv(OUT / "aggregation_rule_summary.csv", summarize(rule_rows, "rule", metrics))
    write_csv(OUT / "accepted_region_results.csv", accepted_rows)
    write_csv(OUT / "accepted_region_summary.csv", summarize(accepted_rows, "score", ("accepted_region_auc",)))
    write_csv(OUT / "openness_results.csv", openness_rows)
    write_csv(OUT / "openness_summary.csv", summarize(openness_rows, "R", metrics))
    (OUT / "PROTOCOL.md").write_text(
        "# Final fixed-AST complementary-only paper evaluation\n\n"
        "All nine Stage-A and Stage-B checkpoints are frozen. The learned physical branch uses AST only and "
        "z_p=Norm(h_cmp); phase/correlation remains in the complete explicit RF descriptor. Fisher thresholds "
        "use known validation Q90. Review thresholds use Fisher-accepted known validation Q95. Lambda and beta "
        "come from strict inner LOEO. Outer rogue samples are evaluation-only.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
