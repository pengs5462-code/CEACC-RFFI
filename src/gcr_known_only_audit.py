#!/usr/bin/env python3
"""Leakage-free known-only calibration and frozen-output GCR evaluation.

Two deliberately separate commands are provided:

``calibrate`` reads only known-validation samples and frozen bundles.  It
creates an immutable calibration manifest.  ``evaluate`` consumes that
manifest plus already-exported test per-sample outputs.  It never changes the
backbone, Fisher calibration, Fisher threshold, or the canonical rogue order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SPLITS = (1, 2, 3)
SEEDS = (42, 2023, 3407)
R_VALUES = (1, 4, 5, 6, 7, 8)
STAGES = ("fisher_only", "fisher_plus_gcr_known_only")
METRICS = (
    "auc", "eer", "known_acc", "known_accept", "known_correct_accept",
    "unknown_recall", "unknown_precision", "unknown_f1", "oscr",
    "overall_open_accuracy",
)
PROTOCOL_ID = "gcr_known_only_symmetric_joint_tail_v1"
BASE_QUANTILE = 0.80
QUANTILE_STEP = 0.01
MAX_QUANTILE = 0.99
KNOWN_REJECTION_BUDGET = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, float), q, method="linear"))


def choose_rule(cosine: np.ndarray, entropy: np.ndarray) -> dict:
    """Choose a distribution-only joint-tail rule under a 5% known budget.

    The rule starts with cosine <= Q20 and entropy >= Q80.  If their
    intersection exceeds the budget, both tails are tightened symmetrically
    in one-percentile increments.  Neither labels/correctness nor unknown
    observations enter this procedure.
    """
    cosine = np.asarray(cosine, float)
    entropy = np.asarray(entropy, float)
    n = len(cosine)
    cap = int(np.floor(KNOWN_REJECTION_BUDGET * n))
    selected = None
    for q_int in range(int(round(BASE_QUANTILE * 100)), int(round(MAX_QUANTILE * 100)) + 1):
        q = q_int / 100.0
        tau_c = quantile(cosine, 1.0 - q)
        tau_h = quantile(entropy, q)
        hit = (cosine <= tau_c) & (entropy >= tau_h)
        count = int(hit.sum())
        if count <= cap:
            selected = (q, tau_c, tau_h, count)
            break
    if selected is None:
        raise RuntimeError("no feasible known-only GCR rule up to q=0.99")
    q, tau_c, tau_h, count = selected
    return {
        "tail_quantile": q,
        "cosine_max": tau_c,
        "entropy_min": tau_h,
        "n_fisher_accepted_known_validation": n,
        "known_validation_budget_fraction": KNOWN_REJECTION_BUDGET,
        "known_validation_budget_count": cap,
        "known_validation_gcr_reject_count": count,
        "known_validation_gcr_reject_rate": count / max(n, 1),
    }


def calibrate(args: argparse.Namespace) -> None:
    # Heavy imports are confined to calibration.  The test evaluation command
    # does not load a model or dataset.
    from baseline_api import load_baseline
    from experiments.error_forensics import build_features, infer

    dataset = Path(args.dataset)
    x = np.load(dataset / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(dataset / "Y_global.npy")
    role = np.load(dataset / "sample_role.npy")
    rows = []
    for split in SPLITS:
        split_manifest_path = dataset / f"split_{split}.json"
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        known_classes = np.asarray(split_manifest["known_classes"], int)
        # This is the only sample index constructed in calibration mode.
        # role==1 is the known-validation partition; role==2/test is never read.
        val_index = np.flatnonzero(np.isin(y, known_classes) & (role == 1))
        for seed in SEEDS:
            bundle_path = Path(args.bundle_template.format(split=split, seed=seed))
            model = load_baseline(str(bundle_path), args.device)
            if int(model.config["deployment_split"]) != split:
                raise RuntimeError(f"bundle split mismatch: {bundle_path}")
            if int(model.config["deployment_seed"]) != seed:
                raise RuntimeError(f"bundle seed mismatch: {bundle_path}")
            val_out = infer(model, x, val_index, args.batch_size)
            features = build_features(model, val_out)
            fisher_tau = float(model.bundle["fused_threshold"])
            fisher_accept = features["fisher"] <= fisher_tau
            if not fisher_accept.any():
                raise RuntimeError(f"no Fisher-accepted validation samples: split={split}, seed={seed}")
            rule = choose_rule(
                features["view_projected_cos_mean"][fisher_accept],
                features["classifier_entropy"][fisher_accept],
            )
            pred = val_out["predicted_global_class"].astype(int)
            correct = pred == y[val_index]
            gcr = (
                fisher_accept
                & (features["view_projected_cos_mean"] <= rule["cosine_max"])
                & (features["classifier_entropy"] >= rule["entropy_min"])
            )
            rows.append({
                "split": split,
                "seed": seed,
                "known_classes": list(map(int, known_classes)),
                "bundle": str(bundle_path.resolve()),
                "bundle_sha256": sha256(bundle_path),
                "split_manifest": str(split_manifest_path.resolve()),
                "split_manifest_sha256": sha256(split_manifest_path),
                "fisher_threshold_unchanged": fisher_tau,
                "n_known_validation": int(len(val_index)),
                "n_fisher_rejected_known_validation": int((~fisher_accept).sum()),
                "n_gcr_rejected_known_validation": int(gcr.sum()),
                "n_gcr_rejected_correct_known_validation": int(np.sum(gcr & correct)),
                "n_gcr_rejected_misclassified_known_validation": int(np.sum(gcr & ~correct)),
                **rule,
            })
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_BEFORE_TEST_EVALUATION",
        "calibration_scope": "per split/seed, known-validation only",
        "sample_access": "known classes with sample_role == 1 only; no role == 2 indices are constructed",
        "fisher": "unchanged bundle fused_threshold and unchanged Fisher implementation",
        "gcr_application": "Fisher-accepted samples only",
        "rule": {
            "base": "cosine <= Q20 and normalized entropy >= Q80 among Fisher-accepted known-validation samples",
            "budget": "at most floor(0.05 * n_fisher_accepted_known_validation) additional rejections",
            "tightening": "if needed, use Q(1-q) cosine and Q(q) entropy for q=0.81,...,0.99; take first feasible q",
            "quantile_method": "numpy linear",
            "uses_validation_truth_or_correctness_for_selection": False,
            "uses_known_test": False,
            "uses_any_rogue_or_unknown": False,
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"calibration_manifest": str(output), "n_runs": len(rows)}, indent=2))


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    lab = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(lab == 1)[ends]
    fp = np.cumsum(lab == 0)[ends]
    if len(fp) > 2:
        keep = np.where(np.r_[True, np.logical_or(np.diff(fp, 2), np.diff(tp, 2)), True])[0]
        fp, tp = fp[keep], tp[keep]
    return np.r_[0.0, fp / max(int((labels == 0).sum()), 1)], np.r_[0.0, tp / max(int((labels == 1).sum()), 1)]


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = roc_points(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def oscr(is_known: np.ndarray, correct: np.ndarray, score: np.ndarray) -> float:
    fpr, ccr = [], []
    nk = max(int(is_known.sum()), 1)
    nu = max(int((~is_known).sum()), 1)
    for threshold in np.r_[-np.inf, np.unique(score), np.inf]:
        accepted = score <= threshold
        fpr.append(np.sum(accepted & ~is_known) / nu)
        ccr.append(np.sum(accepted & is_known & correct) / nk)
    return float(np.trapz(ccr, fpr))


def metric_row(is_known: np.ndarray, correct: np.ndarray, score: np.ndarray, reject: np.ndarray) -> dict:
    labels = (~is_known).astype(int)
    tp = int(np.sum(reject & ~is_known))
    fp = int(np.sum(reject & is_known))
    fn = int(np.sum(~reject & ~is_known))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    kca_count = int(np.sum(is_known & correct & ~reject))
    return {
        "n_known": int(is_known.sum()),
        "n_unknown": int((~is_known).sum()),
        "auc": auc_rank(labels, score),
        "eer": eer(labels, score),
        "known_acc": float(correct[is_known].mean()),
        "known_accept": float((~reject[is_known]).mean()),
        "known_correct_accept": float((correct[is_known] & ~reject[is_known]).mean()),
        "unknown_recall": recall,
        "unknown_precision": precision,
        "unknown_f1": 2 * precision * recall / max(precision + recall, 1e-15),
        "oscr": oscr(is_known, correct, score),
        "overall_open_accuracy": (kca_count + tp) / len(is_known),
    }


def summarize(rows: list[dict], groups: tuple[str, ...]) -> list[dict]:
    keys = sorted({tuple(row[g] for g in groups) for row in rows})
    output = []
    for key in keys:
        subset = [row for row in rows if tuple(row[g] for g in groups) == key]
        item = {g: v for g, v in zip(groups, key)}
        item.update({"n_runs": len(subset), "std_ddof": 1})
        for metric in METRICS:
            values = np.asarray([row[metric] for row in subset], float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    calibration_path = Path(args.calibration)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration["protocol_id"] != PROTOCOL_ID or calibration["status"] != "FROZEN_BEFORE_TEST_EVALUATION":
        raise RuntimeError("calibration manifest is not the frozen known-only protocol")
    cal = {(int(r["split"]), int(r["seed"])): r for r in calibration["runs"]}
    if set(cal) != {(s, q) for s in SPLITS for q in SEEDS}:
        raise RuntimeError("calibration manifest does not contain exactly 3x3 runs")

    full_rows, canonical_rows, delta_rows = [], [], []
    for split in SPLITS:
        for seed in SEEDS:
            result_path = Path(args.test_root) / f"ours_split{split}_seed{seed}" / "result.json"
            score_path = result_path.with_name("per_sample_scores.npz")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            with np.load(score_path, allow_pickle=False) as z:
                # Old GCR fields are read only for the explicit comparison after
                # the new calibration manifest has already been frozen.
                a = {k: z[k].copy() for k in z.files}
            c = cal[(split, seed)]
            if abs(float(result["metrics"]["tau_known_only"]) - float(c["fisher_threshold_unchanged"])) > 1e-12:
                raise RuntimeError(f"Fisher threshold mismatch: split={split}, seed={seed}")
            base = a["fisher"] > float(c["fisher_threshold_unchanged"])
            if not np.array_equal(base, a["base_reject"]):
                raise RuntimeError(f"stored Fisher decision mismatch: split={split}, seed={seed}")
            gcr = (~base) & (a["cosine"] <= float(c["cosine_max"])) & (a["entropy"] >= float(c["entropy_min"]))
            final = base | gcr
            decision = a["fisher"].copy()
            decision[gcr] = np.nextafter(float(c["fisher_threshold_unchanged"]), np.inf)
            stage_data = {
                "fisher_only": (a["fisher"], base),
                "fisher_plus_gcr_known_only": (decision, final),
            }
            common = {
                "protocol_id": PROTOCOL_ID,
                "split": split,
                "seed": seed,
                "known_classes": json.dumps(result["known_classes"], separators=(",", ":")),
                "rogue_classes": json.dumps(result["rogue_classes"], separators=(",", ":")),
                "tau_fisher": float(c["fisher_threshold_unchanged"]),
                "tau_cosine": float(c["cosine_max"]),
                "tau_entropy": float(c["entropy_min"]),
                "tail_quantile": float(c["tail_quantile"]),
                "known_validation_gcr_reject_rate": float(c["known_validation_gcr_reject_rate"]),
                "source_npz": str(score_path.resolve()),
                "source_npz_sha256": sha256(score_path),
            }
            run_metrics = {}
            for stage, (score, reject) in stage_data.items():
                metrics = metric_row(a["is_known"], a["correct"], score, reject)
                run_metrics[stage] = metrics
                full_rows.append({**common, "stage": stage, **metrics})
                for r in R_VALUES:
                    rogue_subset = tuple(result["rogue_classes"][:r])
                    mask = a["is_known"] | np.isin(a["truth"], rogue_subset)
                    canonical_rows.append({
                        **common,
                        "stage": stage,
                        "openness": f"9:{r}",
                        "R": r,
                        "rogue_subset": json.dumps(rogue_subset, separators=(",", ":")),
                        **metric_row(a["is_known"][mask], a["correct"][mask], score[mask], reject[mask]),
                    })
            fisher_metrics = run_metrics["fisher_only"]
            new_metrics = run_metrics["fisher_plus_gcr_known_only"]
            old_metrics = metric_row(a["is_known"], a["correct"], a["decision_score"], a["final_reject"])
            known = a["is_known"]
            rogue = ~known
            delta_rows.append({
                **common,
                "gcr_additional_rejected_known": int(np.sum(known & ~base & gcr)),
                "gcr_additional_rejected_correct_known": int(np.sum(known & a["correct"] & ~base & gcr)),
                "gcr_rescued_fisher_false_accepted_rogue": int(np.sum(rogue & ~base & gcr)),
                "delta_unknown_recall": new_metrics["unknown_recall"] - fisher_metrics["unknown_recall"],
                "delta_known_correct_accept": new_metrics["known_correct_accept"] - fisher_metrics["known_correct_accept"],
                "delta_overall_open_accuracy": new_metrics["overall_open_accuracy"] - fisher_metrics["overall_open_accuracy"],
                "new_minus_old_frozen_unknown_recall": new_metrics["unknown_recall"] - old_metrics["unknown_recall"],
                "new_minus_old_frozen_known_correct_accept": new_metrics["known_correct_accept"] - old_metrics["known_correct_accept"],
                "new_minus_old_frozen_overall_open_accuracy": new_metrics["overall_open_accuracy"] - old_metrics["overall_open_accuracy"],
            })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(full_rows, ("stage",))
    canonical_summary = summarize(canonical_rows, ("stage", "openness", "R"))
    write_csv(output / "gcr_known_only_9run_results.csv", full_rows)
    write_csv(output / "gcr_known_only_9run_summary.csv", summary)
    write_csv(output / "gcr_known_only_deltas.csv", delta_rows)
    write_csv(output / "gcr_known_only_canonical_results.csv", canonical_rows)
    write_csv(output / "gcr_known_only_canonical_summary.csv", canonical_summary)
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "calibration_manifest": str(calibration_path.resolve()),
        "calibration_manifest_sha256": sha256(calibration_path),
        "test_root": str(Path(args.test_root).resolve()),
        "fisher_unchanged": True,
        "backbone_retrained": False,
        "test_threshold_adjustment": False,
        "canonical_subsets": "first R rogue IDs in each already-frozen split manifest order",
        "R_values": list(R_VALUES),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "gcr_known_only_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output),
        "full_rows": len(full_rows),
        "summary_rows": len(summary),
        "delta_rows": len(delta_rows),
        "canonical_rows": len(canonical_rows),
        "canonical_summary_rows": len(canonical_summary),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--dataset", default="dataset/multiknown17_v1")
    c.add_argument("--bundle-template", required=True)
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--batch-size", type=int, default=64)
    c.add_argument("--output", required=True)
    c.set_defaults(func=calibrate)
    e = sub.add_parser("evaluate")
    e.add_argument("--calibration", required=True)
    e.add_argument("--test-root", required=True)
    e.add_argument("--output-dir", required=True)
    e.set_defaults(func=evaluate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
