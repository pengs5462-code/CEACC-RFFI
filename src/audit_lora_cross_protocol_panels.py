#!/usr/bin/env python3
"""Transfer the final ADS-B Fisher-retaining score to frozen LoRa runs.

No LoRa model is trained and no LoRa test/rogue sample selects lambda, beta,
calibration distributions, or thresholds.  Seed-matched lambda and beta are
loaded from the final ADS-B split-1 inner-LOEO selection tables.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from calibrated_rejection_fusion_multiknown import reference_cdf
from continuous_gcr_loeo import empirical_p, loo_p, quantile
from gcr_known_only_audit import metric_row
from rf_physics_feature_probe import extract_all
from train_causal_ablation import build_model, extract, seed_everything
from train_multiknown17_baseline import make_loader


SEEDS = (42, 2023, 3407)
RANK_METRICS = ("auc", "eer", "oscr")
OPERATING_METRICS = ("known_acc", "known_accept", "known_correct_accept",
                     "unknown_recall", "unknown_precision", "unknown_f1",
                     "overall_open_accuracy")


def safe_cosine_mean(projected):
    pairs = []
    for i in range(projected.shape[1]):
        for j in range(i + 1, projected.shape[1]):
            a, b = projected[:, i], projected[:, j]
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            pairs.append(np.sum(a * b, axis=1) / np.clip(denom, 1e-12, None))
    return np.column_stack(pairs).mean(1)


@torch.inference_mode()
def head_outputs(head, deep, physics, batch_size, device):
    """Extract review cues for the configured physical view set (AST here)."""
    logits, physical, projected = [], [], []
    head.eval()
    for start in range(0, len(deep), batch_size):
        d = torch.from_numpy(deep[start:start + batch_size].astype(np.float32)).to(device)
        p = torch.from_numpy(physics[start:start + batch_size].astype(np.float32)).to(device)
        z_p, view_projected, _ = head.encode_physics(p)
        logit = head.classifier(torch.cat([F.normalize(d, dim=1), z_p], dim=1))
        logits.append(logit.cpu().numpy())
        physical.append(z_p.cpu().numpy())
        projected.append(torch.stack(view_projected, dim=1).cpu().numpy())
    logits = np.concatenate(logits)
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(1) / np.log(logits.shape[1])
    projected = np.concatenate(projected)
    if projected.shape[1:] != (3, 32):
        raise RuntimeError(f"expected three AST 32D projected views, got {projected.shape}")
    return {
        "logits": logits,
        "z_p": np.concatenate(physical),
        "cosine": safe_cosine_mean(projected),
        "entropy": entropy,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected(path: Path, field: str, split: int = 1) -> dict[int, float]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            is_selected = row.get("selected", "true").lower() == "true"
            if int(row["split"]) == split and is_selected:
                seed = int(row["seed"])
                if seed in SEEDS:
                    result[seed] = float(row[field])
    missing = sorted(set(SEEDS) - set(result))
    if missing:
        raise RuntimeError(f"missing {field} for seeds {missing} in {path}")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate_seed(args, seed: int, lam: float, beta: float, entropy_beta: float) -> dict:
    result_root = args.project_root / "results/lora_rffi_known30_unknown15_v1"
    base_path = result_root / f"nettime_ce_mondrian_split1_seed{seed}/best.pt"
    head_path = result_root / f"physical_ast_complementary_split1_seed{seed}/best.pt"
    score_path = result_root / "ast_complementary" / f"rejection_scores_split1_seed{seed}.npz"
    benchmark_path = result_root / "ast_complementary" / f"rejection_split1_seed{seed}.json"

    base = torch.load(base_path, map_location="cpu")
    head_checkpoint = torch.load(head_path, map_location="cpu")
    model_args = argparse.Namespace(**base["args"])
    if int(model_args.seed) != seed or int(model_args.split) != 1:
        raise RuntimeError("checkpoint seed/split mismatch")
    data_root = Path(model_args.dataset_root)
    manifest_path = data_root / "split_1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    x = np.load(data_root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(data_root / "Y_global.npy")
    role = np.load(data_root / "sample_role.npy")
    known = list(map(int, manifest["known_classes"]))
    rogue = list(map(int, manifest["rogue_classes"]))
    mapping = {class_id: position for position, class_id in enumerate(known)}
    indices = {
        "validation": np.flatnonzero(np.isin(y, known) & (role == 1)),
        "known_test": np.flatnonzero(np.isin(y, known) & (role == 2)),
        "rogue_test": np.flatnonzero(np.isin(y, rogue) & (role == 2)),
    }

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    backbone = build_model(model_args, device)
    backbone.load_state_dict(base["model"]); backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    from train_inner_physical_candidate import (
        InnerPhysicalCandidateHead, configure as configure_candidate,
    )
    configure_candidate(head_checkpoint["candidate"])
    head = InnerPhysicalCandidateHead(64, len(known)).to(device)
    head.load_state_dict(head_checkpoint["head"])
    keep = head_checkpoint.get("descriptor_keep_indices")

    def outputs_for(name: str, global_labels: bool = False) -> dict:
        ids = indices[name]
        if global_labels:
            loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(x[ids])),
                                torch.zeros(len(ids), dtype=torch.int64)),
                                batch_size=args.batch_size, shuffle=False)
        else:
            loader = make_loader(x, y, ids, mapping, args.batch_size, False)
        deep, _, labels = extract(backbone, loader, device)
        raw = extract_all(x[ids], 64)
        if keep is not None:
            raw = raw[:, np.asarray(keep, dtype=np.int64)]
        physics = ((raw - head_checkpoint["physics_mean"]) / head_checkpoint["physics_std"]).astype(np.float32)
        output = head_outputs(head, deep.astype(np.float32), physics, args.batch_size, device)
        output["labels"] = labels
        return output

    # Known-only calibration is completed before either test partition is read.
    archive = np.load(score_path)
    if int(archive["seed"]) != seed:
        raise RuntimeError("score/checkpoint seed mismatch")
    val_score = np.asarray(archive["validation_score"], dtype=np.float64)
    val_output = outputs_for("validation")
    tau_f = quantile(val_score, 0.90, "linear")
    val_accept = val_score <= tau_f
    ref_c = val_output["cosine"][val_accept]
    ref_h = val_output["entropy"][val_accept]
    val_sc = 1.0 - loo_p(ref_c, "lower")
    val_sh = 1.0 - loo_p(ref_h, "upper")
    val_sr = val_sh + lam * val_sc
    val_f_accepted = val_score[val_accept]

    # One-shot test evaluation after all transferred choices are frozen.
    known_output = outputs_for("known_test")
    rogue_output = outputs_for("rogue_test", global_labels=True)
    known_score = np.asarray(archive["known_score"], dtype=np.float64)
    rogue_score = np.asarray(archive["rogue_score"], dtype=np.float64)
    known_correct = np.asarray(archive["known_correct"], dtype=np.bool_)

    def review(output):
        sc = 1.0 - empirical_p(ref_c, output["cosine"], "lower")
        sh = 1.0 - empirical_p(ref_h, output["entropy"], "upper")
        return sh, sc, sh + lam * sc

    known_sh, known_sc, known_sr = review(known_output)
    rogue_sh, rogue_sc, rogue_sr = review(rogue_output)
    known_accept, rogue_accept = known_score <= tau_f, rogue_score <= tau_f

    def current_cascade(score, sr, accepted):
        return np.where(accepted, sr, 1.0 + lam + reference_cdf(val_score, score))

    def retaining_cascade(score, sr, accepted, validation_review, current_beta):
        out = 1.0 + reference_cdf(val_score, score)
        rf = reference_cdf(val_f_accepted, score[accepted])
        rr = reference_cdf(validation_review, sr[accepted])
        out[accepted] = (1.0 - current_beta) * rf + current_beta * rr
        return out

    known_current = current_cascade(known_score, known_sr, known_accept)
    rogue_current = current_cascade(rogue_score, rogue_sr, rogue_accept)
    known_retaining = retaining_cascade(known_score, known_sr, known_accept, val_sr, beta)
    rogue_retaining = retaining_cascade(rogue_score, rogue_sr, rogue_accept, val_sr, beta)
    known_entropy = retaining_cascade(known_score, known_sh, known_accept, val_sh, entropy_beta)
    rogue_entropy = retaining_cascade(rogue_score, rogue_sh, rogue_accept, val_sh, entropy_beta)

    is_known = np.r_[np.ones(len(known_score), bool), np.zeros(len(rogue_score), bool)]
    correct = np.r_[known_correct, np.zeros(len(rogue_score), bool)]
    fisher_score = np.r_[known_score, rogue_score]
    accepted = np.r_[known_accept, rogue_accept]
    primary_reject = fisher_score > tau_f
    tau_r = quantile(val_sr, 0.95, "higher")
    final_reject = primary_reject | (accepted & (np.r_[known_sr, rogue_sr] > tau_r))
    score_sets = {
        "primary_fisher": fisher_score,
        "entropy_only_fisher_retaining": np.r_[known_entropy, rogue_entropy],
        "fisher_retaining_with_transferred_lambda_beta": np.r_[known_retaining, rogue_retaining],
    }
    metrics = {}
    labels = (~is_known).astype(int)
    from sklearn.metrics import roc_auc_score
    for name, score in score_sets.items():
        row = metric_row(is_known, correct, score, primary_reject)
        metrics[name] = {metric: float(row[metric]) for metric in RANK_METRICS}
        metrics[name]["accepted_region_auc"] = float(roc_auc_score(labels[accepted], score[accepted]))

    cue_scores = {
        "primary_fisher": fisher_score,
        "entropy_tail": np.r_[known_sh, rogue_sh],
        "geometry_tail": np.r_[known_sc, rogue_sc],
        "combined_review": np.r_[known_sr, rogue_sr],
        "fisher_retaining_final": np.r_[known_retaining, rogue_retaining],
    }
    accepted_region = {
        name: float(roc_auc_score(labels[accepted], score[accepted]))
        for name, score in cue_scores.items()
    }
    operating = {}
    for name, score, decision in (
            ("primary_fisher", fisher_score, primary_reject),
            ("fisher_retaining_final", np.r_[known_retaining, rogue_retaining], final_reject)):
        row = metric_row(is_known, correct, score, decision)
        operating[name] = {metric: float(row[metric]) for metric in OPERATING_METRICS}

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    by_name = {row["score"]: row for row in benchmark["results"]}
    spaces = {}
    for label, source in (
            ("Deep", "deep"),
            ("Deep + explicit RF", "fisher_deep_raw_baseline"),
            ("Deep + explicit RF + learned AST", "fisher_three_space_latent_or_mondrian_msp")):
        spaces[label] = {metric: float(by_name[source][metric]) for metric in ("auc", "oscr")}

    result = {
        "method": "lora_fixed_ast_complementary_fisher_retaining_transfer_v1",
        "seed": seed, "split": 1,
        "transferred": {
            "lambda": lam, "beta": beta, "entropy_beta": entropy_beta,
            "source": "ADS-B split-1 seed-matched strict inner-LOEO selections",
        },
        "formula": {
            "accepted_fisher_rank": "r_F = empirical CDF of S_F against Fisher-accepted LoRa known validation",
            "accepted_review_rank": "r_R = empirical CDF of S_R against Fisher-accepted LoRa known validation",
            "accepted_branch": "T_beta = (1-beta)*r_F + beta*r_R",
            "rejected_branch": "1 + empirical CDF of S_F against all LoRa known validation",
        },
        "metrics": metrics,
        "accepted_region_auc": accepted_region,
        "fixed_decision_metrics": operating,
        "space_increment": spaces,
        "protocol": {
            "stage_a_retrained": False,
            "stage_b_retrained_on_enrolled_lora_only": True,
            "lora_target_unknown_used_for_lambda_or_beta_selection": False,
            "lora_known_test_used_for_lambda_or_beta_selection": False,
            "calibration_uses_known_validation_only": True,
            "test_used_only_for_final_metrics": True,
        },
        "hashes": {
            "script": sha256(Path(__file__)), "base_checkpoint": sha256(base_path),
            "head_checkpoint": sha256(head_path), "score_archive": sha256(score_path),
            "manifest": sha256(manifest_path), "benchmark": sha256(benchmark_path),
            "lambda_selection": sha256(args.lambda_selection), "beta_selection": sha256(args.beta_selection),
        },
    }
    del backbone, head, base, head_checkpoint
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def entropy_betas(inner_dir: Path) -> dict[int, float]:
    rows = []
    for path in sorted(inner_dir.glob("inner_worker*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    selected = {}
    for seed in SEEDS:
        candidates = []
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            rr = [r for r in rows if int(r["split"]) == 1 and int(r["seed"]) == seed
                  and float(r["lambda"]) == 0.0 and float(r["beta"]) == beta]
            candidates.append((np.mean([float(r["retaining_oscr"]) for r in rr]),
                               min(float(r["retaining_oscr"]) for r in rr), -beta, beta))
        selected[seed] = max(candidates)[3]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lambda-selection", type=Path, required=True)
    parser.add_argument("--beta-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-results-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.lambda_selection = args.lambda_selection.resolve()
    args.beta_selection = args.beta_selection.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lambdas = load_selected(args.lambda_selection, "lambda")
    betas = load_selected(args.beta_selection, "beta")
    entropy_beta = entropy_betas(args.inner_results_dir.resolve())

    results = []
    for seed in SEEDS:
        result = evaluate_seed(args, seed, lambdas[seed], betas[seed], entropy_beta[seed])
        results.append(result)
        (args.output_dir / f"seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"completed": seed, "lambda": lambdas[seed], "beta": betas[seed],
                          "entropy_beta": entropy_beta[seed]}), flush=True)

    rows = []
    for result in results:
        for method, metrics in result["metrics"].items():
            rows.append({"seed": result["seed"], "lambda": result["transferred"]["lambda"],
                         "beta": result["transferred"]["beta"], "method": method, **metrics})
    write_csv(args.output_dir / "lora_fisher_retaining_by_seed.csv", rows)
    summary = []
    for method in results[0]["metrics"]:
        rr = [r for r in rows if r["method"] == method]
        item = {"method": method, "n_runs": len(rr)}
        for metric in ("accepted_region_auc",) + RANK_METRICS:
            values = np.asarray([r[metric] for r in rr])
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_sample_std"] = float(values.std(ddof=1))
        summary.append(item)
    write_csv(args.output_dir / "lora_fisher_retaining_summary.csv", summary)
    ar_rows, op_rows, space_rows = [], [], []
    for result in results:
        for score, value in result["accepted_region_auc"].items():
            ar_rows.append({"seed": result["seed"], "score": score, "accepted_region_auc": value})
        for method, values in result["fixed_decision_metrics"].items():
            op_rows.append({"seed": result["seed"], "method": method, **values})
        for evidence, values in result["space_increment"].items():
            space_rows.append({"seed": result["seed"], "evidence_spaces": evidence, **values})
    write_csv(args.output_dir / "lora_accepted_region_by_seed.csv", ar_rows)
    write_csv(args.output_dir / "lora_operating_point_by_seed.csv", op_rows)
    write_csv(args.output_dir / "lora_space_increment_by_seed.csv", space_rows)
    def grouped_summary(rows, key, fields):
        output = []
        for value in dict.fromkeys(row[key] for row in rows):
            rr = [row for row in rows if row[key] == value]
            item = {key: value, "n_runs": len(rr)}
            for field in fields:
                vals = np.asarray([float(row[field]) for row in rr])
                item[field + "_mean"] = float(vals.mean())
                item[field + "_sample_std"] = float(vals.std(ddof=1))
            output.append(item)
        return output
    write_csv(args.output_dir / "lora_accepted_region_summary.csv",
              grouped_summary(ar_rows, "score", ("accepted_region_auc",)))
    write_csv(args.output_dir / "lora_operating_point_summary.csv",
              grouped_summary(op_rows, "method", OPERATING_METRICS))
    write_csv(args.output_dir / "lora_space_increment_summary.csv",
              grouped_summary(space_rows, "evidence_spaces", ("auc", "oscr")))
    (args.output_dir / "audit_manifest.json").write_text(json.dumps({
        "status": "COMPLETE_EVALUATION_ONLY", "seeds": list(SEEDS),
        "transferred_lambdas": lambdas, "transferred_betas": betas,
        "transferred_entropy_betas": entropy_beta,
        "lambda_selection_sha256": sha256(args.lambda_selection),
        "beta_selection_sha256": sha256(args.beta_selection),
        "no_training": True, "no_lora_target_unknown_selection": True,
    }, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
