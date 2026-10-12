#!/usr/bin/env python3
"""Strict inner-LOEO selection for the continuous geometry-aware review score."""
from __future__ import annotations

import argparse
import csv
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
from spectral_conformal_benchmark import deterministic
from train_causal_ablation import build_model, extract, seed_everything
from train_multiknown17_baseline import make_loader
from train_physics_structure_baseline import head_from_checkpoint


LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
SPLITS = (1, 2, 3)
SEEDS = (42, 2023, 3407)
EPS = 1e-12


def quantile(x: np.ndarray, q: float, method: str = "linear") -> float:
    return float(np.quantile(np.asarray(x, float), q, method=method))


def empirical_p(reference: np.ndarray, values: np.ndarray, side: str) -> np.ndarray:
    reference = np.asarray(reference, float)
    values = np.asarray(values, float)
    if side == "lower":
        count = np.asarray([(reference <= v).sum() for v in values])
    elif side == "upper":
        count = np.asarray([(reference >= v).sum() for v in values])
    else:
        raise ValueError(side)
    return (1.0 + count) / (len(reference) + 1.0)


def loo_p(values: np.ndarray, side: str) -> np.ndarray:
    values = np.asarray(values, float)
    out = np.empty(len(values), float)
    for i, value in enumerate(values):
        reference = np.delete(values, i)
        if side == "lower":
            count = int((reference <= value).sum())
        elif side == "upper":
            count = int((reference >= value).sum())
        else:
            raise ValueError(side)
        out[i] = (1.0 + count) / len(values)
    return out


def safe_cosine_mean(projected: np.ndarray) -> np.ndarray:
    # projected: N x 4 x 32, exactly the four current view projectors.
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = projected[:, i], projected[:, j]
            pairs.append(
                np.sum(a * b, axis=1)
                / np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), EPS, None)
            )
    return np.column_stack(pairs).mean(1)


@torch.inference_mode()
def head_outputs(head, deep, physics, batch_size, device):
    logits, physical, projected = [], [], []
    head.eval()
    for start in range(0, len(deep), batch_size):
        d = torch.from_numpy(deep[start:start + batch_size].astype(np.float32)).to(device)
        p = torch.from_numpy(physics[start:start + batch_size].astype(np.float32)).to(device)
        z_p, four_projected, _ = head.encode_physics(p)
        logit = head.classifier(torch.cat([F.normalize(d, dim=1), z_p], dim=1))
        logits.append(logit.cpu().numpy())
        physical.append(z_p.cpu().numpy())
        projected.append(torch.stack(four_projected, dim=1).cpu().numpy())
    logits = np.concatenate(logits)
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, EPS, None))).sum(1) / np.log(logits.shape[1])
    projected = np.concatenate(projected)
    if projected.shape[1:] != (4, 32):
        raise RuntimeError(f"expected four 32D projected views, got {projected.shape}")
    return {
        "logits": logits,
        "z_p": np.concatenate(physical),
        "cosine": safe_cosine_mean(projected),
        "entropy": entropy,
    }


def oscr(is_known: np.ndarray, correct: np.ndarray, score: np.ndarray) -> float:
    nk, nu = max(int(is_known.sum()), 1), max(int((~is_known).sum()), 1)
    fpr, ccr = [], []
    for threshold in np.r_[-np.inf, np.unique(score), np.inf]:
        accepted = score <= threshold
        fpr.append(np.sum(accepted & ~is_known) / nu)
        ccr.append(np.sum(accepted & is_known & correct) / nk)
    order = np.argsort(fpr)
    return float(np.trapz(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def metrics(is_known, correct, reject):
    tp = int(np.sum(reject & ~is_known)); fp = int(np.sum(reject & is_known))
    fn = int(np.sum(~reject & ~is_known))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {
        "known_accept": float((~reject[is_known]).mean()),
        "known_correct_accept": float((correct[is_known] & ~reject[is_known]).mean()),
        "unknown_recall": recall,
        "unknown_precision": precision,
        "unknown_f1": 2 * precision * recall / max(precision + recall, EPS),
    }


def evaluate_fold(root: Path, split: int, seed: int, held_out: int, batch_size: int, device):
    exp = root / "results/continuous_gcr_loeo_v2"
    tag = f"split{split}_seed{seed}_heldout{held_out}"
    base_path = exp / "inner" / tag / "stage_a/best.pt"
    head_path = exp / "inner" / tag / "stage_b/best.pt"
    if not base_path.exists() or not head_path.exists():
        return []
    base = torch.load(base_path, map_location="cpu")
    head_ckpt = torch.load(head_path, map_location="cpu")
    active = list(map(int, base["active_known_classes"]))
    if held_out in active or len(active) != 8:
        raise RuntimeError(f"bad active classes for {tag}: {active}")
    args = argparse.Namespace(**base["args"])
    data = Path(args.dataset_root)
    x = np.load(data / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(data / "Y_global.npy")
    role = np.load(data / "sample_role.npy")
    mapping = {v: i for i, v in enumerate(active)}
    train_idx = np.flatnonzero(np.isin(y, active) & (role == 0))
    val_idx = np.flatnonzero(np.isin(y, active) & (role == 1))
    pseudo_idx = np.flatnonzero(y == held_out)  # completely excluded emitter, evaluation only
    if np.any(np.isin(y[train_idx], [held_out])) or np.any(np.isin(y[val_idx], [held_out])):
        raise RuntimeError("held-out emitter leaked into active-known fit/calibration")

    train_loader = make_loader(x, y, train_idx, mapping, args.batch_size, True)
    val_loader = make_loader(x, y, val_idx, mapping, batch_size, False)
    pseudo_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(x[pseudo_idx])), torch.zeros(len(pseudo_idx), dtype=torch.long)),
        batch_size=batch_size, shuffle=False,
    )
    seed_everything(seed)
    backbone = build_model(args, device); backbone.load_state_dict(base["model"]); backbone.eval()
    train_e, _, train_y = extract(backbone, deterministic(train_loader, batch_size), device)
    val_e, _, val_y = extract(backbone, val_loader, device)
    pseudo_e, _, _ = extract(backbone, pseudo_loader, device)

    keep = np.asarray(head_ckpt["descriptor_keep_indices"], dtype=np.int64)
    descriptor = lambda idx: extract_all(x[idx], 64)[:, keep]
    raw_train, raw_val, raw_pseudo = map(descriptor, (train_idx, val_idx, pseudo_idx))
    standardize = lambda a: ((a - head_ckpt["physics_mean"]) / head_ckpt["physics_std"]).astype(np.float32)
    p_train, p_val, p_pseudo = map(standardize, (raw_train, raw_val, raw_pseudo))
    head = head_from_checkpoint(train_e.shape[1], len(active), head_ckpt).to(device)
    head.load_state_dict(head_ckpt["head"])
    h_train = head_outputs(head, train_e, p_train, batch_size, device)
    h_val = head_outputs(head, val_e, p_val, batch_size, device)
    h_pseudo = head_outputs(head, pseudo_e, p_pseudo, batch_size, device)

    dummy = np.zeros((len(train_e), len(active)), np.float32)
    deep = MondrianMahalanobis(train_e, dummy, train_y, val_e, val_y)
    physical = MondrianMahalanobis(h_train["z_p"], dummy, train_y, h_val["z_p"], val_y)
    scaler = StandardScaler().fit(raw_train)
    pca = PCA(n_components=32, whiten=True, random_state=0).fit(scaler.transform(raw_train))
    raw_transform = lambda a: pca.transform(scaler.transform(a))
    raw_svm = SVC(C=10.0, kernel="rbf", gamma="scale", decision_function_shape="ovr").fit(raw_transform(raw_train), train_y)
    val_native = {
        "deep": 1 - leave_one_out_validation_p(deep, val_e, val_y).max(1),
        "raw": -raw_svm.decision_function(raw_transform(raw_val)).max(1),
        "physical": 1 - leave_one_out_validation_p(physical, h_val["z_p"], val_y).max(1),
    }
    pseudo_native = {
        "deep": 1 - deep.p_values(pseudo_e).max(1),
        "raw": -raw_svm.decision_function(raw_transform(raw_pseudo)).max(1),
        "physical": 1 - physical.p_values(h_pseudo["z_p"]).max(1),
    }
    val_cdf = {k: loo_cdf(v) for k, v in val_native.items()}
    pseudo_cdf = {k: reference_cdf(val_native[k], v) for k, v in pseudo_native.items()}
    floor = 1.0 / (len(val_idx) + 1.0)
    fisher = lambda d: -2 * sum(np.log(np.clip(1 - d[k] + floor, EPS, 1)) for k in ("deep", "raw", "physical"))
    val_f, pseudo_f = fisher(val_cdf), fisher(pseudo_cdf)
    tau_f = quantile(val_f, 0.90)
    val_f_accept, pseudo_f_accept = val_f <= tau_f, pseudo_f <= tau_f
    ref_c, ref_h = h_val["cosine"][val_f_accept], h_val["entropy"][val_f_accept]
    val_sc, val_sh = 1 - loo_p(ref_c, "lower"), 1 - loo_p(ref_h, "upper")
    pseudo_sc = 1 - empirical_p(ref_c, h_pseudo["cosine"], "lower")
    pseudo_sh = 1 - empirical_p(ref_h, h_pseudo["entropy"], "upper")
    val_correct = h_val["logits"].argmax(1) == val_y
    known = np.r_[np.ones(len(val_idx), bool), np.zeros(len(pseudo_idx), bool)]
    correct = np.r_[val_correct, np.zeros(len(pseudo_idx), bool)]
    rows = []
    for lam in LAMBDAS:
        val_sr_pool = val_sh + lam * val_sc
        # ``higher`` guarantees empirical acceptance >=95% under finite n;
        # Fisher keeps its established linear Q90 above.
        tau_r = quantile(val_sr_pool, 0.95, method="higher")
        val_sr = np.zeros(len(val_idx)); val_sr[val_f_accept] = val_sr_pool
        pseudo_sr = pseudo_sh + lam * pseudo_sc
        val_reject = (~val_f_accept) | (val_f_accept & (val_sr > tau_r))
        pseudo_reject = (~pseudo_f_accept) | (pseudo_f_accept & (pseudo_sr > tau_r))
        # True continuous cascade: all Fisher-rejected points rank above the
        # bounded review-score interval; within that region Fisher retains its order.
        bound = 1.0 + lam
        val_rank = loo_cdf(val_f); pseudo_rank = reference_cdf(val_f, pseudo_f)
        val_final = np.where(val_f_accept, val_sr, bound + val_rank)
        pseudo_final = np.where(pseudo_f_accept, pseudo_sr, bound + pseudo_rank)
        reject = np.r_[val_reject, pseudo_reject]
        row = {
            "split": split, "seed": seed, "held_out_emitter": held_out,
            "active_known_classes": json.dumps(active, separators=(",", ":")),
            "lambda": lam, "review_threshold_q95": tau_r,
            "n_inner_known_validation": len(val_idx), "n_pseudo_unknown": len(pseudo_idx),
            "n_fisher_accepted_known_validation": int(val_f_accept.sum()),
            "review_stage_known_validation_accept": float((val_sr_pool <= tau_r).mean()),
            "full_known_validation_accept": float((~val_reject).mean()),
            "inner_oscr": oscr(known, correct, np.r_[val_final, pseudo_final]),
            **metrics(known, correct, reject),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--available-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root); out = root / "results/continuous_gcr_loeo_v2"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    data = root / "dataset/multiknown17_v1"
    for split in SPLITS:
        known = json.loads((data / f"split_{split}.json").read_text())["known_classes"]
        for seed in SEEDS:
            for held_out in known:
                part = evaluate_fold(root, split, seed, int(held_out), args.batch_size, device)
                if not part and not args.available_only:
                    raise RuntimeError(f"missing completed fold split={split} seed={seed} heldout={held_out}")
                rows.extend(part)
    write_csv(out / "inner_loeo_all_folds.csv", rows)
    summary = []
    for split in SPLITS:
        for seed in SEEDS:
            group = [r for r in rows if r["split"] == split and r["seed"] == seed]
            if len(group) != 9 * len(LAMBDAS):
                continue
            candidates = []
            for lam in LAMBDAS:
                rr = [r for r in group if r["lambda"] == lam]
                candidates.append({
                    "split": split, "seed": seed, "lambda": lam,
                    "mean_inner_oscr": float(np.mean([r["inner_oscr"] for r in rr])),
                    "sample_std_inner_oscr": float(np.std([r["inner_oscr"] for r in rr], ddof=1)),
                    "worst_fold_oscr": float(np.min([r["inner_oscr"] for r in rr])),
                    "mean_review_stage_known_accept": float(np.mean([r["review_stage_known_validation_accept"] for r in rr])),
                    "selected": False,
                })
            best = max(candidates, key=lambda r: (r["mean_inner_oscr"], r["worst_fold_oscr"], -r["lambda"]))
            best["selected"] = True
            summary.extend(candidates)
    write_csv(out / "lambda_selection_summary.csv", summary)
    print(json.dumps({"fold_lambda_rows": len(rows), "selection_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
