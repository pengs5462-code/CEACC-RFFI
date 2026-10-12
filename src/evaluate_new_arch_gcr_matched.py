#!/usr/bin/env python3
"""Leakage-free Primary/entropy/geometry comparison for selected physical heads.

Inner mode evaluates only the architecture already selected for each split/seed.
It calibrates every review threshold from Fisher-accepted known validation and
uses the held-out emitter only to select lambda.  Outer mode is one-shot and
uses the frozen lambda selected from the corresponding nine inner folds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from calibrated_rejection_fusion_multiknown import loo_cdf, reference_cdf
from conformal_benchmark import MondrianMahalanobis
from continuous_gcr_loeo import empirical_p, loo_p, quantile
from recalibrate_multiknown17 import leave_one_out_validation_p
from rf_descriptor_pruning import KEEP_INDICES
from rf_physics_feature_probe import extract_all
from spectral_conformal_benchmark import deterministic
from train_causal_ablation import build_model, extract, seed_everything
from train_inner_physical_candidate import InnerPhysicalCandidateHead, configure
from train_multiknown17_baseline import make_loader


LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
SPLITS = (1, 2, 3)
SEEDS = (42, 2023, 3407)
EPS = 1e-12


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean_pairwise_cosine(projected):
    # Current learned views only: 3 for no-phase, 4 for the full candidate.
    pairs = []
    for left in range(projected.shape[1]):
        for right in range(left + 1, projected.shape[1]):
            a, b = projected[:, left], projected[:, right]
            pairs.append(np.sum(a * b, axis=1) / np.clip(
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), EPS, None))
    return np.column_stack(pairs).mean(1)


@torch.inference_mode()
def head_outputs(head, deep, physics, batch_size, device):
    logits, physical, projected = [], [], []
    head.eval()
    for start in range(0, len(deep), batch_size):
        d = torch.from_numpy(deep[start:start + batch_size].astype(np.float32)).to(device)
        p = torch.from_numpy(physics[start:start + batch_size].astype(np.float32)).to(device)
        z_p, views, _ = head.encode_physics(p)
        logit = head.classifier(torch.cat([F.normalize(d, dim=1), z_p], dim=1))
        logits.append(logit.cpu().numpy()); physical.append(z_p.cpu().numpy())
        projected.append(torch.stack(views, dim=1).cpu().numpy())
    logits = np.concatenate(logits); projected = np.concatenate(projected)
    probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, EPS, None))).sum(1) / np.log(logits.shape[1])
    return {"logits": logits, "z_p": np.concatenate(physical),
            "cosine": mean_pairwise_cosine(projected), "entropy": entropy,
            "n_views": projected.shape[1]}


def oscr(is_known, correct, score):
    nk, nu = int(is_known.sum()), int((~is_known).sum())
    fpr, ccr = [], []
    for threshold in np.r_[-np.inf, np.unique(score), np.inf]:
        accepted = score <= threshold
        fpr.append(np.sum(accepted & ~is_known) / nu)
        ccr.append(np.sum(accepted & is_known & correct) / nk)
    order = np.argsort(fpr)
    return float(np.trapz(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def metric_row(is_known, correct, score, reject):
    labels = (~is_known).astype(int)
    fpr, tpr, _ = roc_curve(labels, score); fnr = 1 - tpr; pos = np.argmin(abs(fpr - fnr))
    tp = int(np.sum(reject & ~is_known)); fp = int(np.sum(reject & is_known)); fn = int(np.sum(~reject & ~is_known))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {
        "auc": float(roc_auc_score(labels, score)), "eer": float((fpr[pos] + fnr[pos]) / 2),
        "known_acc": float(correct[is_known].mean()), "known_accept": float((~reject[is_known]).mean()),
        "known_correct_accept": float((correct[is_known] & ~reject[is_known]).mean()),
        "unknown_recall": recall, "unknown_precision": precision,
        "unknown_f1": 2 * precision * recall / max(precision + recall, EPS),
        "oscr": oscr(is_known, correct, score),
        "overall_open_accuracy": float((np.sum(correct[is_known] & ~reject[is_known]) + np.sum(reject[~is_known])) / len(score)),
    }


def selected_map(root):
    rows = json.loads((root / "results/inner_physical_arch_selection_v1/selected_candidates.json").read_text())
    return {(int(r["split"]), int(r["seed"])): r["candidate"] for r in rows}


def common_features(root, split, seed, active, train_idx, val_idx, unknown_idx,
                    base_path, head_path, candidate, batch_size, device,
                    test_circular_shift=0):
    data = root / "dataset/multiknown17_v1"
    base = torch.load(base_path, map_location="cpu"); checkpoint = torch.load(head_path, map_location="cpu")
    args = argparse.Namespace(**base["args"]); x = np.load(data / "X_pool_float32.npy", mmap_mode="r"); y = np.load(data / "Y_global.npy")
    mapping = {value: position for position, value in enumerate(active)}
    train_loader = make_loader(x, y, train_idx, mapping, args.batch_size, True)
    val_loader = make_loader(x, y, val_idx, mapping, batch_size, False)
    unknown_iq = np.asarray(x[unknown_idx])
    if test_circular_shift:
        unknown_iq = np.roll(unknown_iq, int(test_circular_shift), axis=1)
    unknown_loader = DataLoader(TensorDataset(torch.from_numpy(unknown_iq), torch.zeros(len(unknown_idx), dtype=torch.long)), batch_size=batch_size, shuffle=False)
    seed_everything(seed); backbone = build_model(args, device); backbone.load_state_dict(base["model"]); backbone.eval()
    train_e, _, train_y = extract(backbone, deterministic(train_loader, batch_size), device)
    val_e, _, val_y = extract(backbone, val_loader, device); unknown_e, _, _ = extract(backbone, unknown_loader, device)
    desc = lambda values: extract_all(values, 64)[:, KEEP_INDICES]
    raw_train, raw_val = desc(x[train_idx]), desc(x[val_idx])
    raw_unknown = desc(unknown_iq)
    standardize = lambda values: ((values - checkpoint["physics_mean"]) / checkpoint["physics_std"]).astype(np.float32)
    p_train, p_val, p_unknown = map(standardize, (raw_train, raw_val, raw_unknown))
    configure(candidate); head = InnerPhysicalCandidateHead(train_e.shape[1], len(active)).to(device); head.load_state_dict(checkpoint["head"])
    h_train = head_outputs(head, train_e, p_train, batch_size, device); h_val = head_outputs(head, val_e, p_val, batch_size, device); h_unknown = head_outputs(head, unknown_e, p_unknown, batch_size, device)
    dummy = np.zeros((len(train_e), len(active)), np.float32)
    deep = MondrianMahalanobis(train_e, dummy, train_y, val_e, val_y)
    physical = MondrianMahalanobis(h_train["z_p"], dummy, train_y, h_val["z_p"], val_y)
    scaler = StandardScaler().fit(raw_train); pca = PCA(n_components=32, whiten=True, random_state=0).fit(scaler.transform(raw_train)); tx = lambda a: pca.transform(scaler.transform(a))
    svm = SVC(C=10.0, kernel="rbf", gamma="scale", decision_function_shape="ovr").fit(tx(raw_train), train_y)
    def natives(e, raw, h, validation=False):
        if validation:
            ds = 1 - leave_one_out_validation_p(deep, e, val_y).max(1); ps = 1 - leave_one_out_validation_p(physical, h["z_p"], val_y).max(1)
        else:
            ds = 1 - deep.p_values(e).max(1); ps = 1 - physical.p_values(h["z_p"]).max(1)
        msp = 1 - F.softmax(torch.from_numpy(h["logits"]), dim=1).max(1).values.numpy()
        return {"deep": ds, "raw": -svm.decision_function(tx(raw)).max(1), "mahal": ps, "msp": msp}
    val_native = natives(val_e, raw_val, h_val, True); unknown_native = natives(unknown_e, raw_unknown, h_unknown)
    def cdfs(native, validation=False):
        out = {name: (loo_cdf(values) if validation else reference_cdf(val_native[name], values)) for name, values in native.items()}
        out["physical"] = np.maximum(out["mahal"], out["msp"])
        return out
    vc, uc = cdfs(val_native, True), cdfs(unknown_native)
    floor = 1 / (len(val_idx) + 1); fisher = lambda d: -2 * sum(np.log(np.clip(1 - d[name] + floor, EPS, 1)) for name in ("deep", "raw", "physical"))
    vf, uf = fisher(vc), fisher(uc); tau_f = quantile(vf, .90, "linear"); va = vf <= tau_f; ua = uf <= tau_f
    ref_c, ref_h = h_val["cosine"][va], h_val["entropy"][va]
    return {"val_y": val_y, "h_val": h_val, "h_unknown": h_unknown,
            "val_cdf": vc, "unknown_cdf": uc,
            "vf": vf, "uf": uf, "tau_f": tau_f, "va": va, "ua": ua,
            "val_sc": 1 - loo_p(ref_c, "lower"), "val_sh": 1 - loo_p(ref_h, "upper"),
            "unk_sc": 1 - empirical_p(ref_c, h_unknown["cosine"], "lower"), "unk_sh": 1 - empirical_p(ref_h, h_unknown["entropy"], "upper")}


def run_inner(root, worker, workers, batch_size, device):
    data = root / "dataset/multiknown17_v1"; y = np.load(data / "Y_global.npy"); role = np.load(data / "sample_role.npy"); chosen = selected_map(root)
    tasks = []
    for split in SPLITS:
        known = list(map(int, json.loads((data / f"split_{split}.json").read_text())["known_classes"]))
        for seed in SEEDS:
            for held in known: tasks.append((split, seed, held, known))
    rows = []
    for index, (split, seed, held, known) in enumerate(tasks):
        if index % workers != worker: continue
        candidate = chosen[(split, seed)]; active = [value for value in known if value != held]
        train_idx = np.flatnonzero(np.isin(y, active) & (role == 0)); val_idx = np.flatnonzero(np.isin(y, active) & (role == 1)); pseudo_idx = np.flatnonzero(y == held)
        tag = f"split{split}_seed{seed}_heldout{held}"
        features = common_features(root, split, seed, active, train_idx, val_idx, pseudo_idx,
            root / "results/continuous_gcr_loeo_v2/inner" / tag / "stage_a/best.pt",
            root / "results/inner_physical_arch_selection_v1/inner" / tag / candidate / "stage_b/best.pt",
            candidate, batch_size, device)
        is_known = np.r_[np.ones(len(val_idx), bool), np.zeros(len(pseudo_idx), bool)]
        correct = np.r_[features["h_val"]["logits"].argmax(1) == features["val_y"], np.zeros(len(pseudo_idx), bool)]
        for lam in LAMBDAS:
            val_review_pool = features["val_sh"] + lam * features["val_sc"]
            tau = quantile(val_review_pool, .95, "higher")
            val_review = np.zeros(len(features["vf"]), dtype=float)
            val_review[features["va"]] = val_review_pool
            vr = (~features["va"]) | (features["va"] & (val_review > tau))
            ur = (~features["ua"]) | (features["ua"] & (features["unk_sh"] + lam * features["unk_sc"] > tau))
            bound = 1 + lam; vs = np.where(features["va"], val_review, bound + loo_cdf(features["vf"])); us = np.where(features["ua"], features["unk_sh"] + lam * features["unk_sc"], bound + reference_cdf(features["vf"], features["uf"]))
            rows.append({"split": split, "seed": seed, "held_out_emitter": held, "candidate": candidate, "lambda": lam, "review_threshold_q95": tau,
                "n_views": features["h_val"]["n_views"], "review_known_accept": float((val_review_pool <= tau).mean()),
                "inner_oscr": oscr(is_known, correct, np.r_[vs, us]), **metric_row(is_known, correct, np.r_[vs, us], np.r_[vr, ur]), "outer_rogue_accessed": False})
        print(json.dumps({"worker": worker, "completed": [split, seed, held], "candidate": candidate}), flush=True)
    write_csv(root / "results/new_arch_gcr_matched_v1" / f"inner_worker{worker}.csv", rows)


def summarize_inner(root):
    out = root / "results/new_arch_gcr_matched_v1"; rows = []
    for path in sorted(out.glob("inner_worker*.csv")):
        rows.extend(list(csv.DictReader(path.open())))
    rows.sort(key=lambda r: (int(r["split"]), int(r["seed"]), int(r["held_out_emitter"]), float(r["lambda"])))
    write_csv(out / "inner_loeo_all_folds.csv", rows); summary = []
    for split in SPLITS:
        for seed in SEEDS:
            candidates = []
            for lam in LAMBDAS:
                rr = [r for r in rows if int(r["split"]) == split and int(r["seed"]) == seed and float(r["lambda"]) == lam]
                if len(rr) != 9: raise RuntimeError((split, seed, lam, len(rr)))
                candidates.append({"split": split, "seed": seed, "candidate": rr[0]["candidate"], "lambda": lam,
                    "mean_inner_oscr": float(np.mean([float(r["inner_oscr"]) for r in rr])), "worst_fold_oscr": float(np.min([float(r["inner_oscr"]) for r in rr])),
                    "mean_review_known_accept": float(np.mean([float(r["review_known_accept"]) for r in rr])), "selected": False})
            best = max(candidates, key=lambda r: (r["mean_inner_oscr"], r["worst_fold_oscr"], -r["lambda"])); best["selected"] = True; summary.extend(candidates)
    write_csv(out / "lambda_selection_summary.csv", summary)


def run_outer(root, batch_size, device):
    out = root / "results/new_arch_gcr_matched_v1"; chosen = selected_map(root); selection = {}
    for row in csv.DictReader((out / "lambda_selection_summary.csv").open()):
        if row["selected"].lower() == "true": selection[(int(row["split"]), int(row["seed"]))] = float(row["lambda"])
    data = root / "dataset/multiknown17_v1"; y = np.load(data / "Y_global.npy"); role = np.load(data / "sample_role.npy"); rows = []
    for split in SPLITS:
        manifest = json.loads((data / f"split_{split}.json").read_text()); known = list(map(int, manifest["known_classes"])); rogue = list(map(int, manifest["rogue_classes"]))
        for seed in SEEDS:
            candidate, lam = chosen[(split, seed)], selection[(split, seed)]
            train_idx = np.flatnonzero(np.isin(y, known) & (role == 0)); val_idx = np.flatnonzero(np.isin(y, known) & (role == 1)); known_idx = np.flatnonzero(np.isin(y, known) & (role == 2)); rogue_idx = np.flatnonzero(np.isin(y, rogue) & (role == 2))
            # Extract known and rogue jointly, preserving known first for exact accounting.
            unknown_idx = np.r_[known_idx, rogue_idx]
            features = common_features(root, split, seed, known, train_idx, val_idx, unknown_idx,
                root / f"results/multiknown17_auxce_backbone_split{split}_seed{seed}/best.pt",
                root / "results/inner_physical_arch_selection_v1/outer_selected" / f"split{split}_seed{seed}_{candidate}" / "stage_b/best.pt",
                candidate, batch_size, device)
            nk = len(known_idx); kf, rf = features["uf"][:nk], features["uf"][nk:]; ka, ra = features["ua"][:nk], features["ua"][nk:]
            kh = {key: value[:nk] for key, value in features["h_unknown"].items() if hasattr(value, "__len__")}; correct_known = kh["logits"].argmax(1) == np.asarray([known.index(int(v)) for v in y[known_idx]])
            is_known = np.r_[np.ones(nk, bool), np.zeros(len(rogue_idx), bool)]; correct = np.r_[correct_known, np.zeros(len(rogue_idx), bool)]; fscore = np.r_[kf, rf]; freject = fscore > features["tau_f"]
            rows.append({"split": split, "seed": seed, "candidate": candidate, "method": "primary_fisher", "selected_lambda": lam, "lambda": "", "review_threshold_q95": "", **metric_row(is_known, correct, fscore, freject)})
            for method, current_lambda in (("entropy_only_lambda0", 0.0), ("continuous_geometry_gcr", lam)):
                tau = quantile(features["val_sh"] + current_lambda * features["val_sc"], .95, "higher")
                review = features["unk_sh"] + current_lambda * features["unk_sc"]; reject = (~features["ua"]) | (features["ua"] & (review > tau)); bound = 1 + current_lambda
                cascade = np.where(features["ua"], review, bound + reference_cdf(features["vf"], features["uf"]))
                rows.append({"split": split, "seed": seed, "candidate": candidate, "method": method, "selected_lambda": lam, "lambda": current_lambda, "review_threshold_q95": tau, **metric_row(is_known, correct, cascade, reject)})
            print(json.dumps({"outer_completed": [split, seed], "candidate": candidate, "lambda": lam}), flush=True)
    write_csv(out / "outer_matched_results.csv", rows)
    summary = []
    for method in ("primary_fisher", "entropy_only_lambda0", "continuous_geometry_gcr"):
        rr = [r for r in rows if r["method"] == method]; item = {"method": method, "n_runs": len(rr)}
        for metric in ("auc", "eer", "known_acc", "known_accept", "known_correct_accept", "unknown_recall", "unknown_precision", "unknown_f1", "oscr", "overall_open_accuracy"):
            values = np.asarray([r[metric] for r in rr]); item[metric + "_mean"] = float(values.mean()); item[metric + "_std"] = float(values.std(ddof=1))
        summary.append(item)
    write_csv(out / "matched_summary.csv", summary)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("inner", "summarize", "outer")); parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1])); parser.add_argument("--worker", type=int, default=0); parser.add_argument("--workers", type=int, default=1); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    root = Path(args.root); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.mode == "inner": run_inner(root, args.worker, args.workers, args.batch_size, device)
    elif args.mode == "summarize": summarize_inner(root)
    else: run_outer(root, args.batch_size, device)


if __name__ == "__main__": main()
