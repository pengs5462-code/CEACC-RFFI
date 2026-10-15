"""BLi2 reproduction on the registered multiknown17_v1 ADS-B protocol.

Representation and BL-Net follow Eqs. (2)-(16) and p=100, k=2, q=50,
b=2. The runner records a known-only diagnostic plus the paper-native
test-batch K=2 clustering separately. The paper does not disclose code or
numeric OSR thresholds and explicitly mentions surrogate unknowns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

EPS = 1e-12


def auc_rank(labels, scores):
    labels = np.asarray(labels, dtype=np.int64); scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort"); sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64); start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]: end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0; start = end
    positives = int(labels.sum()); negatives = len(labels) - positives
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) /
                 (positives * negatives))


def roc_points(known, rogue):
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[known, rogue]))[::-1], -np.inf]
    return (np.asarray([np.mean(known >= t) for t in thresholds]),
            np.asarray([np.mean(rogue >= t) for t in thresholds]), thresholds)


def eer(known, rogue):
    fpr, tpr, _ = roc_points(known, rogue); i = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    return float(0.5 * (fpr[i] + 1.0 - tpr[i]))


def oscr(known_scores, rogue_scores, known_correct):
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[known_scores, rogue_scores])), -np.inf]
    fpr = np.asarray([np.mean(rogue_scores <= t) for t in thresholds])
    ccr = np.asarray([np.mean(known_correct & (known_scores <= t)) for t in thresholds])
    order = np.argsort(fpr); return float(np.trapz(ccr[order], fpr[order]))


def metric_row(validation_score, known_score, rogue_score, known_correct, reject_rate,
               threshold_override=None):
    threshold = (float(np.quantile(validation_score, 1.0 - reject_rate))
                 if threshold_override is None else float(threshold_override))
    known_reject = known_score > threshold; rogue_reject = rogue_score > threshold
    tp = int(rogue_reject.sum()); fp = int(known_reject.sum()); fn = int((~rogue_reject).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    fpr, tpr, _ = roc_points(known_score, rogue_score); reaches = np.flatnonzero(tpr >= 0.95)
    labels = np.r_[np.zeros(len(known_score), int), np.ones(len(rogue_score), int)]
    return {"auc": auc_rank(labels, np.r_[known_score, rogue_score]),
            "eer": eer(known_score, rogue_score),
            "fpr95": float(np.min(fpr[reaches])) if len(reaches) else 1.0,
            "known_closed_acc": float(known_correct.mean()),
            "known_accept_rate": float((~known_reject).mean()),
            "unknown_recall": float(recall),
            "unknown_f1": float(2 * precision * recall / max(precision + recall, EPS)),
            "oscr": oscr(known_score, rogue_score, known_correct),
            "tau_known_only": threshold}


def build_iar_cache(x, cache_path, max_lag, bins, batch_size=64):
    """Eq. (4): P=max_lag phase trajectories, circularly resampled to fixed frames."""
    expected = (len(x), 2 * max_lag * bins)
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == expected: return cached, 0.0
        raise RuntimeError(f"Unexpected IAR cache shape {cached.shape}, expected {expected}")
    start = time.time(); cache_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=expected)
    for begin in range(0, len(x), batch_size):
        end = min(begin + batch_size, len(x)); raw = np.asarray(x[begin:end], dtype=np.float32)
        z = raw[..., 0].astype(np.complex64) + 1j * raw[..., 1].astype(np.complex64)
        features = []
        for lag in range(1, max_lag + 1):
            corr = z[:, :-lag] * np.conj(z[:, lag:])
            corr /= np.maximum(np.abs(corr), 1e-8)
            usable = (corr.shape[1] // bins) * bins
            framed = corr[:, :usable].reshape(len(z), bins, usable // bins).mean(axis=2)
            features.extend((framed.real, framed.imag))
        output[begin:end] = np.concatenate(features, axis=1).astype(np.float32)
    output.flush(); return np.load(cache_path, mmap_mode="r"), time.time() - start


class BroadLearningSystem:
    def __init__(self, input_dim, classes, seed, p=100, k=2, q=50, b=2, ridge=1e-3):
        self.classes = classes; self.ridge = ridge; self.seed = seed
        self.p = p; self.k = k; self.q = q; self.b = b
        self.feature_beta = []; self.feature_min = []; self.feature_range = []
        self.enhancement_weight = None; self.enhancement_shrink = None
        self.output_weight = None; self.input_dim = input_dim

    @staticmethod
    def preprocess(value):
        """Canonical BLS preprocessing.scale(..., axis=1)."""
        mean = value.mean(axis=1, keepdims=True); std = value.std(axis=1, keepdims=True)
        std[std < 1e-12] = 1.0
        return (value - mean) / std

    @staticmethod
    def sparse_bls(a, target, lam=1e-3, iterations=50):
        aa = a.T @ a; eye = np.eye(a.shape[1]); l1 = np.linalg.inv(aa + eye)
        l2 = l1 @ a.T @ target; ok = np.zeros_like(l2); uk = np.zeros_like(l2)
        for _ in range(iterations):
            ck = l2 + l1 @ (ok - uk)
            ok = np.maximum(ck + uk - lam, 0.0) - np.maximum(-ck - uk - lam, 0.0)
            uk += ck - ok
        return ok

    def _feature_train(self, value):
        value = self.preprocess(value); value_b = np.c_[value, 0.1 * np.ones(len(value))]
        output = np.empty((len(value), self.p * self.k), dtype=np.float64)
        for i in range(self.p):
            rng = np.random.RandomState(self.seed + i)
            random_weight = 2.0 * rng.randn(value_b.shape[1], self.k) - 1.0
            projected = value_b @ random_weight
            lo = projected.min(axis=0); span = projected.max(axis=0) - lo; span[span < 1e-12] = 1.0
            projected01 = (projected - lo) / span
            beta = self.sparse_bls(projected01, value_b).T
            mapped = value_b @ beta
            mapped_lo = mapped.min(axis=0); mapped_span = mapped.max(axis=0) - mapped_lo
            mapped_span[mapped_span < 1e-12] = 1.0
            output[:, i*self.k:(i+1)*self.k] = (mapped - mapped_lo) / mapped_span
            self.feature_beta.append(beta); self.feature_min.append(mapped_lo); self.feature_range.append(mapped_span)
        return output

    def _feature_test(self, value):
        value = self.preprocess(value); value_b = np.c_[value, 0.1 * np.ones(len(value))]
        output = np.empty((len(value), self.p * self.k), dtype=np.float64)
        for i, beta in enumerate(self.feature_beta):
            mapped = value_b @ beta
            output[:, i*self.k:(i+1)*self.k] = (mapped - self.feature_min[i]) / self.feature_range[i]
        return output

    def _make_hidden(self, feature):
        feature_b = np.c_[feature, 0.1 * np.ones(len(feature))]
        temp = feature_b @ self.enhancement_weight
        return np.c_[feature, np.tanh(temp * self.enhancement_shrink)]

    def fit(self, value, target):
        feature = self._feature_train(value); rng = np.random.RandomState(self.seed + 10000)
        raw = 2.0 * rng.randn(feature.shape[1] + 1, self.q * self.b) - 1.0
        self.enhancement_weight = np.linalg.qr(raw)[0][:, :self.q*self.b]
        temp = np.c_[feature, 0.1 * np.ones(len(feature))] @ self.enhancement_weight
        self.enhancement_shrink = 0.8 / max(float(np.max(np.abs(temp))), 1e-12)
        hidden = self._make_hidden(feature); one_hot = np.eye(self.classes)[target]
        gram = hidden.T @ hidden; gram.flat[::len(gram) + 1] += self.ridge
        self.output_weight = np.linalg.solve(gram, hidden.T @ one_hot); return self

    def logits(self, value): return self._make_hidden(self._feature_test(value)) @ self.output_weight

    @property
    def mapping_parameters(self):
        return int(sum(x.size for x in self.feature_beta) + self.enhancement_weight.size + self.output_weight.size)


def sigmoid(value): return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def ecdf_score(reference, value):
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(reference, value, side="right") / max(len(reference), 1)


def known_only_scores(train_logits, train_y, validation_logits, *targets):
    prototypes = np.stack([train_logits[train_y == c].mean(0) for c in range(train_logits.shape[1])])
    def cues(logits):
        distance = np.min(np.linalg.norm(logits[:, None] - prototypes[None], axis=2), axis=1)
        low_confidence = 1.0 - sigmoid(logits).max(axis=1)
        return distance, low_confidence
    vd, vc = cues(validation_logits); scores = []
    for logits in targets:
        distance, confidence = cues(logits)
        scores.append(np.maximum(ecdf_score(vd, distance), ecdf_score(vc, confidence)))
    validation_score = np.maximum(ecdf_score(vd, vd), ecdf_score(vc, vc))
    return (validation_score, *scores), prototypes


def transductive_scores(validation_logits, known_logits, rogue_logits, seed):
    mixed = np.concatenate((known_logits, rogue_logits)); km = KMeans(
        n_clusters=2, init="k-means++", n_init=10, random_state=seed).fit(mixed)
    known_cluster = int(np.argmin(np.linalg.norm(km.cluster_centers_ - validation_logits.mean(0), axis=1)))
    unknown_cluster = 1 - known_cluster
    def cues(logits):
        d_known = np.linalg.norm(logits - km.cluster_centers_[known_cluster], axis=1)
        d_unknown = np.linalg.norm(logits - km.cluster_centers_[unknown_cluster], axis=1)
        return d_known - d_unknown, 1.0 - sigmoid(logits).max(axis=1)
    vm, _ = cues(validation_logits); margin, _ = cues(mixed)
    # The K-means decision boundary is exactly margin=0: nearer to the
    # unknown-aligned centroid is rejected, as specified by the paper's stage 1.
    return vm, margin[:len(known_logits)], margin[len(known_logits):]


def fingerprint(indices, y):
    digest = hashlib.sha256(); digest.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(y[indices], dtype=np.int64).tobytes())
    return {"sha256": digest.hexdigest(), "samples": int(len(indices))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", type=int, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--p", type=int, default=100); parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--q", type=int, default=50); parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-3); parser.add_argument("--iar-lag", type=int, default=10)
    parser.add_argument("--iar-bins", type=int, default=240)
    args = parser.parse_args(); os.environ.setdefault("OMP_NUM_THREADS", "1"); start = time.time()
    root = Path(args.dataset_root); manifest = json.loads((root / f"split_{args.split}.json").read_text())
    known = list(map(int, manifest["known_classes"])); rogue = list(map(int, manifest["rogue_classes"]))
    mapping = {label: i for i, label in enumerate(known)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r"); y = np.load(root / "Y_global.npy")
    role = np.load(root / "sample_role.npy"); known_mask = np.isin(y, known)
    indices = {"train": np.flatnonzero(known_mask & (role == 0)),
               "validation": np.flatnonzero(known_mask & (role == 1)),
               "known": np.flatnonzero(known_mask & (role == 2)),
               "rogue": np.flatnonzero(np.isin(y, rogue) & (role == 2))}
    cache = root / f"IAR_DCT_phase_delays1-{args.iar_lag}_bins{args.iar_bins}_circle_float32.npy"
    representation, cache_seconds = build_iar_cache(x, cache, args.iar_lag, args.iar_bins)
    train_x = np.asarray(representation[indices["train"]], dtype=np.float64)
    transform = lambda idx: np.asarray(representation[idx], dtype=np.float64)
    train_y = np.asarray([mapping[int(v)] for v in y[indices["train"]]], dtype=np.int64)
    validation_y = np.asarray([mapping[int(v)] for v in y[indices["validation"]]], dtype=np.int64)
    known_y = np.asarray([mapping[int(v)] for v in y[indices["known"]]], dtype=np.int64)
    model = BroadLearningSystem(train_x.shape[1], len(known), args.seed,
                                args.p, args.k, args.q, args.b, args.ridge)
    model.fit(train_x, train_y)
    train_logits = model.logits(train_x)
    validation_logits = model.logits(transform(indices["validation"]))
    known_logits = model.logits(transform(indices["known"])); rogue_logits = model.logits(transform(indices["rogue"]))
    (validation_score, known_score, rogue_score), _ = known_only_scores(
        train_logits, train_y, validation_logits, known_logits, rogue_logits)
    native_validation, trans_known, trans_rogue = transductive_scores(
        validation_logits, known_logits, rogue_logits, args.seed)
    known_prediction = known_logits.argmax(1); known_correct = known_prediction == known_y
    metrics = metric_row(validation_score, known_score, rogue_score, known_correct, args.known_reject_rate)
    native_metrics = metric_row(native_validation, trans_known, trans_rogue, known_correct,
                                args.known_reject_rate, threshold_override=0.0)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_sample_scores.npz", validation_score=native_validation,
        validation_label=validation_y, known_score=trans_known, known_label=known_y,
        known_prediction=known_prediction, rogue_score=trans_rogue, rogue_label=y[indices["rogue"]],
        diagnostic_inductive_validation_score=validation_score,
        diagnostic_inductive_known_score=known_score,
        diagnostic_inductive_rogue_score=rogue_score)
    result = {"method": "BLi2", "implementation": "paper_faithful_standard_bls_v5",
        "official_code_available": False, "paper": {"title": "A Lightweight Open-Set Specific Emitter Identification Based on Broad Learning",
        "doi": "10.1109/JIOT.2025.3637514", "venue": "IEEE Internet of Things Journal 13(4):6158-6171 (2026)"},
        "split": args.split, "seed": args.seed, "known_classes": known, "rogue_classes": rogue,
        "runtime_seconds": time.time() - start, "iar_cache_build_seconds": cache_seconds,
        "metrics": native_metrics, "diagnostic_known_only_metrics": metrics,
        "split_fingerprints": {name: fingerprint(value, y) for name, value in indices.items()},
        "protocol": {"representation": "phase of differential constellation trajectory",
        "iar_formula": "angle(s[t]*conj(s[t+lambda]))", "iar_delays": list(range(1,args.iar_lag+1)),
        "iar_fixed_frame_bins": args.iar_bins, "phase_encoding": "equivalent unit-circle real/imag",
        "p": args.p, "k": args.k, "q": args.q, "b": args.b, "hidden_dimension": args.p*args.k+args.q*args.b,
        "mapping_parameters": model.mapping_parameters, "ridge": args.ridge,
        "bls_mapping": "canonical sparse feature windows, min-max mapping, orthogonal enhancement, tansig",
        "reported_variant": "paper-native K=2 test-logit clustering and closed-set label mapping",
        "operating_threshold": "nearest-centroid K-means boundary (signed distance margin = 0)",
        "paper_native_variant": "K=2 k-means over the mixed target test batch",
        "target_rogue_used_by_paper_native_variant": True,
        "paper_states_surrogate_unknown_samples_are_used": True,
        "numeric_osr_thresholds_disclosed_in_paper": False},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "result.json").write_text(json.dumps(result, indent=2)); print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__": main()
