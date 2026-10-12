"""Cheap development-split probe of interpretable RF fingerprint statistics.

This is a diagnostic, not a claimed final method.  It tests whether information
missing from the learned embedding is present in amplitude, phase-increment,
autocorrelation, and FFT-envelope statistics.  Only development class splits
1--3 are accepted; reserved splits are rejected by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EPS = 1e-8


def feature_names() -> list[str]:
    """Return the exact 0-based column names emitted by :func:`extract_batch`."""
    names = ["dc.real", "dc.imag", "log_rms"]
    for base in ("raw_i", "raw_q", "amp"):
        names += [f"{base}.{name}" for name in ("mean", "std", "skew", "raw_kurtosis")]
    names += ["iq_raw_cross_over_stdprod", "log_var_i_over_var_q"]
    names += [f"amp.quantile_{int(100 * q):02d}" for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)]
    for order in range(1, 5):
        names += [f"phase_increment.circular_{order}.{part}" for part in ("real", "imag", "magnitude")]
    for order in range(2, 7):
        names += [f"z.moment_{order}.{part}" for part in ("real", "imag", "magnitude")]
    for lag in (1, 2, 4, 8, 16, 32, 64, 128):
        names += [f"z.autocorrelation_lag_{lag}.{part}" for part in ("real", "imag", "magnitude")]
        names.append(f"amp.autocorrelation_lag_{lag}")
    names += [f"amp.histogram_{index:02d}" for index in range(32)]
    names += [f"phase_increment.histogram_{index:02d}" for index in range(32)]
    names += [f"spectrum.band_{index:02d}" for index in range(96)]
    names += [f"temporal.block_{index:02d}.mean" for index in range(48)]
    names += [f"temporal.block_{index:02d}.std" for index in range(48)]
    if len(names) != 339:
        raise AssertionError(f"feature metadata has {len(names)} columns, expected 339")
    return names


FEATURE_NAMES = tuple(feature_names())
FEATURE_GROUPS = {
    "amplitude_statistics": np.r_[0:24, 83:115],
    "phase_correlation": np.r_[24:83, 115:147],
    "spectral": np.arange(147, 243),
    "temporal": np.arange(243, 339),
}


def moments(x: np.ndarray) -> list[np.ndarray]:
    mean = x.mean(1)
    centered = x - mean[:, None]
    std = np.sqrt(np.mean(centered * centered, axis=1) + EPS)
    skew = np.mean((centered / std[:, None]) ** 3, axis=1)
    kurt = np.mean((centered / std[:, None]) ** 4, axis=1)
    return [mean, std, skew, kurt]


def fixed_histogram(x: np.ndarray, low: float, high: float, bins: int) -> np.ndarray:
    clipped = np.clip((x - low) / (high - low), 0.0, 1.0 - 1e-7)
    index = np.floor(clipped * bins).astype(np.int64)
    result = np.zeros((len(x), bins), dtype=np.float32)
    for row in range(len(x)):
        result[row] = np.bincount(index[row], minlength=bins) / x.shape[1]
    return result


def relative_block_statistic(
    x: np.ndarray, blocks: int, statistic: str
) -> np.ndarray:
    """Summarize equal relative intervals without assuming a fixed sample count.

    At the legacy length of 4800 this is exactly equivalent to the former
    ``reshape(...).mean/std`` implementation.  For other lengths, including
    LoRa's 8192 samples, the intervals differ by at most one sample.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D [batch,time] array, got {x.shape}")
    if x.shape[1] < blocks:
        raise ValueError(
            f"sequence length {x.shape[1]} is shorter than {blocks} relative blocks"
        )
    pieces = np.array_split(x, blocks, axis=1)
    if statistic == "mean":
        return np.stack([piece.mean(axis=1) for piece in pieces], axis=1)
    if statistic == "std":
        return np.stack([piece.std(axis=1) for piece in pieces], axis=1)
    raise ValueError(f"unsupported statistic {statistic!r}")


def extract_batch(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[-1] != 2:
        raise ValueError(f"IQ must have shape [N,T,2], got {iq.shape}")
    if iq.shape[1] <= 128:
        raise ValueError("IQ sequence must contain more than 128 samples")
    raw_i, raw_q = iq[:, :, 0], iq[:, :, 1]
    z_raw = raw_i + 1j * raw_q
    dc = z_raw.mean(1)
    centered = z_raw - dc[:, None]
    rms = np.sqrt(np.mean(np.abs(centered) ** 2, axis=1) + EPS)
    z = centered / rms[:, None]
    amp = np.abs(z)
    phase_increment = np.angle(z[:, 1:] * np.conj(z[:, :-1]))

    columns: list[np.ndarray] = []
    columns += [dc.real, dc.imag, np.log(rms + EPS)]
    columns += moments(raw_i)
    columns += moments(raw_q)
    columns += moments(amp)
    columns += [
        np.mean(raw_i * raw_q, axis=1)
        / (np.std(raw_i, axis=1) * np.std(raw_q, axis=1) + EPS),
        np.log((np.var(raw_i, axis=1) + EPS) / (np.var(raw_q, axis=1) + EPS)),
    ]
    for quantile in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        columns.append(np.quantile(amp, quantile, axis=1))

    for order in range(1, 5):
        circular = np.mean(np.exp(1j * order * phase_increment), axis=1)
        columns += [circular.real, circular.imag, np.abs(circular)]
    for order in range(2, 7):
        cm = np.mean(z**order, axis=1)
        columns += [cm.real, cm.imag, np.abs(cm)]
    for lag in (1, 2, 4, 8, 16, 32, 64, 128):
        corr = np.mean(z[:, lag:] * np.conj(z[:, :-lag]), axis=1)
        columns += [corr.real, corr.imag, np.abs(corr)]
        amplitude_corr = np.mean(amp[:, lag:] * amp[:, :-lag], axis=1)
        columns.append(amplitude_corr)

    scalar = np.stack(columns, axis=1).astype(np.float32)
    amp_hist = fixed_histogram(amp, 0.0, 3.0, 32)
    phase_hist = fixed_histogram(phase_increment, -np.pi, np.pi, 32)

    spectrum = np.fft.fftshift(np.fft.fft(z, axis=1), axes=1)
    log_power = np.log1p(np.abs(spectrum) ** 2).astype(np.float32)
    # Preserve the 96-band normalized-frequency envelope while defining each
    # band by its relative position.  This remains bit-for-bit equivalent at
    # T=4800 and also supports lengths not divisible by 96, such as T=8192.
    spectral_envelope = relative_block_statistic(log_power, 96, "mean")
    spectral_envelope = (
        spectral_envelope - spectral_envelope.mean(1, keepdims=True)
    ) / (spectral_envelope.std(1, keepdims=True) + EPS)

    # Coarse time-envelope summaries retain burst structure without using raw
    # sample coordinates as thousands of independent features.
    envelope_mean = relative_block_statistic(amp, 48, "mean")
    envelope_std = relative_block_statistic(amp, 48, "std")
    return np.concatenate(
        [scalar, amp_hist, phase_hist, spectral_envelope, envelope_mean, envelope_std],
        axis=1,
    ).astype(np.float32)


def extract_all(x: np.ndarray, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [extract_batch(x[start : start + batch_size]) for start in range(0, len(x), batch_size)],
        axis=0,
    )


def main() -> None:
    # Keep descriptor extraction importable in lightweight preprocessing and
    # compatibility-audit environments.  These dependencies are only required
    # by the optional benchmark CLI below.
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from conformal_benchmark import MondrianMahalanobis
    from recalibrate_multiknown17 import leave_one_out_validation_p, metric_row
    from score_benchmark import metric_row as scalar_metric_row

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    args = parser.parse_args()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    root = Path(args.dataset_root)
    manifests = {}
    allowed_classes: set[int] = set()
    for split in args.splits:
        manifest = json.loads((root / f"split_{split}.json").read_text())
        if manifest["status"] != "development":
            raise ValueError(f"split {split} is not a development split")
        manifests[split] = manifest
        allowed_classes.update(map(int, manifest["known_classes"]))
        allowed_classes.update(map(int, manifest["rogue_classes"]))

    x = np.load(root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(root / "Y_global.npy")
    role = np.load(root / "sample_role.npy")
    selected = np.flatnonzero(np.isin(y, sorted(allowed_classes)))
    feature = extract_all(x[selected], args.batch_size)
    index_to_row = {int(index): row for row, index in enumerate(selected)}

    results = []
    for split in args.splits:
        manifest = manifests[split]
        known_classes = list(map(int, manifest["known_classes"]))
        rogue_classes = list(map(int, manifest["rogue_classes"]))
        mapping = {class_id: position for position, class_id in enumerate(known_classes)}
        masks = {
            "train": np.isin(y, known_classes) & (role == 0),
            "validation": np.isin(y, known_classes) & (role == 1),
            "known_test": np.isin(y, known_classes) & (role == 2),
            "rogue_test": np.isin(y, rogue_classes) & (role == 2),
        }

        def take(name: str):
            indices = np.flatnonzero(masks[name])
            rows = np.asarray([index_to_row[int(index)] for index in indices])
            labels = y[indices].astype(int)
            return feature[rows], labels

        train_x, train_global = take("train")
        val_x, val_global = take("validation")
        known_x, known_global = take("known_test")
        rogue_x, rogue_global = take("rogue_test")
        train_y = np.asarray([mapping[int(label)] for label in train_global])
        val_y = np.asarray([mapping[int(label)] for label in val_global])
        known_y = np.asarray([mapping[int(label)] for label in known_global])

        scaler = StandardScaler().fit(train_x)
        train_s = scaler.transform(train_x)
        val_s = scaler.transform(val_x)
        known_s = scaler.transform(known_x)
        rogue_s = scaler.transform(rogue_x)
        pca = PCA(n_components=args.pca_dim, whiten=True, random_state=0).fit(train_s)
        train_e = pca.transform(train_s)
        val_e = pca.transform(val_s)
        known_e = pca.transform(known_s)
        rogue_e = pca.transform(rogue_s)

        svm = SVC(C=args.svm_c, kernel="rbf", gamma="scale", decision_function_shape="ovr")
        svm.fit(train_e, train_y)
        val_decision = svm.decision_function(val_e)
        known_decision = svm.decision_function(known_e)
        rogue_decision = svm.decision_function(rogue_e)
        svm_row = scalar_metric_row(
            "negative_max_rbf_svm_decision",
            -val_decision.max(1),
            -known_decision.max(1),
            -rogue_decision.max(1),
            svm.predict(known_e) == known_y,
            args.known_reject_rate,
        )
        svm_row["known_closed_acc"] = float((svm.predict(known_e) == known_y).mean())
        svm_row["known_validation_acc"] = float((svm.predict(val_e) == val_y).mean())

        dummy_logits = np.zeros((len(train_e), len(known_classes)), dtype=np.float32)
        conformal = MondrianMahalanobis(train_e, dummy_logits, train_y, val_e, val_y)
        validation_p_loo = leave_one_out_validation_p(conformal, val_e, val_y)
        threshold = float(
            np.quantile(1.0 - validation_p_loo.max(1), 1.0 - args.known_reject_rate)
        )
        conformal_row = metric_row(
            conformal.p_values(known_e),
            conformal.p_values(rogue_e),
            known_y,
            threshold,
        )
        results.append(
            {
                "split": split,
                "known_classes": known_classes,
                "rogue_classes": rogue_classes,
                "feature_dimension": int(feature.shape[1]),
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "svm": svm_row,
                "mondrian": conformal_row,
                "rogue_global_labels": rogue_global.tolist(),
            }
        )

    output = {
        "method": "fixed_interpretable_rf_statistics_probe_v1",
        "runner_sha256": runner_sha256,
        "protocol": {
            "purpose": "diagnose missing RF information; not a final claimed method",
            "development_splits_only": list(args.splits),
            "reserved_splits_accessed": False,
            "feature_definition": "fixed before development rogue evaluation",
            "scaler_pca_classifier_geometry_fit": "known training only per split",
            "operating_threshold": "known validation only per split",
            "target_rogue_used_for_fit_or_threshold": False,
            "pca_dim": args.pca_dim,
            "svm_c": args.svm_c,
        },
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
