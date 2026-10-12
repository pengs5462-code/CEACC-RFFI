"""Frozen structural pruning for the 339-D RF descriptor.

The sparse-bin decision was made once from the union of development splits 1--3
known-training samples only (900 samples).  No validation, known-test, reserved,
or rogue sample participates in the mask.  The default extractor remains 339-D;
diagnostic runners opt into this frozen 304-D view explicitly.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from rf_physics_feature_probe import FEATURE_GROUPS, FEATURE_NAMES


SOURCE_DIMENSION = 339
SPARSE_THRESHOLD = 0.01
SPARSE_REFERENCE = "development splits 1-3 union, known-train role only, n=900"

# Exact copies or values algebraically recoverable from retained columns.
STRUCTURAL_REMOVALS = {
    2: "log RMS recoverable from retained I/Q population standard deviations",
    3: "exact duplicate of dc.real",
    7: "exact duplicate of dc.imag",
    11: "global amplitude mean is the mean of the 48 retained block means",
    12: "amplitude std is recoverable from RMS normalization and amplitude mean",
    16: "log I/Q variance ratio recoverable from retained I/Q standard deviations",
    26: "magnitude recoverable from circular moment real/imag",
    29: "magnitude recoverable from circular moment real/imag",
    32: "magnitude recoverable from circular moment real/imag",
    35: "magnitude recoverable from circular moment real/imag",
    36: "second complex-moment real part recoverable from normalized I/Q variances",
    37: "second complex-moment imag part recoverable from retained I/Q cross moment and DC",
    38: "second complex-moment magnitude recoverable from real/imag",
    41: "magnitude recoverable from complex moment real/imag",
    44: "magnitude recoverable from complex moment real/imag",
    47: "magnitude recoverable from complex moment real/imag",
    50: "magnitude recoverable from complex moment real/imag",
    53: "magnitude recoverable from complex autocorrelation real/imag",
    57: "magnitude recoverable from complex autocorrelation real/imag",
    61: "magnitude recoverable from complex autocorrelation real/imag",
    65: "magnitude recoverable from complex autocorrelation real/imag",
    69: "magnitude recoverable from complex autocorrelation real/imag",
    73: "magnitude recoverable from complex autocorrelation real/imag",
    77: "magnitude recoverable from complex autocorrelation real/imag",
    81: "magnitude recoverable from complex autocorrelation real/imag",
    146: "phase histogram simplex closure; final bin recoverable from the other 31",
    242: "per-sample centered spectrum; final band recoverable from the other 95",
}

# Nonzero rate was below 1% in the frozen known-training reference above.
SPARSE_REMOVALS = {
    107: 4 / 900,
    108: 4 / 900,
    109: 3 / 900,
    110: 2 / 900,
    111: 2 / 900,
    112: 2 / 900,
    113: 1 / 900,
    114: 1 / 900,
}

REMOVED_INDICES = np.asarray(sorted(set(STRUCTURAL_REMOVALS) | set(SPARSE_REMOVALS)), dtype=np.int64)
KEEP_INDICES = np.asarray(sorted(set(range(SOURCE_DIMENSION)) - set(REMOVED_INDICES)), dtype=np.int64)
PRUNED_DIMENSION = len(KEEP_INDICES)


def pruned_groups() -> tuple[np.ndarray, ...]:
    """Return physical group indices remapped into the pruned column space."""
    old_to_new = {int(old): new for new, old in enumerate(KEEP_INDICES)}
    return tuple(
        np.asarray([old_to_new[int(old)] for old in old_group if int(old) in old_to_new], dtype=np.int64)
        for old_group in FEATURE_GROUPS.values()
    )


PRUNED_GROUPS = pruned_groups()
PRUNED_GROUP_SIZES = tuple(map(len, PRUNED_GROUPS))


def select(features: np.ndarray) -> np.ndarray:
    if features.shape[1] != SOURCE_DIMENSION:
        raise ValueError(f"expected {SOURCE_DIMENSION} RF columns, received {features.shape[1]}")
    return features[:, KEEP_INDICES]


def manifest() -> dict:
    payload = {
        "method": "rf_descriptor_structural_pruning_v1",
        "source_dimension": SOURCE_DIMENSION,
        "pruned_dimension": PRUNED_DIMENSION,
        "group_sizes": PRUNED_GROUP_SIZES,
        "sparse_threshold": SPARSE_THRESHOLD,
        "sparse_reference": SPARSE_REFERENCE,
        "keep_indices": KEEP_INDICES.tolist(),
        "removed": [
            {
                "index": int(index),
                "name": FEATURE_NAMES[int(index)],
                "reason": STRUCTURAL_REMOVALS.get(int(index), "known-train nonzero rate below threshold"),
                "known_train_nonzero_rate": SPARSE_REMOVALS.get(int(index)),
            }
            for index in REMOVED_INDICES
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


if PRUNED_DIMENSION != 304 or PRUNED_GROUP_SIZES != (42, 71, 95, 96):
    raise AssertionError((PRUNED_DIMENSION, PRUNED_GROUP_SIZES))


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2))
