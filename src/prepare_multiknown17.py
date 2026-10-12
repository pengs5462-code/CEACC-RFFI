"""Prepare a provenance-tracked 17-class low-sample multi-known benchmark.

This never writes to the legacy dataset.  It repurposes only the existing,
balanced 100-sample-per-class test arrays so every global class has the same
source partition and sample budget.  Each class is deterministically split into
60 train, 20 validation, and 20 test samples before its known/rogue role is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROLE_SPLIT_SEEDS = [1101, 2202, 3303, 4404, 5505]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset/multiknown17_v1"))
    args = parser.parse_args()
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    source_files = {
        "known_x": source / "9_8rogue/X_test_known.npy",
        "known_y": source / "9_8rogue/Y_test_known.npy",
        "rogue_x": source / "9_8rogue/X_test_rogue.npy",
        "rogue_y_global": source / "9_8rogue/Y_test_rogue_global.npy",
        "train_metadata": source / "train/metadata.json",
    }
    metadata = json.loads(source_files["train_metadata"].read_text())
    known_global = np.asarray(metadata["known_classes"], dtype=np.int64)
    known_x = np.load(source_files["known_x"])
    known_y_local = np.load(source_files["known_y"]).astype(int)
    rogue_x = np.load(source_files["rogue_x"])
    rogue_y = np.load(source_files["rogue_y_global"]).astype(np.int64)
    known_y = known_global[known_y_local]
    x = np.concatenate([known_x, rogue_x]).astype(np.float32)
    y = np.concatenate([known_y, rogue_y]).astype(np.int64)
    classes = np.asarray(sorted(np.unique(y)), dtype=np.int64)
    if len(classes) != 17:
        raise ValueError(f"expected 17 classes, got {classes.tolist()}")
    counts = {int(k): int((y == k).sum()) for k in classes}
    if set(counts.values()) != {100}:
        raise ValueError(f"expected exactly 100 samples per class, got {counts}")

    role = np.full(len(y), -1, dtype=np.int8)  # 0 train, 1 validation, 2 test
    class_local_index = np.full(len(y), -1, dtype=np.int16)
    for class_id in classes:
        index = np.flatnonzero(y == class_id)
        permutation = np.random.default_rng(91_000 + int(class_id)).permutation(index)
        role[permutation[:60]] = 0
        role[permutation[60:80]] = 1
        role[permutation[80:]] = 2
        class_local_index[permutation] = np.arange(100, dtype=np.int16)
    if (role < 0).any() or (class_local_index < 0).any():
        raise RuntimeError("incomplete sample-role assignment")

    sample_hashes = [
        hashlib.sha256(np.ascontiguousarray(sample).tobytes()).hexdigest()
        for sample in x
    ]
    duplicate_hashes = len(sample_hashes) - len(set(sample_hashes))
    if duplicate_hashes:
        raise ValueError(f"found {duplicate_hashes} exact duplicate sample hashes")
    np.save(output / "X_pool_float32.npy", x)
    np.save(output / "Y_global.npy", y)
    np.save(output / "sample_role.npy", role)
    np.save(output / "class_local_index.npy", class_local_index)

    manifests = []
    for split_index, seed in enumerate(ROLE_SPLIT_SEEDS, start=1):
        rng = np.random.default_rng(seed)
        known_classes = sorted(int(v) for v in rng.choice(classes, size=9, replace=False))
        rogue_classes = sorted(int(v) for v in classes if int(v) not in known_classes)
        manifest = {
            "split_index": split_index,
            "role_split_seed": seed,
            "known_classes": known_classes,
            "rogue_classes": rogue_classes,
            "known_train_samples": 9 * 60,
            "known_validation_samples": 9 * 20,
            "known_test_samples": 9 * 20,
            "rogue_test_samples": 8 * 20,
            "status": "development" if split_index <= 3 else "reserved_untouched",
        }
        (output / f"split_{split_index}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        manifests.append(manifest)

    provenance = {
        "name": "multiknown17_v1",
        "purpose": "supplementary low-sample multi-known/rogue stability benchmark",
        "limitations": [
            "not a substitute for the unavailable original 30-class source dataset",
            "repurposes legacy balanced test samples into new disjoint 60/20/20 roles",
            "only 17 global classes are available and every class has 100 samples",
        ],
        "source_is_read_only": True,
        "source_files_sha256": {
            name: file_sha256(path) for name, path in source_files.items()
        },
        "pool_shape": list(x.shape),
        "pool_dtype": str(x.dtype),
        "classes": classes.tolist(),
        "per_class_counts": counts,
        "within_class_partition": {"train": 60, "validation": 20, "test": 20},
        "within_class_partition_seed_formula": "91000 + global_class_id",
        "exact_duplicate_sample_hashes": duplicate_hashes,
        "role_manifests": manifests,
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
