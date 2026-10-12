#!/usr/bin/env python3
"""Freeze the official LoRa known30/unknown15 experiment protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


STATUS = "FROZEN_BEFORE_OPEN_SET_TEST"
ROLE_TRAIN = 0
ROLE_VALIDATION = 1
ROLE_TEST = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def link(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"existing link {target} points to {target.resolve()}")
        return
    if target.exists():
        raise FileExistsError(target)
    target.symlink_to(os.path.relpath(source, start=target.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    balanced = args.balanced_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    x_path = balanced / "X_pool_float32.npy"
    y_path = balanced / "Y_global.npy"
    packet_path = balanced / "source_packet_index.npy"
    adapter_manifest_path = balanced / "manifest.json"
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    packet = np.load(packet_path)
    adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    if adapter_manifest["status"] != "ADAPTER_FROZEN_BEFORE_OPEN_SET_TEST":
        raise ValueError("adapter is not frozen")
    if x.shape != (9000, 8192, 2) or sorted(np.unique(y).tolist()) != list(range(45)):
        raise ValueError("unexpected balanced45 adapter contents")

    known_classes = list(range(30))
    rogue_classes = list(range(30, 45))
    role = np.full(len(y), ROLE_TEST, dtype=np.int8)
    for class_id in known_classes:
        indices = np.flatnonzero(y == class_id)
        ordered = indices[np.argsort(packet[indices], kind="stable")]
        if packet[ordered].tolist() != list(range(200)):
            raise ValueError(f"class {class_id}: packet order is not 0..199")
        role[ordered[:120]] = ROLE_TRAIN
        role[ordered[120:160]] = ROLE_VALIDATION
        role[ordered[160:]] = ROLE_TEST

    role_path = output / "sample_role.npy"
    np.save(role_path, role)
    link(x_path, output / x_path.name)
    link(y_path, output / y_path.name)
    link(packet_path, output / packet_path.name)
    link(balanced / "source_file_index.npy", output / "source_file_index.npy")
    link(adapter_manifest_path, output / "adapter_manifest.json")

    masks = {
        "known_train_samples": np.isin(y, known_classes) & (role == ROLE_TRAIN),
        "known_validation_samples": np.isin(y, known_classes)
        & (role == ROLE_VALIDATION),
        "known_test_samples": np.isin(y, known_classes) & (role == ROLE_TEST),
        "rogue_test_samples": np.isin(y, rogue_classes) & (role == ROLE_TEST),
    }
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    expected = {
        "known_train_samples": 3600,
        "known_validation_samples": 1200,
        "known_test_samples": 1200,
        "rogue_test_samples": 3000,
    }
    if counts != expected:
        raise RuntimeError(f"protocol counts {counts} != {expected}")

    split = {
        "dataset_id": "lora_rffi_known30_unknown15_v1",
        "protocol_id": "official_grouping_first120_val40_test40_v1",
        "split_index": 1,
        "status": STATUS,
        "known_classes": known_classes,
        "rogue_classes": rogue_classes,
        "known_devices_one_based": list(range(1, 31)),
        "unknown_devices_one_based": list(range(31, 46)),
        **counts,
        "role_codes": {
            "0": "known_train",
            "1": "known_validation",
            "2": "known_test_or_unknown_test",
        },
        "packet_ranges_per_known_device": {
            "train": [0, 120],
            "validation": [120, 160],
            "known_test": [160, 200],
        },
        "unknown_rule": "all 200 packets/device are test-only",
        "source_limitation": (
            "official released Test HDF5 pool; known devices use fixed 60/20/20 "
            "within-device split; unknown devices are never used before final evaluation"
        ),
        "selection_uses_model_outputs_or_performance": False,
        "adapter_manifest_sha256": sha256(adapter_manifest_path),
        "sample_role_sha256": sha256(role_path),
        "protocol_script": str(Path(__file__).resolve()),
        "protocol_script_sha256": sha256(Path(__file__)),
    }
    (output / "split_1.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    protocol = {
        "status": STATUS,
        "freeze_rule": (
            "No adapter, class assignment, packet range, preprocessing, model-selection, "
            "or threshold rule may be changed in response to known-test or unknown results."
        ),
        "formal_training_started_when_created": False,
        "open_set_test_accessed_when_created": False,
        "split_manifest": "split_1.json",
        "split_manifest_sha256": sha256(output / "split_1.json"),
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output), **counts}, indent=2))


if __name__ == "__main__":
    main()
