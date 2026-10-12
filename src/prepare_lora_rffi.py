#!/usr/bin/env python3
"""Audit and convert the official LoRa_RFFI HDF5 files without training.

The adapter follows the official ``LoadDataset`` convention: ``label`` is
one-based in HDF5 and each ``data`` row stores all I samples followed by all Q
samples.  Exactly the first 200 packets of every device are selected, matching
``pkt_range=range(0, 200)`` in the official loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from numpy.lib.format import open_memmap


SOURCES = (
    ("dataset_seen_devices.h5", range(1, 31), 400),
    ("dataset_residential.h5", range(31, 41), 400),
    ("dataset_rogue.h5", range(41, 46), 200),
)
PACKETS_PER_DEVICE = 200
EXPECTED_T = 8192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_source(path: Path, devices: range, source_packets: int) -> dict:
    with h5py.File(path, "r") as handle:
        datasets = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in handle.items()
            if isinstance(value, h5py.Dataset)
        }
        if "data" not in datasets or "label" not in datasets:
            raise ValueError(f"{path}: expected HDF5 datasets 'data' and 'label'")
        data = handle["data"]
        labels = np.asarray(handle["label"][:]).astype(np.int64).reshape(-1)
        if data.ndim != 2 or data.shape[1] % 2:
            raise ValueError(f"{path}: data must be [N,2T], got {data.shape}")
        if data.shape[1] // 2 != EXPECTED_T:
            raise ValueError(f"{path}: expected T={EXPECTED_T}, got {data.shape[1] // 2}")
        counts = Counter(map(int, labels))
        expected_devices = list(devices)
        if sorted(counts) != expected_devices:
            raise ValueError(
                f"{path}: labels {sorted(counts)} do not match {expected_devices}"
            )
        if any(counts[device] != source_packets for device in expected_devices):
            raise ValueError(f"{path}: unexpected per-device counts {dict(counts)}")
        if data.shape[0] != len(labels):
            raise ValueError(f"{path}: data/label row mismatch")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "root_keys": sorted(datasets),
        "datasets": datasets,
        "raw_label_mapping": {
            "stored": "one-based device id 1..45",
            "official_LoadDataset": "stored label - 1",
            "adapter_global_label": "stored label - 1 (0..44)",
        },
        "device_packet_counts_raw": {str(k): counts[k] for k in sorted(counts)},
    }


def convert(raw_root: Path, output_root: Path) -> dict:
    source_audits = [
        inspect_source(raw_root / filename, devices, source_packets)
        for filename, devices, source_packets in SOURCES
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    total = 45 * PACKETS_PER_DEVICE
    x_path = output_root / "X_pool_float32.npy"
    y_path = output_root / "Y_global.npy"
    source_path = output_root / "source_file_index.npy"
    packet_path = output_root / "source_packet_index.npy"
    x_out = open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(total, EXPECTED_T, 2)
    )
    y_out = open_memmap(y_path, mode="w+", dtype=np.int64, shape=(total,))
    source_out = open_memmap(source_path, mode="w+", dtype=np.uint8, shape=(total,))
    packet_out = open_memmap(packet_path, mode="w+", dtype=np.uint16, shape=(total,))

    cursor = 0
    selection_counts: Counter[int] = Counter()
    for source_id, (filename, devices, _) in enumerate(SOURCES):
        with h5py.File(raw_root / filename, "r") as handle:
            labels = np.asarray(handle["label"][:]).astype(np.int64).reshape(-1)
            data = handle["data"]
            for device in devices:
                indices = np.flatnonzero(labels == device)[:PACKETS_PER_DEVICE]
                if len(indices) != PACKETS_PER_DEVICE:
                    raise ValueError(f"{filename}: device {device} has too few packets")
                # HDF5 fancy indexing requires increasing indices, which the
                # official files already provide in acquisition order.
                rows = np.asarray(data[indices], dtype=np.float32)
                stop = cursor + PACKETS_PER_DEVICE
                x_out[cursor:stop, :, 0] = rows[:, :EXPECTED_T]
                x_out[cursor:stop, :, 1] = rows[:, EXPECTED_T:]
                y_out[cursor:stop] = device - 1
                source_out[cursor:stop] = source_id
                packet_out[cursor:stop] = np.arange(PACKETS_PER_DEVICE, dtype=np.uint16)
                selection_counts[device] += PACKETS_PER_DEVICE
                cursor = stop
    for value in (x_out, y_out, source_out, packet_out):
        value.flush()
    if cursor != total or set(selection_counts.values()) != {PACKETS_PER_DEVICE}:
        raise RuntimeError("balanced-pool construction failed")

    # Re-open the output read-only and validate the adapter contract without
    # computing any model score or any open-set/test performance metric.
    x_check = np.load(x_path, mmap_mode="r")
    y_check = np.load(y_path, mmap_mode="r")
    counts = Counter(map(int, y_check))
    if x_check.shape != (total, EXPECTED_T, 2) or x_check.dtype != np.float32:
        raise RuntimeError(f"unexpected output array {x_check.shape} {x_check.dtype}")
    if sorted(counts) != list(range(45)) or set(counts.values()) != {200}:
        raise RuntimeError(f"unexpected output labels/counts {dict(counts)}")
    if not np.isfinite(x_check).all():
        raise RuntimeError("converted pool contains non-finite values")

    manifest = {
        "dataset_id": "lora_rffi_official_balanced45_v1",
        "status": "ADAPTER_FROZEN_BEFORE_OPEN_SET_TEST",
        "created_by": str(Path(__file__).resolve()),
        "adapter_script_sha256": sha256(Path(__file__)),
        "official_loader_reference": "Openset_RFFI_TIFS/dataset_preparation.py:LoadDataset",
        "selection": {
            "devices_one_based": [1, 45],
            "packets_per_device": PACKETS_PER_DEVICE,
            "rule": "first 200 packets per device in HDF5 row/acquisition order",
            "equivalent_official_pkt_range": "range(0, 200)",
            "uses_performance_or_model_outputs": False,
        },
        "conversion": {
            "source_layout": "[N,2T], first T=I and second T=Q",
            "output_layout": "[N,T,2] with channels [I,Q]",
            "output_shape": list(x_check.shape),
            "output_dtype": str(x_check.dtype),
            "T": EXPECTED_T,
            "label_dtype": str(y_check.dtype),
            "label_mapping": "global zero-based label = official stored device id - 1",
        },
        "source_files": source_audits,
        "device_packet_counts": {str(k): counts[k - 1] for k in range(1, 46)},
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (x_path, y_path, source_path, packet_path)
        },
        "protocol_guards": {
            "formal_training_started": False,
            "open_set_test_metrics_computed": False,
            "adapter_must_remain_frozen_after_this_manifest": True,
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        payload = [
            inspect_source(args.raw_root / name, devices, count)
            for name, devices, count in SOURCES
        ]
    else:
        payload = convert(args.raw_root, args.output_root)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
