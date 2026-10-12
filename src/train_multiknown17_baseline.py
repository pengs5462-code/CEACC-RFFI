"""CE + Mondrian conformal baseline on a pre-registered multi-known split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from conformal_benchmark import MondrianMahalanobis, result_row
from spectral_conformal_benchmark import deterministic
from train_causal_ablation import build_model, evaluate_known, extract, move_batch, seed_everything
from train_representation import dataset_fingerprint


def make_loader(x, y, index, mapping, batch_size, shuffle):
    mapped = np.asarray([mapping[int(v)] for v in y[index]], dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x[index])), torch.from_numpy(mapped)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--known-reject-rate", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.variant = "backbone"
    args.input_normalization = "dc_rms"
    args.topk = 2
    args.fixed_rho = 0.25
    args.R = 8
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    root = Path(args.dataset_root)
    manifest_path = root / f"split_{args.split}.json"
    manifest = json.loads(manifest_path.read_text())
    allowed_status = {"development", "FROZEN_BEFORE_OPEN_SET_TEST"}
    if manifest["status"] not in allowed_status:
        raise ValueError(
            f"split manifest status must be one of {sorted(allowed_status)}, "
            f"got {manifest['status']!r}"
        )
    known_classes = manifest["known_classes"]
    args.num_classes = len(known_classes)
    mapping = {int(class_id): i for i, class_id in enumerate(known_classes)}
    x = np.load(root / "X_pool_float32.npy", mmap_mode="r")
    y = np.load(root / "Y_global.npy")
    role = np.load(root / "sample_role.npy")
    is_known = np.isin(y, known_classes)
    train_index = np.flatnonzero(is_known & (role == 0))
    val_index = np.flatnonzero(is_known & (role == 1))
    known_test_index = np.flatnonzero(is_known & (role == 2))
    expected = (manifest["known_train_samples"], manifest["known_validation_samples"], manifest["known_test_samples"])
    if tuple(map(len, (train_index, val_index, known_test_index))) != expected:
        raise ValueError("known split counts do not match manifest")

    seed_everything(args.seed + 20_000)
    train_loader = make_loader(x, y, train_index, mapping, args.batch_size, True)
    val_loader = make_loader(x, y, val_index, mapping, args.eval_batch_size, False)
    known_loader = make_loader(x, y, known_test_index, mapping, args.eval_batch_size, False)
    fingerprints = {
        "known_train": dataset_fingerprint(train_loader),
        "known_validation": dataset_fingerprint(val_loader),
        "known_test": dataset_fingerprint(known_loader),
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    model = build_model(args, device)
    seed_everything(args.seed + 20_000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    best_loss = float("inf")
    stale = 0
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = correct = samples = 0
        for data, target in train_loader:
            data, target = move_batch(data, target, device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch = len(target)
            loss_sum += loss.item() * batch
            correct += (logits.argmax(1) == target).sum().item()
            samples += batch
        val_loss, val_acc = evaluate_known(model, val_loader, device)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / samples,
            "train_acc": correct / samples,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "split_manifest": manifest,
                    "split_fingerprints": fingerprints,
                    "runner_sha256": runner_sha256,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    train_eval = deterministic(train_loader, args.eval_batch_size)
    train_e, train_l, train_y = extract(model, train_eval, device)
    val_e, _, val_y = extract(model, val_loader, device)
    known_e, known_l, known_y = extract(model, known_loader, device)
    calibrator = MondrianMahalanobis(train_e, train_l, train_y, val_e, val_y)
    known_p = calibrator.p_values(known_e)

    # Rogue samples are materialized only after training, model selection, and
    # known calibration have completed.
    rogue_classes = manifest["rogue_classes"]
    rogue_test_index = np.flatnonzero(np.isin(y, rogue_classes) & (role == 2))
    if len(rogue_test_index) != manifest["rogue_test_samples"]:
        raise ValueError("rogue test count does not match manifest")
    rogue_y_global = y[rogue_test_index].astype(np.int64)
    rogue_dataset = TensorDataset(
        torch.from_numpy(np.asarray(x[rogue_test_index])),
        torch.from_numpy(rogue_y_global),
    )
    rogue_loader = DataLoader(rogue_dataset, batch_size=args.eval_batch_size, shuffle=False)
    rogue_e, _, extracted_rogue_y = extract(model, rogue_loader, device)
    rogue_p = calibrator.p_values(rogue_e)
    metrics = result_row(known_p, rogue_p, known_y, args.known_reject_rate)
    metrics["network_known_closed_acc"] = float((known_l.argmax(1) == known_y).mean())
    per_rogue_class = {
        str(int(class_id)): result_row(
            known_p,
            rogue_p[extracted_rogue_y == class_id],
            known_y,
            args.known_reject_rate,
        )
        for class_id in sorted(np.unique(extracted_rogue_y))
    }
    result = {
        "method": "ce_dcrms_mondrian_conformal",
        "benchmark": manifest.get("dataset_id", "multiknown17_v1_low_sample"),
        "split": args.split,
        "seed": args.seed,
        "known_classes": known_classes,
        "rogue_classes": rogue_classes,
        "best_epoch": checkpoint["epoch"],
        "train_seconds": time.time() - start,
        "split_fingerprints": fingerprints,
        "runner_sha256": runner_sha256,
        "protocol": {
            "training": "known train role only",
            "checkpoint_selection": "known validation loss only",
            "calibration": "classwise known validation only",
            "target_rogue_materialized": "after training, selection, and calibration",
            "target_rogue_used_for_fit_selection_or_threshold": False,
            "source_limitation": manifest.get(
                "source_limitation",
                "repurposed balanced test pool; 60/20/20 per class",
            ),
        },
        "metrics": metrics,
        "per_rogue_class": per_rogue_class,
        "history": history,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
