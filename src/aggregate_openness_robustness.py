"""Exact openness-subset evaluation for a frozen open-set score.

Each NPZ input must contain the unchanged known-validation, known-test, and
per-sample rogue scores exported by physics_latent_rejection_benchmark.py.
The script never fits, calibrates, selects, or changes a threshold.  It only
chooses rogue samples by class label and recomputes the requested metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from score_benchmark import metric_row


METRICS = (
    "auc",
    "eer",
    "known_acc",
    "known_accept",
    "unknown_recall",
    "unknown_f1",
    "oscr",
)
SUBSET_SIZES = (2, 4, 6, 8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(bundle: np.lib.npyio.NpzFile, name: str):
    return bundle[name].item()


def load_run(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "score_name",
            "split",
            "seed",
            "known_classes",
            "rogue_classes",
            "validation_score",
            "known_score",
            "rogue_score",
            "rogue_label",
            "known_correct",
            "known_reject_rate",
            "split_fingerprint_json",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        run = {
            "path": path,
            "sha256": sha256(path),
            "score_name": str(scalar(bundle, "score_name")),
            "split": int(scalar(bundle, "split")),
            "seed": int(scalar(bundle, "seed")),
            "known_classes": bundle["known_classes"].astype(int),
            "rogue_classes": bundle["rogue_classes"].astype(int),
            "validation_score": bundle["validation_score"].astype(float),
            "known_score": bundle["known_score"].astype(float),
            "rogue_score": bundle["rogue_score"].astype(float),
            "rogue_label": bundle["rogue_label"].astype(int),
            "known_correct": bundle["known_correct"].astype(bool),
            "known_reject_rate": float(scalar(bundle, "known_reject_rate")),
            "split_fingerprint_json": str(scalar(bundle, "split_fingerprint_json")),
        }
    if len(run["known_classes"]) != 9 or len(run["rogue_classes"]) != 8:
        raise ValueError(f"{path} is not a 9-known/8-rogue run")
    if set(np.unique(run["rogue_label"])) != set(run["rogue_classes"]):
        raise ValueError(f"{path} rogue labels do not match declared rogue classes")
    if len(run["rogue_score"]) != len(run["rogue_label"]):
        raise ValueError(f"{path} rogue score/label lengths differ")
    return run


def evaluate(run: dict, subset: tuple[int, ...]) -> dict:
    mask = np.isin(run["rogue_label"], subset)
    rogue_score = run["rogue_score"][mask]
    if not len(rogue_score):
        raise ValueError(f"empty subset {subset} in {run['path']}")
    metric = metric_row(
        run["score_name"],
        run["validation_score"],
        run["known_score"],
        rogue_score,
        run["known_correct"],
        run["known_reject_rate"],
    )
    return {
        "method": "Ours frozen physical4view-304D three-space",
        "score": run["score_name"],
        "split": run["split"],
        "seed": run["seed"],
        "openness": f"9:{len(subset)}",
        "subset_size": len(subset),
        "subset_id": "-".join(map(str, subset)),
        "rogue_classes": json.dumps(list(subset), separators=(",", ":")),
        "n_known": len(run["known_score"]),
        "n_unknown": len(rogue_score),
        "auc": metric["auc"],
        "eer": metric["eer"],
        "known_acc": float(run["known_correct"].mean()),
        "known_accept": metric["known_accept_rate"],
        "unknown_recall": metric["unknown_recall"],
        "unknown_f1": metric["unknown_f1"],
        "oscr": metric["oscr"],
        "tau_known_only": metric["tau_known_only"],
        "known_reject_rate": run["known_reject_rate"],
        "split_fingerprint_json": run["split_fingerprint_json"],
        "raw_score_file": str(run["path"].resolve()),
        "raw_score_sha256": run["sha256"],
        "rogue_used_for_calibration_or_selection": False,
    }


def validate_full_set(
    rows: list[dict], reference_dir: Path, reference_pattern: str, tolerance: float
) -> None:
    mapping = {
        "auc": "auc",
        "eer": "eer",
        "known_acc": "known_closed_acc",
        "known_accept": "known_accept_rate",
        "unknown_recall": "unknown_recall",
        "unknown_f1": "unknown_f1",
        "oscr": "oscr",
    }
    checked = 0
    for row in rows:
        if row["subset_size"] != 8:
            continue
        reference = reference_dir / reference_pattern.format(
            split=row["split"], seed=row["seed"]
        )
        payload = json.loads(reference.read_text(encoding="utf-8"))
        candidate = next(item for item in payload["results"] if item["score"] == row["score"])
        for current_name, reference_name in mapping.items():
            delta = abs(float(row[current_name]) - float(candidate[reference_name]))
            if delta > tolerance:
                raise AssertionError(
                    f"9:8 regression mismatch {reference.name} {current_name}: {delta}"
                )
        if abs(float(row["tau_known_only"]) - float(candidate["tau_known_only"])) > tolerance:
            raise AssertionError(f"9:8 threshold mismatch in {reference.name}")
        checked += 1
    if checked != 9:
        raise AssertionError(f"expected 9 full-set regression checks, got {checked}")


def write_results(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for subset_size in SUBSET_SIZES:
        group = [row for row in rows if row["subset_size"] == subset_size]
        summary = {
            "method": "Ours frozen physical4view-304D three-space",
            "score": group[0]["score"],
            "openness": f"9:{subset_size}",
            "subset_size": subset_size,
            "n_records": len(group),
            "n_outer_splits": len({row["split"] for row in group}),
            "n_seeds": len({row["seed"] for row in group}),
            "subsets_per_split_seed": len(group) // 9,
            "std_ddof": 1,
            "rogue_used_for_calibration_or_selection": False,
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, summary: list[dict]) -> None:
    x = np.arange(len(summary))
    labels = [row["openness"] for row in summary]
    series = (
        ("auc", "AUC", "#1f4e79", "o"),
        ("oscr", "OSCR", "#2a9d8f", "s"),
        ("unknown_f1", "Unknown F1", "#e76f51", "^"),
        ("unknown_recall", "Unknown Recall", "#7b2cbf", "D"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, label, color, marker in series:
        mean = np.asarray([row[f"{key}_mean"] for row in summary])
        std = np.asarray([row[f"{key}_std"] for row in summary])
        ax.errorbar(
            x,
            mean,
            yerr=std,
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=2.0,
            capsize=3,
            label=label,
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Known : rogue classes")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument(
        "--reference-pattern",
        default="physics_latent_consensus_zp_split{split}_seed{seed}.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    runs = [load_run(Path(value)) for value in args.inputs]
    keys = [(run["split"], run["seed"]) for run in runs]
    expected = {(split, seed) for split in (1, 2, 3) for seed in (42, 2023, 3407)}
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError(f"inputs must contain each split/seed exactly once; got {sorted(keys)}")
    score_names = {run["score_name"] for run in runs}
    if len(score_names) != 1:
        raise ValueError(f"mixed scores are forbidden: {sorted(score_names)}")

    rows = []
    for run in sorted(runs, key=lambda item: (item["split"], item["seed"])):
        classes = tuple(sorted(map(int, run["rogue_classes"])))
        for subset_size in SUBSET_SIZES:
            for subset in itertools.combinations(classes, subset_size):
                rows.append(evaluate(run, subset))

    expected_rows = 9 * (28 + 70 + 28 + 1)
    if len(rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} subset-run rows, got {len(rows)}")
    validate_full_set(
        rows, Path(args.reference_dir), args.reference_pattern, args.tolerance
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_results(output_dir / "openness_results.csv", rows)
    write_summary(output_dir / "openness_summary.csv", summary)
    plot_summary(output_dir / "openness_robustness_curve.png", summary)
    print(json.dumps({"rows": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
