# Data and checkpoint layout

The repository contains split manifests and aggregate numerical results, but
does not redistribute raw RF recordings or trained weights. This avoids
silently relicensing third-party datasets and keeps device recordings out of
Git history.

## ADS-B benchmark

Prepare the 17-device balanced pool with:

```bash
python src/prepare_multiknown17.py \
  --source /path/to/emitter_based_splits_balanced \
  --output dataset/multiknown17_v1
```

The generated directory contains:

```text
dataset/multiknown17_v1/
├── X_pool_float32.npy
├── Y_global.npy
├── sample_role.npy
├── class_local_index.npy
├── PROVENANCE.json
└── split_1.json ... split_5.json
```

The registered manifests used by the reported development experiments are in
`data/manifests/`. Compare their class assignments and source hashes before
running. Splits 1–3 and seeds 42, 2023, and 3407 form the nine reported runs.

Stage-A and Stage-B training scripts write checkpoints below `results/` in the
layout referenced by `src/final_ast_complementary_paper_eval.py`. Checkpoints
are ignored by Git; only their compact result tables are versioned.

## LoRa benchmark

`src/prepare_lora_rffi.py` adapts the public LoRa RFFI dataset into the common
`[sample, time, I/Q]` layout, and `src/prepare_lora_protocol.py` creates the
known-only training/validation roles. The paper freezes the architecture and
mixing coefficients selected on ADS-B, retrains on enrolled LoRa devices, and
recalibrates with LoRa known data only. It does not tune against target LoRa
unknowns.

## Expected placement

Either place the prepared material under this repository or point scripts at a
separate working root:

```bash
export PLSAR_ROOT=/path/to/plsar-working-root
```

The working root may contain private `dataset/`, checkpoint, and intermediate
directories while this source checkout remains clean.
