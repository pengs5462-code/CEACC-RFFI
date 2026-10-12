# PLSAR-RFFI

Official research artifact for **“Multi-View RF Evidence Aggregation with
Geometry-Aware Re-Examination for Open-Set RF Fingerprint Identification.”**

PLSAR follows a **preserve–learn selectively–aggregate–re-examine** pipeline.
It preserves a complete explicit RF descriptor, learns complementary
amplitude/statistical, spectral, and temporal-envelope views, calibrates three
open-set evidence channels with known data only, and conditionally re-examines
provisionally accepted samples using cross-view geometry and classifier entropy.

## Main results

| Dataset / setting | AUC | OSCR | Known accuracy | Unknown recall |
|---|---:|---:|---:|---:|
| ADS-B, primary three-space detector | 0.8499 | 0.7431 | 0.8593 | 0.5917 |
| ADS-B, complete PLSAR | **0.8557** | **0.7505** | **0.8593** | **0.6854** |
| LoRa, transferred design | **0.9669** | **0.9645** | — | — |

The ADS-B numbers aggregate three registered known/rogue splits and three
random seeds. Model fitting, selection, score construction, and operating-point
calibration use enrolled-device data only; target unknown samples are exposed
only for final evaluation.

## Repository map

The numbered folders follow the experimental section of the paper:

1. [`experiments/01_external_baselines`](experiments/01_external_baselines) —
   NS-RFF, OpenMax, and HyperRSI protocol adaptations.
2. [`experiments/02_representation_preservation`](experiments/02_representation_preservation) —
   fixed-view and complementary-view architecture studies.
3. [`experiments/03_evidence_aggregation`](experiments/03_evidence_aggregation) —
   deep, explicit-RF, learned-view, and aggregation-rule comparisons.
4. [`experiments/04_conditional_reexamination`](experiments/04_conditional_reexamination) —
   accepted-region analysis and Fisher-retaining re-examination.
5. [`experiments/05_lora_transfer`](experiments/05_lora_transfer) —
   cross-protocol LoRa transfer without target-unknown tuning.
6. [`experiments/06_robustness`](experiments/06_robustness) —
   openness, leave-one-view, and temporal-shift tests.

Shared implementations live in [`src`](src); the compact CSV/JSON artifacts
used to report the paper tables live in [`results`](results). Raw RF recordings
and trained checkpoints are intentionally not committed; see
[`docs/DATA_AND_CHECKPOINTS.md`](docs/DATA_AND_CHECKPOINTS.md).

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD/baselines"
```

Python 3.10 and a CUDA-capable PyTorch installation were used for the reported
runs. The evaluation code also falls back to CPU, although full retraining is
substantially slower.

## Reproducing the paper panels

After placing the data and checkpoints under the paths documented in
[`docs/DATA_AND_CHECKPOINTS.md`](docs/DATA_AND_CHECKPOINTS.md), run the numbered
experiment launchers. For example:

```bash
bash experiments/03_evidence_aggregation/run.sh
bash experiments/04_conditional_reexamination/run.sh
```

The central evaluator regenerates the space-increment, aggregation-rule,
accepted-region, final outer-test, and openness tables from frozen checkpoints.
All scripts accept explicit paths or resolve the repository through
`PLSAR_ROOT`, so no machine-specific absolute paths are required.

## Baseline scope

The external methods are author-released components adapted to the same RF
input, registered splits, seeds, and known-only calibration protocol. They are
not claimed as original PLSAR components. The result folders retain per-run
outputs and implementation notes so the distinction between native scores and
protocol adaptation is explicit.

## Citation

Please cite the accompanying paper. Machine-readable metadata is provided in
[`CITATION.cff`](CITATION.cff); bibliographic publication fields will be updated
when the article record is available.

## License

Original code in this repository is released under the [MIT License](LICENSE).
External datasets and referenced baseline repositories remain subject to their
respective licenses and terms.
