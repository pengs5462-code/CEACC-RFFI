# BLi2 paper-described reproduction

Reference: X. Zhang, Q. Huang, X. Ji, L. Shi, G. Hu, and K. Li,
“A Lightweight Open-Set Specific Emitter Identification Based on Broad
Learning,” *IEEE Internet of Things Journal*, vol. 13, no. 4,
pp. 6158–6171, 2026, doi: 10.1109/JIOT.2025.3637514.

No end-to-end author implementation was available. This reproduction follows
the published method rather than substituting the CEACC backbone:

- IAR phase features use differential constellation trajectories with delays
  1–10 and 240 fixed frame bins; phase is encoded by equivalent unit-circle
  real/imaginary coordinates.
- The standard broad learning system uses `p=100`, `k=2`, `q=50`, and `b=2`,
  giving a 300-D mapped-plus-enhancement representation.
- The reported score follows the paper-native second stage: K-means with
  `K=2` is fit to mixed target-test logits and the nearest-centroid boundary
  assigns the known/unknown side.

The final comparison uses the same three registered emitter splits and seeds
42, 2023, and 3407 as CEACC. Values below are mean ± sample standard deviation
over nine runs.

| AUC | EER | Known Acc. | K-Accept | U-Recall | U-F1 | OSCR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6569 ± 0.1001 | 0.3722 ± 0.0916 | 0.5969 ± 0.0309 | 0.6920 ± 0.0889 | 0.5396 ± 0.1192 | 0.5710 ± 0.1204 | 0.3717 ± 0.0477 |

## Protocol caveat

BLi2's native mixed-test clustering is transductive and therefore observes the
unlabeled target batch, which contains both enrolled and rogue samples. It does
not use rogue labels or rogue samples for gradient training, but it is not an
enrolled-only calibration rule. This exception is stated explicitly in the
manuscript and should be preserved when quoting these results.

The machine-readable summary is
[`bli2_paper_method_v5_summary.json`](bli2_paper_method_v5_summary.json), and
the executable implementation is
[`baselines/train_bli2_paper.py`](../../baselines/train_bli2_paper.py).
