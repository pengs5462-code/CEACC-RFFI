# Fisher-retaining continuous cascade experiment

## Frozen protocol

- No checkpoint was retrained.
- The primary Fisher detector, its threshold, the selected review weight lambda, Q95 review threshold, and final binary decisions were unchanged.
- The experiment changes only the continuous score used for AUC, EER, and OSCR.
- Inside the Fisher-accepted region, Fisher and review scores were independently converted to empirical CDF ranks using Fisher-accepted known validation only.
- The retained score is `T_beta = (1-beta) r_F + beta r_R`, with beta in `{0, 0.25, 0.5, 0.75, 1}`.
- Beta was selected independently for each split/seed using the existing strict inner LOEO folds by mean inner OSCR, then worst-fold OSCR, then smaller beta.
- No outer-known or outer-rogue sample was used for beta selection.

## Inner selection

All nine configurations selected a non-endpoint mixture:

- beta = 0.5: 5/9 configurations
- beta = 0.75: 4/9 configurations
- beta = 0 or 1: 0/9 configurations

This provides inner-validation evidence that neither Fisher-only nor review-only ordering is preferred.

## Outer 9-run mean results (%)

| Continuous ranking | Accepted-region AUC | Global AUC | EER | OSCR | K-Accept | KCA | U-Recall | U-F1 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary Fisher | 76.62 | 84.89 | 22.25 | 74.55 | 89.32 | 78.64 | 58.68 | 68.57 | 69.25 |
| Current review replacement | 73.01 | 83.60 | 23.01 | 73.77 | 85.62 | 76.36 | 67.29 | 73.24 | 72.09 |
| Fisher-retaining review | **76.83** | **84.95** | **22.19** | **74.96** | 85.62 | 76.36 | 67.29 | 73.24 | 72.09 |

Fixed-point GCR metrics are exactly identical between the current replacement and Fisher-retaining scores, as required.

## Paired differences

Fisher-retaining minus current review replacement:

| Metric | Mean difference (pp) | Paired bootstrap 95% CI (pp) |
|---|---:|---:|
| Accepted-region AUC | +3.82 | [+2.45, +5.35] |
| Global AUC | +1.35 | [+0.90, +1.85] |
| EER | -0.82 | [-1.40, -0.27] |
| OSCR | +1.19 | [+0.73, +1.71] |

The Fisher-retaining score improves AUC and OSCR over the current replacement score in all 9/9 outer runs.

Fisher-retaining minus Primary Fisher:

| Metric | Mean difference (pp) | Paired bootstrap 95% CI (pp) |
|---|---:|---:|
| Accepted-region AUC | +0.21 | [-2.83, +2.64] |
| Global AUC | +0.06 | [-1.18, +1.02] |
| EER | -0.06 | [-1.59, +1.73] |
| OSCR | +0.41 | [-0.53, +1.18] |

The new score recovers the ranking loss and is numerically slightly better than Primary Fisher on average, but these differences are not statistically significant. Its defensible advantage is that it preserves Fisher-level global ranking while retaining the frozen GCR operating-point gains in unknown recall, U-F1, and Overall accuracy.
