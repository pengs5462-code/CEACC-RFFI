# Official-code baseline comparison

All rows use the same three outer emitter splits, seeds 42/2023/3407, known-only checkpoint selection, and a known-validation Q90 operating threshold.

| Method / score | AUC | EER | Known Acc. | K-Accept | U-Recall | U-F1 | OSCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| HyperRSI native hypersphere score | 0.6317 ± 0.0767 | 0.4137 ± 0.0702 | 0.7580 ± 0.0279 | 0.8963 ± 0.0261 | 0.2403 ± 0.0851 | 0.3476 ± 0.1005 | 0.5297 ± 0.0591 |
| HyperRSI released GPD-tail probability | 0.5712 ± 0.0426 | 0.4410 ± 0.0468 | 0.7580 ± 0.0279 | 0.9710 ± 0.0577 | 0.0444 ± 0.0926 | 0.0652 ± 0.1340 | 0.4655 ± 0.0431 |
| ARPL official head/loss | 0.5809 ± 0.1432 | 0.4425 ± 0.1211 | 0.6562 ± 0.0524 | 0.8969 ± 0.0306 | 0.2111 ± 0.1147 | 0.3070 ± 0.1489 | 0.4095 ± 0.0801 |
| OpenMax audited EVT | 0.6330 ± 0.0269 | 0.4013 ± 0.0209 | 0.6025 ± 0.0357 | 0.8988 ± 0.0256 | 0.1667 ± 0.0820 | 0.2530 ± 0.1092 | 0.4349 ± 0.0312 |

Notes:
- HyperRSI is the authors' native nine-layer complex CNN with CosFace (s=8, m=0.2), 512-D normalized embeddings, class prototypes, and a 17-sample GPD tail.
- The released GPD mapping assigns the same zero unknownness to all observations below the fitted tail boundary. It is retained for reproducibility, but its ties discard continuous ordering. The native negative maximum prototype cosine is therefore the appropriate HyperRSI continuous score for AUC/EER/OSCR; both versions are reported.
- ARPL uses the exact official reciprocal-point logits, margin loss, temperature 1, weight_pl 0.1, and negative maximum reciprocal-point logit. Only the official head is trained on the exact frozen NetTime representation used by the comparison protocol.
- OpenMax fits class MAVs on correctly classified known-training logits, uses eucos distance, Weibull tail size 20, alpha rank 3, and the exact frozen NetTime checkpoint.
- No target rogue sample is used for training, checkpoint selection, score fitting, or threshold selection.
