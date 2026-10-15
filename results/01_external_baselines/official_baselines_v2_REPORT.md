# Additional official-code baseline audit

All reported rows use the same three outer emitter splits, seeds
42/2023/3407, known-only fitting/checkpoint selection, and a
known-validation Q90 operating threshold. Values are mean ± sample standard
deviation over nine runs.

| Method | AUC | EER | Known Acc. | K-Accept | U-Recall | U-F1 | OSCR | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NS-RFF CNN-L2Softmax | 0.6774 ± 0.0649 | 0.3729 ± 0.0602 | 0.7086 ± 0.0360 | 0.8759 ± 0.0337 | 0.2792 ± 0.1000 | 0.3863 ± 0.1163 | 0.5217 ± 0.0492 | Main table |
| PROSER | 0.5757 ± 0.0665 | 0.4536 ± 0.0528 | 0.6321 ± 0.0342 | 0.9019 ± 0.0307 | 0.1472 ± 0.0712 | 0.2282 ± 0.0949 | 0.4131 ± 0.0441 | Appendix / robustness |
| GCPL | 0.5374 ± 0.0916 | 0.4879 ± 0.0789 | 0.6660 ± 0.0495 | 0.8710 ± 0.0518 | 0.1840 ± 0.0957 | 0.2660 ± 0.1182 | 0.4146 ± 0.0657 | Appendix only |

## Recommended main comparison

The defensible official-code core is now:

| Method | AUC | Known Acc. | OSCR |
|---|---:|---:|---:|
| HyperRSI, native hypersphere score | 0.6317 ± 0.0767 | 0.7580 ± 0.0279 | 0.5297 ± 0.0591 |
| NS-RFF, native hypersphere/cosine score | 0.6774 ± 0.0649 | 0.7086 ± 0.0360 | 0.5217 ± 0.0492 |
| OpenMax, audited EVT | 0.6330 ± 0.0269 | 0.6025 ± 0.0357 | 0.4349 ± 0.0312 |
| BLi2, paper-described native clustering | 0.6569 ± 0.1001 | 0.5969 ± 0.0309 | 0.3717 ± 0.0477 |

These entries avoid the appearance that all competing methods failed to train:
each has a recognizable open-set mechanism, non-collapsed closed-set accuracy,
and AUC above 0.63. BLi2 is listed separately from the strictly known-only
protocol because its published decision clusters the mixed target test batch;
see `BLi2_REPORT.md` for the implementation and disclosure details.

## Implementation audit

- **NS-RFF** follows the authors' released CNN-L2Softmax architecture
  (`d2=24`, 512-D embedding), L2Softmax scale 10, Adam learning rate `1e-4`
  with betas `(0.5, 0.99)`, and cosine matching. The official architecture is
  hard-coded for a 1,280-complex-sample ZigBee preamble, so every 4,800-sample
  record is deterministically adapted by taking its leading 1,280 samples.
  This limitation must be disclosed in the paper. Repository:
  https://github.com/xrj-com/NS-RFF (audited commit
  `b5b4180b6d0d6a67a641fe326d47ce7a69e638f8`)
- **PROSER** uses the authors' one-placeholder head, Beta(1,1) manifold
  mixup, three published loss terms and fixed ten-epoch fine-tuning. The
  objective is applied at the frozen 64-D NetTime feature layer. Repository:
  https://github.com/LAMDA-CL/CVPR21-Proser (audited commit
  `f5b53b90509e6460a783a0baa72bda52364810fd`)
- **GCPL** uses the exact distance division, distance cross-entropy and
  prototype compactness term in the public ARPL repository. Repository:
  https://github.com/iCGY96/ARPL (audited commit
  `3ede8b38e1cfb9d70e106cc19d563453110c36ab`)
- **RPL was rejected before the full matrix.** Its pilot produced 0% known
  test accuracy when only the official reciprocal-point head was fitted on
  frozen features. Reporting its AUC of 0.737 would therefore be misleading.
- **RED/DMNet was not run as an official reproduction.** The linked repository
  exposes a ComplexCNN plus center loss but does not contain the paper's
  denoising-autoencoder rogue detector or an open-set scoring/evaluation path.

No target rogue sample was used for training, checkpoint selection, score
fitting, model selection, or threshold calibration.
