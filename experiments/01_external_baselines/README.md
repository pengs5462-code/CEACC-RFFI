# 01 — External open-set baselines

This folder covers the paper's external comparison under the same balanced RF
input, registered splits, and random seeds. Except for BLi2's paper-native
mixed-test clustering rule, calibration uses enrolled-device data only.

Implemented entries:

- **NS-RFF**: author-style angular classifier and native score.
- **OpenMax**: class-conditional EVT recalibration on frozen NetTime logits.
- **HyperRSI**: public complex CNN, CosFace, 512-D hypersphere embedding, and
  native negative maximum prototype-cosine score.
- **BLi2**: paper-described IAR representation, standard broad learning system
  with a 300-D hidden representation, and the published two-stage decision
  instantiated as mixed-test $K=2$ logit clustering with the nearest-centroid
  boundary. No end-to-end author code was available.

The scripts record both protocol metadata and per-run metrics. HyperRSI's native
hypersphere score is reported separately from its optional GPD-tail variant.
See the versioned reports and summaries in `results/01_external_baselines/`.

```bash
bash run.sh hyperrsi --dataset-root dataset/multiknown17_v1 --split 1 --seed 42 \
  --output-dir runs/hyperrsi_split1_seed42
```

Replace `hyperrsi` with `nsrff`, `openmax`, or `bli2`. OpenMax additionally requires
`--checkpoint` pointing to the matched frozen Stage-A checkpoint.

For BLi2, one registered run is launched as follows:

```bash
bash run.sh bli2 --dataset-root dataset/multiknown17_v1 --split 1 --seed 42 \
  --output-dir runs/bli2_split1_seed42
```

The main-table BLi2 row is the mean and sample standard deviation over splits
1/2/3 and seeds 42/2023/3407. Its native score is transductive: K-means is fit
to the mixed known/rogue target test logits, as described by the paper. The
runner also writes a `diagnostic_known_only_metrics` block, but that diagnostic
is not the value reported in the main table.
