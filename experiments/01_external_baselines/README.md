# 01 — External open-set baselines

This folder covers the paper's matched external comparison. Each adaptation
uses the same balanced RF input, registered split, random seed, and known-only
calibration rule as CEACC.

Implemented entries:

- **NS-RFF**: author-style angular classifier and native score.
- **OpenMax**: class-conditional EVT recalibration on frozen NetTime logits.
- **HyperRSI**: public complex CNN, CosFace, 512-D hypersphere embedding, and
  native negative maximum prototype-cosine score.

The scripts record both protocol metadata and per-run metrics. HyperRSI's native
hypersphere score is reported separately from its optional GPD-tail variant.
See the versioned reports and summaries in `results/01_external_baselines/`.

```bash
bash run.sh hyperrsi --dataset-root dataset/multiknown17_v1 --split 1 --seed 42 \
  --output-dir runs/hyperrsi_split1_seed42
```

Replace `hyperrsi` with `nsrff` or `openmax`. OpenMax additionally requires
`--checkpoint` pointing to the matched frozen Stage-A checkpoint.
