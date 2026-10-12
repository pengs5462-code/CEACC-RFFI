# 04 — Conditional geometry-aware re-examination

This stage operates only on samples provisionally accepted by the primary
detector. It combines cross-view geometry with classifier entropy and retains
the primary Fisher evidence in the final ranking.

```bash
bash run.sh final
bash run.sh cascade outer
```

On ADS-B, the accepted-region AUC increases by **3.10 points**; unknown recall
rises from **0.5917 to 0.6854**. The global AUC gain is smaller and is not
presented as statistically significant. Run-level values, confidence intervals,
and the known-acceptance tradeoff are retained in
`results/04_conditional_reexamination/`.
