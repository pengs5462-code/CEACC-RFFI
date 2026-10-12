# 03 — Evidence construction and three-channel aggregation

The final evaluator reconstructs empirical known-only tail scores in the deep,
explicit-RF, and learned-view spaces, then compares monotonic aggregation rules.
The paper's primary detector is the fixed Fisher-type log-tail sum. This is an
empirical ranking construction, not a chi-square Fisher test.

```bash
bash run.sh
```

The command also regenerates downstream panels from the same frozen nine-run
checkpoint set. Aggregation tables are written under
`results/03_evidence_aggregation/`.
