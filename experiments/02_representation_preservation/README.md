# 02 — Representation preservation and selective learning

This experiment fixes the amplitude/statistical/temporal (AST) grouping, then
tests the complementary-only learned-view design. It supports the transition
from the 304-D preserved descriptor to the selected multi-view representation.

```bash
bash run.sh fixed-ast inner
bash run.sh complementary inner --worker 0 --workers 2
bash run.sh complementary outer
```

The reported space increments are in `results/02_representation_preservation/`:
deep only **0.7586 AUC**, + explicit RF **0.7919**, and + learned physical view
**0.8499**.
