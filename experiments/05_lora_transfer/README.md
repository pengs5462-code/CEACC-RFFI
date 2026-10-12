# 05 — Cross-protocol LoRa transfer

The LoRa study transfers the architecture, evidence construction, and
ADS-B-selected re-examination coefficients. Only the enrolled LoRa devices are
used for retraining and recalibration; target LoRa unknowns are not used for
tuning.

Prepare the public LoRa recordings with `src/prepare_lora_rffi.py`, create the
registered roles with `src/prepare_lora_protocol.py`, train the corresponding
Stage-A/Stage-B checkpoints, then run:

```bash
bash run.sh
```

The committed three-seed summaries under `results/05_lora_transfer/` report
**0.9669 AUC** and **0.9645 OSCR** for the complete transferred design.
