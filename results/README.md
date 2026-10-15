# Result index

These are compact, paper-facing artifacts copied from the frozen experiment
directories. The numbering matches the paper evaluation sequence.

| Folder | Principal artifact | What it supports |
|---|---|---|
| `01_external_baselines` | `official_baselines_v2_summary.json`, `bli2_paper_method_v5_summary.json` | NS-RFF, OpenMax, HyperRSI, and BLi2 comparison/audit |
| `02_representation_preservation` | `space_increment_summary.csv` | progressive deep → explicit RF → learned-view gains |
| `03_evidence_aggregation` | `aggregation_rule_summary.csv` | fixed three-channel aggregation choice |
| `04_conditional_reexamination` | `final_outer_summary.csv` | primary versus complete CEACC and accepted-region analysis |
| `05_lora_transfer` | `lora_cross_protocol_panels_v1/` | fixed-design LoRa transfer over three seeds |
| `06_robustness` | `openness_summary.csv` | openness, view removal, and temporal shifts |

Raw CSV files retain individual split/seed observations whenever practical.
Summary CSV files report aggregate mean and sample standard deviation. No
checkpoint, raw waveform, or per-sample score archive is included here.
