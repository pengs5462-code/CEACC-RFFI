# Reproducibility contract

- Registered ADS-B roles: three known/rogue splits × seeds 42, 2023, 3407.
- Training and checkpoint selection: known train/validation samples only.
- Calibration and operating thresholds: known validation samples only.
- Target unknowns: final evaluation only.
- Primary fusion: fixed monotonic Fisher-type sum of empirical log-tail scores;
  no chi-square null distribution is assumed.
- Conditional re-examination: applied only inside the primary accepted region.
- LoRa transfer: ADS-B-selected architecture and mixing coefficients are fixed;
  networks are retrained and recalibrated using enrolled LoRa devices only.

CSV files ending in `_results.csv` contain run-level observations. Files ending
in `_summary.csv` contain the paper-facing aggregate means and sample standard
deviations. The committed results are immutable reference outputs; new runs
should be written to a separate directory and compared before replacement.
