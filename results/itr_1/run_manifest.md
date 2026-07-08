# Experiment Run Manifest

- Iteration: itr_1
- Run ID: 20260518_134738
- Started at: 2026-05-18 13:47:38
- Seed range: 0 to 29
- Python command: python
- Main output log: results/itr_1/log.md
- CSV output: results/itr_1/experiments.csv
- JSONL output: results/itr_1/experiments.jsonl

## Experiment Cases

| Case | Fault | Mode |
|---|---|---|
| Base experiment / no fault | none | 0 |
| Label Noise Fault, random | label_noise | 0 |
| Label Noise Fault, hard | label_noise | 1 |
| Data Leakage Fault, direct | data_leakage | 0 |
| Data Leakage Fault, indirect | data_leakage | 1 |
| Spurious Correlation Fault, broken | spurious_correlation | 0 |
| Spurious Correlation Fault, inverted | spurious_correlation | 1 |

## Run Summary

- Finished at: 2026-05-18 14:18:00
- Total planned runs: 210
- Failed runs: 0
- Total runtime: 00:30:21

## Output Files

- Log: `results/itr_1/log.md`
- CSV: `results/itr_1/experiments.csv`
- JSONL: `results/itr_1/experiments.jsonl`
- Failed runs: `results/itr_1/failed_runs.tsv`
