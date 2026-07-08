# Experiment Analysis Report

Input rows: 630

## Validation

All validation checks passed.

## Key Baseline vs. XAI Results

### steps_to_detect

| Case | Baseline mean | XAI mean | Mean improvement by XAI | Wilcoxon p | Cohen dz |
|---|---:|---:|---:|---:|---:|
| none | 0.0000 | 0.0000 | 0.0000 | 1.0000 | n/a |
| Label Noise / random | 1.0000 | 1.0333 | -0.0333 | 0.3173 | -0.1826 |
| Label Noise / hard | 1.0000 | 1.0667 | -0.0667 | 0.1573 | -0.2628 |
| Data Leakage / direct | 1.0000 | 1.0000 | 0.0000 | 1.0000 | n/a |
| Data Leakage / indirect | 16.5000 | 9.0000 | 7.5000 | 0.0000 | 3.7911 |
| Spurious Correlation / broken | 20.0333 | 13.1333 | 6.9000 | 0.0000 | 1.4703 |
| Spurious Correlation / inverted | 20.0333 | 13.1333 | 6.9000 | 0.0000 | 1.4703 |

### mrr

| Case | Baseline mean | XAI mean | Mean improvement by XAI | Wilcoxon p | Cohen dz |
|---|---:|---:|---:|---:|---:|
| none | 0.0000 | 0.0000 | 0.0000 | 1.0000 | n/a |
| Label Noise / random | 1.0000 | 0.9833 | -0.0167 | 0.3173 | -0.1826 |
| Label Noise / hard | 1.0000 | 0.9667 | -0.0333 | 0.1573 | -0.2628 |
| Data Leakage / direct | 1.0000 | 1.0000 | 0.0000 | 1.0000 | n/a |
| Data Leakage / indirect | 0.0608 | 0.1164 | 0.0557 | 0.0000 | 2.2708 |
| Spurious Correlation / broken | 0.0502 | 0.0848 | 0.0346 | 0.0000 | 1.2986 |
| Spurious Correlation / inverted | 0.0502 | 0.0848 | 0.0346 | 0.0000 | 1.2986 |

### hit_at_10

| Case | Baseline mean | XAI mean | Mean improvement by XAI | Wilcoxon p | Cohen dz |
|---|---:|---:|---:|---:|---:|
| none | 0.0000 | 0.0000 | 0.0000 | 1.0000 | n/a |
| Label Noise / random | 1.0000 | 1.0000 | 0.0000 | 1.0000 | n/a |
| Label Noise / hard | 1.0000 | 1.0000 | 0.0000 | 1.0000 | n/a |
| Data Leakage / direct | 1.0000 | 1.0000 | 0.0000 | 1.0000 | n/a |
| Data Leakage / indirect | 0.0000 | 0.7667 | 0.7667 | 0.0000 | 1.7822 |
| Spurious Correlation / broken | 0.0000 | 0.3667 | 0.3667 | 0.0009 | 0.7481 |
| Spurious Correlation / inverted | 0.0000 | 0.3667 | 0.3667 | 0.0009 | 0.7481 |

### precision_at_k

| Case | Baseline mean | XAI mean | Mean improvement by XAI | Wilcoxon p | Cohen dz |
|---|---:|---:|---:|---:|---:|
| none | n/a | n/a | n/a | n/a | n/a |
| Label Noise / random | 0.8539 | 0.8304 | -0.0235 | 0.0305 | -0.4291 |
| Label Noise / hard | 0.7265 | 0.7088 | -0.0176 | 0.0080 | -0.6195 |
| Data Leakage / direct | n/a | n/a | n/a | n/a | n/a |
| Data Leakage / indirect | n/a | n/a | n/a | n/a | n/a |
| Spurious Correlation / broken | n/a | n/a | n/a | n/a | n/a |
| Spurious Correlation / inverted | n/a | n/a | n/a | n/a | n/a |

### runtime_sec

| Case | Baseline mean | XAI mean | Mean improvement by XAI | Wilcoxon p | Cohen dz |
|---|---:|---:|---:|---:|---:|
| none | 0.0838 | 0.0852 | -0.0015 | 0.1642 | -0.2995 |
| Label Noise / random | 1.6226 | 5.7475 | -4.1248 | 0.0000 | -3.4674 |
| Label Noise / hard | 1.4942 | 3.5522 | -2.0580 | 0.0000 | -3.1945 |
| Data Leakage / direct | 0.5343 | 2.9727 | -2.4384 | 0.0000 | -5.6132 |
| Data Leakage / indirect | 0.5269 | 4.0960 | -3.5691 | 0.0000 | -5.2803 |
| Spurious Correlation / broken | 0.5260 | 4.0052 | -3.4792 | 0.0000 | -5.7827 |
| Spurious Correlation / inverted | 0.5319 | 3.9959 | -3.4640 | 0.0000 | -5.6989 |

## Generated files

- `experiments_repaired.csv`
- `validation_checks.csv`
- `workflow_counts.csv`
- `workflow_summary_by_case.csv`
- `paired_baseline_vs_xai_tests.csv`
- `repair_quality_counts.csv`
- `plots/mrr_boxplot.png`
- `plots/steps_to_detect_boxplot.png`
- `plots/hit_at_10_barplot.png`
- `plots/runtime_boxplot.png`