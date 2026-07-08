| Fehlerklasse | Steps Baseline | Steps XAI | Δ Steps (Baseline - XAI) | MRR Baseline | MRR XAI | Hit@10 Baseline | Hit@10 XAI | p Steps | p MRR | p Hit@10 | Effektstärke dz (Steps) | Effektstärke dz (MRR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | 1,00 | 1,03 | -0,03 | 1,000 | 0,983 | 1,000 | 1,000 | 0,3173 | 0,3173 | 1,0000 | -0,18 | -0,18 |
| Label Noise / hard | 1,00 | 1,07 | -0,07 | 1,000 | 0,967 | 1,000 | 1,000 | 0,1573 | 0,1573 | 1,0000 | -0,26 | -0,26 |
| Data Leakage / direct | 1,00 | 1,00 | 0,00 | 1,000 | 1,000 | 1,000 | 1,000 | 1,0000 | 1,0000 | 1,0000 | n/a | n/a |
| Data Leakage / indirect | 16,50 | 9,00 | 7,50 | 0,061 | 0,116 | 0,000 | 0,767 | < 0,001 | < 0,001 | < 0,001 | 3,79 | 2,27 |
| Spurious Correlation / broken | 20,03 | 13,13 | 6,90 | 0,050 | 0,085 | 0,000 | 0,367 | < 0,001 | < 0,001 | < 0,001 | 1,47 | 1,30 |
| Spurious Correlation / inverted | 20,03 | 13,13 | 6,90 | 0,050 | 0,085 | 0,000 | 0,367 | < 0,001 | < 0,001 | < 0,001 | 1,47 | 1,30 |