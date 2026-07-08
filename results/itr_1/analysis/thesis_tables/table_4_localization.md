## Tabelle 4: Aggregierte Lokalisierungsergebnisse von Baseline und XAI über 30 Seeds

| Fehlerklasse | Steps Baseline | Steps XAI | Δ Steps | MRR Baseline | MRR XAI | Hit@10 Baseline | Hit@10 XAI | p Steps | p MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | 1,0000 | 1,0333 | -0,0333 | 1,0000 | 0,9833 | 1,0000 | 1,0000 | 0,3173 | 0,3173 |
| Label Noise / hard | 1,0000 | 1,0667 | -0,0667 | 1,0000 | 0,9667 | 1,0000 | 1,0000 | 0,1573 | 0,1573 |
| Data Leakage / direct | 1,0000 | 1,0000 | 0,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| Data Leakage / indirect | 16,5000 | 9,0000 | +7,5000 | 0,0608 | 0,1164 | 0,0000 | 0,7667 | < 0,001 | < 0,001 |
| Spurious Correlation / broken | 20,0333 | 13,1333 | +6,9000 | 0,0502 | 0,0848 | 0,0000 | 0,3667 | < 0,001 | < 0,001 |
| Spurious Correlation / inverted | 20,0333 | 13,1333 | +6,9000 | 0,0502 | 0,0848 | 0,0000 | 0,3667 | < 0,001 | < 0,001 |
