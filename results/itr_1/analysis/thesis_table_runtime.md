| Fehlerklasse | Laufzeit Baseline (s) | Laufzeit XAI (s) | Laufzeitfaktor XAI/Baseline | p Runtime | Effektstärke dz (Laufzeit) | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | 1,62 | 5,75 | 3,54 | < 0,001 | -3,47 | XAI langsamer |
| Label Noise / hard | 1,49 | 3,55 | 2,38 | < 0,001 | -3,19 | XAI langsamer |
| Data Leakage / direct | 0,53 | 2,97 | 5,56 | < 0,001 | -5,61 | XAI langsamer |
| Data Leakage / indirect | 0,53 | 4,10 | 7,77 | < 0,001 | -5,28 | XAI langsamer |
| Spurious Correlation / broken | 0,53 | 4,01 | 7,61 | < 0,001 | -5,78 | XAI langsamer |
| Spurious Correlation / inverted | 0,53 | 4,00 | 7,51 | < 0,001 | -5,70 | XAI langsamer |