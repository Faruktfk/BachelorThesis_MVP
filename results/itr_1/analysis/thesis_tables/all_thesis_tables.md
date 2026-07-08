## Tabelle 4: Aggregierte Lokalisierungsergebnisse von Baseline und XAI über 30 Seeds

| Fehlerklasse | Steps Baseline | Steps XAI | Δ Steps | MRR Baseline | MRR XAI | Hit@10 Baseline | Hit@10 XAI | p Steps | p MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | 1,0000 | 1,0333 | -0,0333 | 1,0000 | 0,9833 | 1,0000 | 1,0000 | 0,3173 | 0,3173 |
| Label Noise / hard | 1,0000 | 1,0667 | -0,0667 | 1,0000 | 0,9667 | 1,0000 | 1,0000 | 0,1573 | 0,1573 |
| Data Leakage / direct | 1,0000 | 1,0000 | 0,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| Data Leakage / indirect | 16,5000 | 9,0000 | +7,5000 | 0,0608 | 0,1164 | 0,0000 | 0,7667 | < 0,001 | < 0,001 |
| Spurious Correlation / broken | 20,0333 | 13,1333 | +6,9000 | 0,0502 | 0,0848 | 0,0000 | 0,3667 | < 0,001 | < 0,001 |
| Spurious Correlation / inverted | 20,0333 | 13,1333 | +6,9000 | 0,0502 | 0,0848 | 0,0000 | 0,3667 | < 0,001 | < 0,001 |



## Tabelle 5: Precision@k und Recall@k bei Label Noise

| Label-Noise-Modus | Precision@k Baseline | Precision@k XAI | Δ Precision@k | Recall@k Baseline | Recall@k XAI | Δ Recall@k | p Precision@k | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0,8539 | 0,8304 | -0,0235 | 0,8539 | 0,8304 | -0,0235 | 0,0305 | Baseline besser |
| hard | 0,7265 | 0,7088 | -0,0176 | 0,7265 | 0,7088 | -0,0176 | 0,0080 | Baseline besser |



## Tabelle 6: Durchschnittliche Laufzeit der Debugging-Workflows nach Fehlerklasse

| Fehlerklasse | Runtime Baseline (s) | Runtime XAI (s) | XAI/Baseline-Faktor | p Runtime | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Label Noise / random | 1,6226 | 5,7475 | 3,54 | < 0,001 | XAI langsamer, kein klarer Precision-Vorteil |
| Label Noise / hard | 1,4942 | 3,5522 | 2,38 | < 0,001 | XAI langsamer, kein klarer Precision-Vorteil |
| Data Leakage / direct | 0,5343 | 2,9727 | 5,56 | < 0,001 | XAI langsamer ohne Lokalisationsvorteil |
| Data Leakage / indirect | 0,5269 | 4,0960 | 7,77 | < 0,001 | XAI langsamer ohne Lokalisationsvorteil |
| Spurious Correlation / broken | 0,5260 | 4,0052 | 7,61 | < 0,001 | XAI langsamer, aber mit Lokalisationsvorteil |
| Spurious Correlation / inverted | 0,5319 | 3,9959 | 7,51 | < 0,001 | XAI langsamer, aber mit Lokalisationsvorteil |


