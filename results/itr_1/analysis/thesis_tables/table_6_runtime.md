## Tabelle 6: Durchschnittliche Laufzeit der Debugging-Workflows nach Fehlerklasse

| Fehlerklasse | Runtime Baseline (s) | Runtime XAI (s) | XAI/Baseline-Faktor | p Runtime | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Label Noise / random | 1,6226 | 5,7475 | 3,54 | < 0,001 | XAI langsamer, kein klarer Precision-Vorteil |
| Label Noise / hard | 1,4942 | 3,5522 | 2,38 | < 0,001 | XAI langsamer, kein klarer Precision-Vorteil |
| Data Leakage / direct | 0,5343 | 2,9727 | 5,56 | < 0,001 | XAI langsamer ohne Lokalisationsvorteil |
| Data Leakage / indirect | 0,5269 | 4,0960 | 7,77 | < 0,001 | XAI langsamer ohne Lokalisationsvorteil |
| Spurious Correlation / broken | 0,5260 | 4,0052 | 7,61 | < 0,001 | XAI langsamer, aber mit Lokalisationsvorteil |
| Spurious Correlation / inverted | 0,5319 | 3,9959 | 7,51 | < 0,001 | XAI langsamer, aber mit Lokalisationsvorteil |
