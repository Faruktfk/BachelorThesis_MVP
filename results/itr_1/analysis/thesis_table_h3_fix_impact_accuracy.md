| Fehlerklasse | Oracle-Potenzial | Fix-Impact Baseline | Fix-Impact XAI | Δ Fix-Impact (XAI - Baseline) | Oracle-normalisiert Baseline | Oracle-normalisiert XAI | Oracle-Bewertung | p Fix-Impact | Effektstärke dz (Fix-Impact) | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | 0,0020 | 0,0003 | 0,0020 | 0,0018 | 0,4479 | 0,5417 | reparaturrelevant | 0,3108 | 0,19 | Oracle-Potenzial zu schwach |
| Label Noise / hard | 0,0278 | 0,0164 | 0,0149 | -0,0015 | 0,5918 | 0,5566 | reparaturrelevant | 0,1788 | -0,20 | kein klarer Unterschied |
| Data Leakage / direct | 0,1465 | 0,1465 | 0,1465 | 0,0000 | 1,0000 | 1,0000 | reparaturrelevant | 1,0000 | n/a | kein klarer Unterschied |
| Data Leakage / indirect | -0,0015 | 0,0026 | -0,0000 | -0,0026 | 0,7143 | 0,5714 | zu schwach | 0,0228 | -0,50 | Oracle-Potenzial zu schwach |
| Spurious Correlation / broken | 0,0006 | 0,0020 | 0,0009 | -0,0012 | 0,2917 | 0,1458 | zu schwach | 0,6897 | -0,18 | Oracle-Potenzial zu schwach |
| Spurious Correlation / inverted | 0,0009 | 0,0020 | 0,0009 | -0,0012 | 0,1667 | 0,0208 | zu schwach | 0,6897 | -0,18 | Oracle-Potenzial zu schwach |