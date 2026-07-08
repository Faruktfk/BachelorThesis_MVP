# Thesis-ready Tabellen

## Tabelle: Lokalisierungsergebnisse für H1/H2

| Fehlerklasse | Steps Baseline | Steps XAI | Δ Steps (Baseline - XAI) | MRR Baseline | MRR XAI | Hit@10 Baseline | Hit@10 XAI | p Steps | p MRR | p Hit@10 | Effektstärke dz (Steps) | Effektstärke dz (MRR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | 1,00 | 1,03 | -0,03 | 1,000 | 0,983 | 1,000 | 1,000 | 0,3173 | 0,3173 | 1,0000 | -0,18 | -0,18 |
| Label Noise / hard | 1,00 | 1,07 | -0,07 | 1,000 | 0,967 | 1,000 | 1,000 | 0,1573 | 0,1573 | 1,0000 | -0,26 | -0,26 |
| Data Leakage / direct | 1,00 | 1,00 | 0,00 | 1,000 | 1,000 | 1,000 | 1,000 | 1,0000 | 1,0000 | 1,0000 | n/a | n/a |
| Data Leakage / indirect | 16,50 | 9,00 | 7,50 | 0,061 | 0,116 | 0,000 | 0,767 | < 0,001 | < 0,001 | < 0,001 | 3,79 | 2,27 |
| Spurious Correlation / broken | 20,03 | 13,13 | 6,90 | 0,050 | 0,085 | 0,000 | 0,367 | < 0,001 | < 0,001 | < 0,001 | 1,47 | 1,30 |
| Spurious Correlation / inverted | 20,03 | 13,13 | 6,90 | 0,050 | 0,085 | 0,000 | 0,367 | < 0,001 | < 0,001 | < 0,001 | 1,47 | 1,30 |

## Tabelle: Precision@k und Recall@k bei Label Noise

| Label-Noise-Modus | Precision@k Baseline | Precision@k XAI | Δ Precision@k (XAI - Baseline) | Recall@k Baseline | Recall@k XAI | Δ Recall@k (XAI - Baseline) | p Precision@k | p Recall@k | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0,854 | 0,830 | -0,024 | 0,854 | 0,830 | -0,024 | 0,0305 | 0,0305 | Baseline besser |
| hard | 0,726 | 0,709 | -0,018 | 0,726 | 0,709 | -0,018 | 0,0080 | 0,0080 | Baseline besser |

## Tabelle: Laufzeitvergleich

| Fehlerklasse | Laufzeit Baseline (s) | Laufzeit XAI (s) | Laufzeitfaktor XAI/Baseline | p Runtime | Effektstärke dz (Laufzeit) | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | 1,62 | 5,75 | 3,54 | < 0,001 | -3,47 | XAI langsamer |
| Label Noise / hard | 1,49 | 3,55 | 2,38 | < 0,001 | -3,19 | XAI langsamer |
| Data Leakage / direct | 0,53 | 2,97 | 5,56 | < 0,001 | -5,61 | XAI langsamer |
| Data Leakage / indirect | 0,53 | 4,10 | 7,77 | < 0,001 | -5,28 | XAI langsamer |
| Spurious Correlation / broken | 0,53 | 4,01 | 7,61 | < 0,001 | -5,78 | XAI langsamer |
| Spurious Correlation / inverted | 0,53 | 4,00 | 7,51 | < 0,001 | -5,70 | XAI langsamer |

## Tabelle: Fix-Impact für H3 anhand Clean-Holdout Accuracy

| Fehlerklasse | Oracle-Potenzial | Fix-Impact Baseline | Fix-Impact XAI | Δ Fix-Impact (XAI - Baseline) | Oracle-normalisiert Baseline | Oracle-normalisiert XAI | Oracle-Bewertung | p Fix-Impact | Effektstärke dz (Fix-Impact) | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | 0,0020 | 0,0003 | 0,0020 | 0,0018 | 0,4479 | 0,5417 | reparaturrelevant | 0,3108 | 0,19 | Oracle-Potenzial zu schwach |
| Label Noise / hard | 0,0278 | 0,0164 | 0,0149 | -0,0015 | 0,5918 | 0,5566 | reparaturrelevant | 0,1788 | -0,20 | kein klarer Unterschied |
| Data Leakage / direct | 0,1465 | 0,1465 | 0,1465 | 0,0000 | 1,0000 | 1,0000 | reparaturrelevant | 1,0000 | n/a | kein klarer Unterschied |
| Data Leakage / indirect | -0,0015 | 0,0026 | -0,0000 | -0,0026 | 0,7143 | 0,5714 | zu schwach | 0,0228 | -0,50 | Oracle-Potenzial zu schwach |
| Spurious Correlation / broken | 0,0006 | 0,0020 | 0,0009 | -0,0012 | 0,2917 | 0,1458 | zu schwach | 0,6897 | -0,18 | Oracle-Potenzial zu schwach |
| Spurious Correlation / inverted | 0,0009 | 0,0020 | 0,0009 | -0,0012 | 0,1667 | 0,0208 | zu schwach | 0,6897 | -0,18 | Oracle-Potenzial zu schwach |

## Anhangstabelle: Fix-Impact über alle Clean-Holdout-Metriken

| Fehlerklasse | Metrik | Oracle-Potenzial | Fix-Impact Baseline | Fix-Impact XAI | Δ Fix-Impact (XAI - Baseline) | Oracle-normalisiert Baseline | Oracle-normalisiert XAI | p Fix-Impact | Effektstärke dz (Fix-Impact) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Label Noise / random | Accuracy | 0,0020 | 0,0003 | 0,0020 | 0,0018 | 0,4479 | 0,5417 | 0,3108 | 0,19 |
| Label Noise / random | Balanced Accuracy | 0,0011 | 0,0009 | 0,0011 | 0,0002 | 0,5422 | 0,6858 | 0,7944 | 0,02 |
| Label Noise / random | F1-Score | 0,0017 | 0,0002 | 0,0017 | 0,0016 | 0,4222 | 0,5517 | 0,1359 | 0,21 |
| Label Noise / random | ROC-AUC | 0,0040 | 0,0032 | 0,0038 | 0,0005 | 0,6985 | 0,9164 | 0,0599 | 0,22 |
| Label Noise / random | Log-Loss | 0,0489 | 0,0526 | 0,0518 | -0,0008 | 0,7364 | 0,8528 | 0,4427 | -0,01 |
| Label Noise / random | Brier-Score | 0,0207 | 0,0191 | 0,0191 | 0,0000 | 0,9330 | 0,9333 | 0,9914 | 0,00 |
| Label Noise / hard | Accuracy | 0,0278 | 0,0164 | 0,0149 | -0,0015 | 0,5918 | 0,5566 | 0,1788 | -0,20 |
| Label Noise / hard | Balanced Accuracy | 0,0278 | 0,0158 | 0,0140 | -0,0018 | 0,5402 | 0,4638 | 0,2344 | -0,21 |
| Label Noise / hard | F1-Score | 0,0222 | 0,0132 | 0,0121 | -0,0011 | 0,5911 | 0,5627 | 0,3109 | -0,20 |
| Label Noise / hard | ROC-AUC | 0,0064 | 0,0045 | 0,0039 | -0,0006 | 0,7074 | 0,5950 | 0,3052 | -0,26 |
| Label Noise / hard | Log-Loss | 0,0666 | 0,0321 | 0,0211 | -0,0110 | 0,4739 | 0,1618 | 0,1909 | -0,21 |
| Label Noise / hard | Brier-Score | 0,0134 | 0,0075 | 0,0068 | -0,0007 | 0,5073 | 0,4552 | 0,2015 | -0,28 |
| Data Leakage / direct | Accuracy | 0,1465 | 0,1465 | 0,1465 | 0,0000 | 1,0000 | 1,0000 | 1,0000 | n/a |
| Data Leakage / direct | Balanced Accuracy | 0,1388 | 0,1388 | 0,1388 | 0,0000 | 1,0000 | 1,0000 | 1,0000 | n/a |
| Data Leakage / direct | F1-Score | 0,1248 | 0,1248 | 0,1248 | 0,0000 | 1,0000 | 1,0000 | 1,0000 | n/a |
| Data Leakage / direct | ROC-AUC | 0,0751 | 0,0751 | 0,0751 | 0,0000 | 1,0000 | 1,0000 | 1,0000 | n/a |
| Data Leakage / direct | Log-Loss | 0,2344 | 0,2344 | 0,2344 | 0,0000 | 1,0000 | 1,0000 | 1,0000 | n/a |
| Data Leakage / direct | Brier-Score | 0,0843 | 0,0843 | 0,0843 | 0,0000 | 1,0000 | 1,0000 | 1,0000 | n/a |
| Data Leakage / indirect | Accuracy | -0,0015 | 0,0026 | -0,0000 | -0,0026 | 0,7143 | 0,5714 | 0,0228 | -0,50 |
| Data Leakage / indirect | Balanced Accuracy | -0,0012 | 0,0021 | -0,0000 | -0,0021 | 0,6548 | 0,6685 | 0,0326 | -0,35 |
| Data Leakage / indirect | F1-Score | -0,0013 | 0,0021 | -0,0000 | -0,0021 | 0,7245 | 0,5576 | 0,0058 | -0,52 |
| Data Leakage / indirect | ROC-AUC | 0,0004 | 0,0005 | 0,0009 | 0,0005 | 0,7043 | 0,6315 | 0,2273 | 0,15 |
| Data Leakage / indirect | Log-Loss | 0,0122 | 0,0095 | 0,0104 | 0,0010 | 0,4870 | 2,7138 | 0,1790 | 0,01 |
| Data Leakage / indirect | Brier-Score | 0,0006 | 0,0003 | 0,0005 | 0,0001 | n/a | n/a | 0,7369 | 0,07 |
| Spurious Correlation / broken | Accuracy | 0,0006 | 0,0020 | 0,0009 | -0,0012 | 0,2917 | 0,1458 | 0,6897 | -0,18 |
| Spurious Correlation / broken | Balanced Accuracy | 0,0008 | 0,0020 | 0,0012 | -0,0008 | 0,3878 | 0,2580 | 0,5496 | -0,10 |
| Spurious Correlation / broken | F1-Score | 0,0005 | 0,0017 | 0,0007 | -0,0010 | 0,2830 | 0,1351 | 0,0591 | -0,20 |
| Spurious Correlation / broken | ROC-AUC | 0,0002 | 0,0001 | 0,0004 | 0,0003 | 0,5428 | 0,3482 | 0,4688 | 0,11 |
| Spurious Correlation / broken | Log-Loss | 0,0195 | 0,0175 | 0,0176 | 0,0001 | 0,4932 | 0,3143 | 0,9405 | 0,00 |
| Spurious Correlation / broken | Brier-Score | 0,0005 | 0,0004 | 0,0003 | -0,0001 | n/a | n/a | 0,5755 | -0,05 |
| Spurious Correlation / inverted | Accuracy | 0,0009 | 0,0020 | 0,0009 | -0,0012 | 0,1667 | 0,0208 | 0,6897 | -0,18 |
| Spurious Correlation / inverted | Balanced Accuracy | 0,0012 | 0,0020 | 0,0012 | -0,0008 | 0,2628 | 0,1330 | 0,5496 | -0,10 |
| Spurious Correlation / inverted | F1-Score | 0,0007 | 0,0017 | 0,0007 | -0,0010 | 0,1580 | 0,0102 | 0,0591 | -0,20 |
| Spurious Correlation / inverted | ROC-AUC | 0,0003 | 0,0002 | 0,0005 | 0,0003 | 0,5379 | 0,3482 | 0,3957 | 0,12 |
| Spurious Correlation / inverted | Log-Loss | 0,0208 | 0,0176 | 0,0178 | 0,0001 | 0,3687 | 2,8104 | 0,9405 | 0,00 |
| Spurious Correlation / inverted | Brier-Score | 0,0007 | 0,0004 | 0,0003 | -0,0001 | n/a | n/a | 0,5755 | -0,05 |
