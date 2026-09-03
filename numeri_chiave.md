# Numeri chiave per v14 — fonte unica di verita

**Generato automaticamente** da `experiment/p1/make_key_numbers.py` il 2026-09-03, direttamente dai CSV e dalle tracce.

> Se un numero nel manoscritto non coincide con questo file, **questo file ha ragione**: e derivato dai dati, non ricopiato da un report. Rilancia lo script dopo ogni nuovo esperimento.


**Inferenze LLM totali del progetto:** 4,657


## Baseline no-skill (FD001)

| Predittore | RMSE | MAE | NASA |
|---|---|---|---|
| constant = TRAIN mean RUL | 41.94 | 34.83 | 33,354 |
| constant = TRAIN median RUL | 49.20 | 38.01 | 166,491 |
| constant = TEST mean RUL (oracle) | 40.07 | 35.85 | 10,579 |
| constant = TEST median RUL (oracle) | 41.70 | 34.83 | 30,753 |
| constant = 50 | 46.94 | 41.51 | 7,992 |
| constant = RUL_CAP/2 (62.5) | 41.82 | 37.92 | 5,570 |
| constant = RMSE-optimal (oracle) | 40.08 | 35.91 | 10,197 |
| constant = NASA-optimal (oracle) | 42.27 | 38.25 | 5,506 |
| random uniform U(0,125), mean of 1000 | 55.02 | 45.13 | 155,400 |

- Configurazioni LLM che battono la costante media-train: **0/27**
- Miglior RMSE LLM (FD001): **45.60**
- Miglior NASA LLM (FD001): **18,572**

## Correlazione di rango

- Test nella famiglia: 27
- Nominalmente significativi (p<0,05): **10**
- Sopravvivono a BH-FDR 5%: **3**
- Sopravvivono a Bonferroni 5%: **1**
- Massimo |rho| su configurazioni LLM: **0.312** (varianza dei ranghi spiegata: 9.7%)

## Baseline supervisionate (P2.1, dopo riparazione)

| Dataset | Modello | RMSE | MAE | NASA | rho | distinti | costante? |
|---|---|---|---|---|---|---|---|
| FD001 | const_trainmean | 41.94 | 34.83 | 33,354 | n/a | 1 | SI |
| FD001 | RandomForest | 17.96 | 13.27 | 761 | +0.813 | 100 | no |
| FD001 | XGBoost | 15.95 | 11.71 | 477 | +0.844 | 100 | no |
| FD001 | LSTM_original | 43.89 | 39.37 | 5,882 | n/a | 2 | SI |
| FD001 | LSTM_fixed | 13.58 | 10.33 | 261 | +0.889 | 100 | no |
| FD003 | const_trainmean | 43.70 | 36.44 | 57,687 | n/a | 1 | SI |
| FD003 | RandomForest | 17.68 | 12.08 | 1,337 | +0.873 | 100 | no |
| FD003 | XGBoost | 16.22 | 11.08 | 923 | +0.890 | 100 | no |
| FD003 | LSTM_original | 39.87 | 34.70 | 6,063 | n/a | 1 | SI |
| FD003 | LSTM_fixed | 13.81 | 9.73 | 403 | +0.928 | 98 | no |

## Collasso per modello (FD001, zero-shot, n=30)

| Modello | motori | RMSE | distinti | modale (quota) | entropia (bit) |
|---|---|---|---|---|---|
| llama3.1:8b Q4_K_M | 100 | 62.83 | 2 | 23 (85%) | 0.61 |
| deepseek-r1_7b *(campione ridotto, n=50)* | 50 | 67.31 | 19 | 12 (24%) | 3.52 |
| llama3.1_8b-instruct-q8_0 | 100 | 52.40 | 5 | 42 (80%) | 0.99 |
| mistral_7b | 100 | 46.88 | 4 | 60 (50%) | 1.23 |
| qwen2.5_7b | 100 | 71.69 | 1 | 15 (100%) | -0.00 |

## Condizioni operative multiple (P2.2)

| Dataset | Modello | n | RMSE | rho | distinti | modale (quota) |
|---|---|---|---|---|---|---|
| FD002 | llama3.1:8b zero-shot n=30 | 259 | 63.20 | +0.272 | 4 | 23 (83%) |
| FD002 | constant = train mean | 259 | 44.93 | n/a | 1 | 86 (100%) |
| FD002 | RandomForest | 259 | 17.63 | +0.871 | 257 | 12 (1%) |
| FD002 | XGBoost | 259 | 17.34 | +0.880 | 259 | 0 (0%) |
| FD002 | LSTM (repaired) | 259 | 16.95 | +0.891 | 259 | 4 (0%) |
| FD004 | llama3.1:8b zero-shot n=30 | 248 | 66.56 | +0.146 | 4 | 23 (80%) |
| FD004 | constant = train mean | 248 | 45.57 | n/a | 1 | 92 (100%) |
| FD004 | RandomForest | 248 | 23.44 | +0.777 | 246 | 114 (1%) |
| FD004 | XGBoost | 248 | 24.79 | +0.771 | 240 | 125 (4%) |
| FD004 | LSTM (repaired) | 248 | 20.55 | +0.839 | 240 | 125 (4%) |

## Fedelta delle spiegazioni (P0.2a / P0.3)

- Opportunita sensore-traccia: **18,900**; affermazioni direzionali: **3,816**
- Accordo con la direzione canonica: **91.3%**
- Fedelta alla finestra mostrata: **42.7%**
- Tasso di base (asserire sempre il canonico): **46.7%**
- Sensori citati nel prompt: 94.5% vs non citati: 51.4%

### Bracci di controllo

| Braccio | RMSE | distinti | modale (quota) |
|---|---|---|---|
| repl | 61.57 | 3 | 23 (80%) |
| nocue | 59.13 | 4 | 23 (69%) |
| invcue | 62.32 | 5 | 23 (84%) |
| revdata | 61.21 | 4 | 23 (77%) |

## Ancoraggio (P1.3)

- Set di esempi testati: **7**
- Pearson r fra RUL medio degli esempi e media predetta: **+0.941** (p=0.0016)
- **Varianza dell'output spiegata dagli esempi: 89%**
- Retta: media_pred = 22.34 + 0.440 x media_esempi
- Media predetta minima/massima: 26.3 / 67.2 (media reale del test: 74,5)