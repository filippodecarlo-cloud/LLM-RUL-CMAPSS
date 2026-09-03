# P1 - Is the collapse specific to llama3.1:8b Q4_K_M?

Canonical configuration for every LLM row: FD001, zero-shot, n=30, 100 test engines, T=0.1, shipped prompt template. Only the model changes.


| Model | engines | RMSE | MAE | NASA | distinct | modal (share) | entropy (bits) | rho |
|---|---|---|---|---|---|---|---|---|
| llama3.1:8b Q4_K_M [5 runs, mean] | 100 | 61.90 | 51.58 | 45,454 | 2 | 23 (81%) | 0.71 | +0.202 |
| llama3.1:8b-instruct-q8_0 **PARTIAL** | 14 | 59.95 | 57.93 | 2,314 | 2 | 42 (86%) | 0.59 | -0.254 |
| mistral:7b | 100 | 46.88 | 40.39 | 10,641 | 4 | 60 (50%) | 1.23 | +0.073 |
| qwen2.5:7b | 100 | 71.69 | 60.45 | 107,312 | 1 | 15 (100%) | -0.00 | n/a |
| constant = train mean (no skill) | 100 | 41.94 | 34.83 | 33,354 | 1 | 87 (100%) | -0.00 | n/a |
| RandomForest | 100 | 17.96 | 13.27 | 761 | 100 | 8 (1%) | 6.64 | +0.813 |
| XGBoost | 100 | 15.95 | 11.71 | 477 | 100 | 5 (1%) | 6.64 | +0.844 |
| LSTM (as shipped) | 100 | 43.89 | 39.37 | 5,882 | 4 | 57 (74%) | 1.12 | n/a |
| LSTM (repaired) | 100 | 13.58 | 10.33 | 261 | 100 | 5 (1%) | 6.64 | +0.889 |

> **Warning:** 1 run(s) still in progress, computed on fewer than 100 engines (llama3.1:8b-instruct-q8_0: n=14). Rerun this script once the sweep finishes.

## Reading

- Distinct values over 100 engines, across every model tested: **1-4** (a genuine regressor gives up to 100). Output entropy -0.00-1.23 bits against a ceiling of 6.64.
- Models beating the no-skill constant (RMSE 41.94) on RMSE: **0/4**
- If the collapse were an artefact of one model or one quantisation, swapping the model would relieve it.