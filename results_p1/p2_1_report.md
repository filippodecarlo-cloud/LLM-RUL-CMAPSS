# P2.1 - Repairing the baselines

The submitted paper's LSTM baseline is a constant predictor (P0.1b). Here it is diagnosed and retrained, and XGBoost is added as a strong tabular reference.


## FD001

| Model | RMSE | MAE | NASA | rho | pred range | distinct | constant? |
|---|---|---|---|---|---|---|---|
| const_trainmean | 41.94 | 34.83 | 33,354 | n/a | 86.8-86.8 | 1 | **YES** |
| RandomForest | 17.96 | 13.27 | 761 | +0.813 | 7.8-123.4 | 100 | no |
| XGBoost | 15.95 | 11.71 | 477 | +0.844 | 5.4-122.3 | 100 | no |
| LSTM_original | 43.89 | 39.37 | 5,882 | n/a | 56.5-56.5 | 2 | **YES** |
| LSTM_fixed | 13.58 | 10.33 | 261 | +0.889 | 4.5-125.0 | 100 | no |

**Diagnosis.** The shipped training run ends after 30 epochs at train loss 2349.7 / val loss 2558.6. The target variance is 1747.2, so a network that had learned nothing beyond the training mean would sit at 1747.2. It is *above* that: it settled on the constant 56.5 while the training mean is 86.8, which alone accounts for an MSE of about 2664.1. The network did not merely fail to beat the mean - it converged to a constant *worse* than the mean, which is the signature of optimisation that never left its initialisation basin. The corrected run stops after 36 epochs at a validation MSE of 185.0 in RUL^2 units.


## FD003

| Model | RMSE | MAE | NASA | rho | pred range | distinct | constant? |
|---|---|---|---|---|---|---|---|
| const_trainmean | 43.70 | 36.44 | 57,687 | n/a | 93.1-93.1 | 1 | **YES** |
| RandomForest | 17.68 | 12.08 | 1,337 | +0.873 | 6.2-125.0 | 100 | no |
| XGBoost | 16.22 | 11.08 | 923 | +0.890 | 6.0-125.0 | 100 | no |
| LSTM_original | 39.87 | 34.70 | 6,063 | n/a | 66.3-66.3 | 1 | **YES** |
| LSTM_fixed | 13.81 | 9.73 | 403 | +0.928 | 0.0-125.0 | 98 | no |

**Diagnosis.** The shipped training run ends after 30 epochs at train loss 2306.9 / val loss 2183.0. The target variance is 1717.3, so a network that had learned nothing beyond the training mean would sit at 1717.3. It is *above* that: it settled on the constant 66.3 while the training mean is 93.1, which alone accounts for an MSE of about 2437.8. The network did not merely fail to beat the mean - it converged to a constant *worse* than the mean, which is the signature of optimisation that never left its initialisation basin. The corrected run stops after 48 epochs at a validation MSE of 159.7 in RUL^2 units.


## What this means for the paper

- The original LSTM emits a single constant, and one worse than the training mean. It is not a weak baseline, it is an untrained one, and reporting the LLM as 'competitive' with it compared two constants.
- Three defects account for it: an unscaled 0-125 target under MSE, a validation split that cuts a contiguous tail of overlapping windows from the same engines, and no early stopping. All three are corrected in `LSTM_fixed`.
- With repaired baselines the gap the paper has to report is far wider, which *strengthens* the revised thesis rather than weakening it.