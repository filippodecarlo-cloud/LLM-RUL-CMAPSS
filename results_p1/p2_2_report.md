# P2.2 - Multiple operating conditions (FD002, FD004)

Zero-shot, n=30, T=0.1, shipped template, every test engine. Sensors scaled **within each of the 6 operating-condition clusters** (k-means on the three setting channels, fitted on train) rather than globally - on FD001/FD003 a single condition made that distinction irrelevant, here it is essential and it favours the baselines.


## FD002

| Model | n | RMSE | MAE | NASA | rho | distinct | modal (share) | entropy |
|---|---|---|---|---|---|---|---|---|
| llama3.1:8b zero-shot n=30 | 259 | 63.20 | 52.99 | 146,694 | +0.272 | 4 | 23 (83%) | 0.72 |
| constant = train mean | 259 | 44.93 | 38.14 | 115,590 | n/a | 1 | 87 (100%) | -0.00 |
| RandomForest | 259 | 17.63 | 13.39 | 1,699 | +0.871 | 257 | 12 (1%) | 8.00 |
| XGBoost | 259 | 17.34 | 12.80 | 1,500 | +0.880 | 259 | 0 (0%) | 8.02 |
| LSTM (repaired) | 259 | 16.95 | 10.99 | 3,405 | +0.891 | 259 | 4 (0%) | 8.02 |

## FD004

| Model | n | RMSE | MAE | NASA | rho | distinct | modal (share) | entropy |
|---|---|---|---|---|---|---|---|---|
| llama3.1:8b zero-shot n=30 | 248 | 66.56 | 55.94 | 172,813 | +0.146 | 4 | 23 (80%) | 0.79 |
| constant = train mean | 248 | 45.57 | 38.06 | 164,681 | n/a | 1 | 93 (100%) | -0.00 |
| RandomForest | 248 | 23.44 | 16.77 | 4,238 | +0.777 | 246 | 114 (1%) | 7.94 |
| XGBoost | 248 | 24.79 | 16.61 | 8,753 | +0.771 | 240 | 125 (4%) | 7.84 |
| LSTM (repaired) | 248 | 20.55 | 12.95 | 9,286 | +0.839 | 240 | 125 (4%) | 7.84 |