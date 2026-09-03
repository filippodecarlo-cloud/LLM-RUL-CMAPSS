# P0.1 - No-skill baselines and rank-correlation audit

Post-hoc computation on the existing trace files (24 unique configurations + 3 zero-shot replicates). No new inference.


## FD001

Test RUL (capped at 125): n=100, mean=74.45, median=86.0, sd=40.28, range 7-125. Train-label mean RUL = 86.83.


### No-skill / constant predictors

| Predictor | c | RMSE | MAE | NASA |
|---|---|---|---|---|
| constant = TRAIN mean RUL | 86.8 | 41.94 | 34.83 | 33,354 |
| constant = TRAIN median RUL | 103.0 | 49.20 | 38.01 | 166,491 |
| constant = TEST mean RUL (oracle) | 74.5 | 40.07 | 35.85 | 10,579 |
| constant = TEST median RUL (oracle) | 86.0 | 41.70 | 34.83 | 30,753 |
| constant = 50 | 50.0 | 46.94 | 41.51 | 7,992 |
| constant = RUL_CAP/2 (62.5) | 62.5 | 41.82 | 37.92 | 5,570 |
| constant = RMSE-optimal (oracle) | 74.0 | 40.08 | 35.91 | 10,197 |
| constant = NASA-optimal (oracle) | 61.0 | 42.27 | 38.25 | 5,506 |
| random uniform U(0,125), mean of 1000 | - | 55.02 | 45.13 | 155,400 |

### LLM configurations

| Config | RMSE | MAE | NASA | rho | p | distinct | modal (share) | sd_pred/sd_true | beats train-mean? |
|---|---|---|---|---|---|---|---|---|---|
| FS k=3, n=5 | 62.50 | 52.38 | 49,741 | +0.055 | 0.584 | 3 | 23 (82%) | 0.19 | no |
| FS k=3, n=15 | 55.31 | 47.98 | 25,296 | +0.028 | 0.785 | 3 | 45 (64%) | 0.25 | no |
| FS k=3, n=30 | 50.54 | 40.70 | 133,633 | -0.211 | 0.035 | 11 | 95 (39%) | 0.43 | no |
| FS k=5, n=5 | 61.78 | 52.38 | 41,801 | +0.070 | 0.489 | 3 | 23 (76%) | 0.22 | no |
| FS k=5, n=15 | 53.96 | 46.99 | 18,572 | +0.141 | 0.162 | 5 | 45 (57%) | 0.24 | no |
| FS k=5, n=30 | 46.93 | 37.59 | 119,410 | +0.027 | 0.791 | 8 | 95 (40%) | 0.38 | no |
| FS k=10, n=5 | 60.31 | 50.68 | 43,627 | +0.261 | 0.009 | 7 | 23 (53%) | 0.28 | no |
| FS k=10, n=15 | 53.65 | 46.35 | 23,307 | +0.184 | 0.067 | 7 | 45 (37%) | 0.29 | no |
| FS k=10, n=30 | 45.60 | 36.32 | 79,568 | +0.052 | 0.611 | 8 | 95 (34%) | 0.29 | no |
| ZS, n=5 | 60.90 | 51.07 | 41,333 | +0.055 | 0.588 | 3 | 23 (68%) | 0.22 | no |
| ZS, n=15 | 61.91 | 52.01 | 45,218 | +0.294 | 0.003 | 2 | 23 (84%) | 0.17 | no |
| ZS, n=30 | 62.83 | 52.58 | 48,469 | +0.115 | 0.254 | 2 | 23 (85%) | 0.17 | no |
| ZS, n=30 [run2] | 61.81 | 51.34 | 45,488 | +0.210 | 0.036 | 2 | 23 (81%) | 0.19 | no |
| ZS, n=30 [run3] | 61.74 | 51.82 | 43,301 | +0.240 | 0.016 | 2 | 23 (81%) | 0.19 | no |
| ZS, n=30 [run4] | 61.52 | 51.22 | 45,987 | +0.213 | 0.033 | 3 | 23 (80%) | 0.19 | no |

**Summary FD001:** 0/15 configurations beat the train-mean constant on RMSE; 6/15 reach p<0.05 on Spearman rho (max |rho| = 0.294).


## FD003

Test RUL (capped at 125): n=100, mean=73.76, median=77.5, sd=39.36, range 6-125. Train-label mean RUL = 93.14.


### No-skill / constant predictors

| Predictor | c | RMSE | MAE | NASA |
|---|---|---|---|---|
| constant = TRAIN mean RUL | 93.1 | 43.70 | 36.44 | 57,687 |
| constant = TRAIN median RUL | 123.0 | 62.92 | 49.86 | 1,138,745 |
| constant = TEST mean RUL (oracle) | 73.8 | 39.17 | 34.26 | 9,433 |
| constant = TEST median RUL (oracle) | 77.5 | 39.34 | 34.20 | 12,870 |
| constant = 50 | 50.0 | 45.81 | 38.76 | 8,547 |
| constant = RUL_CAP/2 (62.5) | 62.5 | 40.75 | 35.29 | 5,587 |
| constant = RMSE-optimal (oracle) | 74.0 | 39.17 | 34.26 | 9,611 |
| constant = NASA-optimal (oracle) | 62.0 | 40.89 | 35.38 | 5,574 |
| random uniform U(0,125), mean of 1000 | - | 54.34 | 44.50 | 148,440 |

### LLM configurations

| Config | RMSE | MAE | NASA | rho | p | distinct | modal (share) | sd_pred/sd_true | beats train-mean? |
|---|---|---|---|---|---|---|---|---|---|
| FS k=3, n=5 | 55.96 | 46.52 | 32,050 | +0.219 | 0.029 | 5 | 23 (49%) | 0.27 | no |
| FS k=3, n=15 | 51.71 | 42.88 | 20,279 | +0.048 | 0.636 | 3 | 45 (73%) | 0.22 | no |
| FS k=3, n=30 | 46.38 | 38.93 | 108,143 | -0.125 | 0.216 | 8 | 95 (33%) | 0.42 | no |
| FS k=5, n=5 | 55.98 | 46.75 | 32,902 | +0.312 | 0.002 | 3 | 23 (57%) | 0.27 | no |
| FS k=5, n=15 | 51.36 | 42.91 | 19,888 | -0.011 | 0.911 | 5 | 45 (74%) | 0.21 | no |
| FS k=5, n=30 | 45.71 | 38.45 | 55,700 | -0.129 | 0.201 | 10 | 95 (29%) | 0.44 | no |
| FS k=10, n=5 | 73.02 | 62.27 | 149,710 | +0.159 | 0.113 | 5 | 10 (92%) | 0.18 | no |
| FS k=10, n=15 | 52.26 | 43.63 | 21,287 | +0.174 | 0.083 | 3 | 30 (49%) | 0.22 | no |
| FS k=10, n=30 | 47.58 | 40.51 | 71,048 | -0.284 | 0.004 | 8 | 95 (33%) | 0.45 | no |
| ZS, n=5 | 58.56 | 49.13 | 37,418 | +0.114 | 0.258 | 2 | 23 (61%) | 0.24 | no |
| ZS, n=15 | 60.42 | 50.55 | 47,525 | +0.039 | 0.701 | 3 | 23 (71%) | 0.22 | no |
| ZS, n=30 | 56.30 | 46.57 | 33,469 | +0.230 | 0.021 | 2 | 23 (51%) | 0.24 | no |

**Summary FD003:** 0/12 configurations beat the train-mean constant on RMSE; 4/12 reach p<0.05 on Spearman rho (max |rho| = 0.312).
