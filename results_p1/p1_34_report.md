# P1.3 / P1.4 - Does the example set explain the anchor?

FD001, k=5, n=5, 100 test engines, llama3.1:8b, T=0.1. Only the few-shot example set changes. Test-set mean RUL = **74.5**.


| Run | n | RMSE | MAE | NASA | mean pred | rho (p) | distinct | modal (share) | entropy |
|---|---|---|---|---|---|---|---|---|---|
| published k=5, n=5 (same prompt as asis) | 100 | 61.78 | 52.38 | 41,801 | 27.9 | +0.070 (0.49) | 3 | 23 (76%) | 1.03 |
| asis_seed1 | 100 | 62.49 | 52.53 | 48,690 | 26.3 | +0.123 (0.22) | 3 | 23 (84%) | 0.79 |
| random_seed1 | 100 | 43.68 | 37.67 | 10,901 | 63.8 | +0.085 (0.40) | 11 | 53 (26%) | 2.87 |
| random_seed2 | 100 | 41.27 | 36.65 | 7,123 | 66.4 | +0.061 (0.54) | 4 | 73 (65%) | 1.20 |
| random_seed3 | 100 | 38.68 | 33.41 | 5,768 | 67.2 | +0.265 (0.01) | 11 | 80 (43%) | 2.15 |
| strat_seed1 | 100 | 50.60 | 44.07 | 14,098 | 43.7 | +0.206 (0.04) | 5 | 45 (67%) | 1.28 |
| strat_seed2 | 100 | 44.37 | 35.21 | 22,305 | 55.1 | +0.254 (0.01) | 8 | 80 (32%) | 2.41 |
| strat_seed3 | 100 | 60.71 | 49.23 | 50,101 | 31.6 | +0.142 (0.16) | 4 | 23 (85%) | 0.78 |

**Replication control.** `asis` reproduces the shipped prompt and lands at RMSE 62.49 / modal 23 against the published run's 61.78 / 23.


## P1.3 - the mechanistic hypothesis

P1.0 showed the shipped prompt contains one example labelled RUL 0. If that is what drags the predictions down, then real examples spanning 0-125 should raise them towards the test mean of 74.5.

- Mean prediction, shipped example set: **26.3**
- Mean prediction, genuinely stratified examples: **43.5** (mean of 3 seeds)
- Shift: **+17.2 RUL**, against a gap to the test mean of +48.2

The direction of the shift **supports** the hypothesis that the RUL-0 example set is what anchors the predictions low.
- Collapse itself: distinct values go from 3 (shipped) to 5.7 (stratified, mean of seeds); entropy 0.79 -> 1.49 bits against a ceiling of 6.64. Fixing the examples does **not** restore regression.

## Stratified vs unstratified

- stratified: RMSE 51.89 +/- 8.25, mean pred 43.5
- random:     RMSE 41.21 +/- 2.50, mean pred 65.8
- Mann-Whitney on RMSE: U = 9, p = 0.100

## P1.4 - sampling variability the paper never reported

| Condition | seeds | RMSE mean | RMSE sd | RMSE range | modal values |
|---|---|---|---|---|---|
| stratified | 3 | 51.89 | 8.25 | 44.37-60.71 | [23, 45, 80] |
| random | 3 | 41.21 | 2.50 | 38.68-43.68 | [53, 73, 80] |

The paper reported one draw of the example set with no variability estimate. Any k-to-k difference smaller than this spread is not interpretable.

## Why a lower RMSE here is not predictive skill

Random example sets draw mostly high-RUL windows - the target is capped at 125 - so they anchor the model near the test mean of 74.5 and the RMSE falls. That is a constant being relocated, not engines being told apart.

**2 run(s) come in under the no-skill constant (RMSE 41.94):**

| Run | RMSE | mean pred | rho | p | p (BH-FDR) | distinct | rank var. explained |
|---|---|---|---|---|---|---|---|
| random_seed2 | 41.27 | 66.4 | +0.061 | 0.54 | 0.545 | 4 | 0.4% |
| random_seed3 | 38.68 | 67.2 | +0.265 | 0.01 | 0.038 | 11 | 7.0% |

This has to be stated carefully. The strongest of them reaches rho = +0.265 (p = 0.01, BH-adjusted 0.038), so it is **not** literally zero and it would be wrong to write that no configuration orders the engines at all. Three things keep it from being evidence of prognostic ability:
1. **It explains 7% of the rank variance**, against 66% for Random Forest (+0.813) and 79% for the repaired LSTM (+0.889) on the same 100 engines.
2. **It is unstable across seeds of its own condition**: the random runs span rho +0.061 to +0.265 with nothing changed but which examples were drawn. A capability does not come and go with the draw.
3. **The RMSE gain is anchoring, not discrimination.** Its mean prediction is 67.2 against a test mean of 74.5, on 11 distinct values over 100 engines (2.15 bits of 6.64).

So a configuration can beat the no-skill constant on RMSE while carrying almost no prognostic content - which is exactly why the rewritten paper must report a rank correlation next to every error metric, and why RMSE alone was never going to settle this question.

Across all completed runs the largest |rho| is **0.265**; Random Forest reaches +0.813 and the repaired LSTM +0.889 on the same 100 engines.