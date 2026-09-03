# P0.1b - Rank correlation: significance, effect size, scale reference

Family of 27 tests (all trace files, both datasets). Bootstrap: 10,000 resamples.


## Reference: the supervised baselines

| Dataset | Model | RMSE | NASA | rho | 95% CI | p | pred range | distinct preds |
|---|---|---|---|---|---|---|---|---|
| FD001 | RandomForest | 17.96 | 761 | +0.813 | [+0.70, +0.89] | 1.01e-24 | 7.805-123.440 | 100 |
| FD001 | LSTM | 43.83 | 5,857 | n/a | n/a | n/a | 56.691-56.691 | 1 |
| FD003 | RandomForest | 17.68 | 1,337 | +0.873 | [+0.80, +0.92] | 2.18e-32 | 6.195-125.000 | 100 |
| FD003 | LSTM | 40.07 | 5,872 | n/a | n/a | n/a | 65.318-65.318 | 1 |

> **The LSTM baseline is itself a constant predictor.** Its entire output range across the 100 test engines is below 1e-3, i.e. floating-point noise: it emits one number for every engine (FD001: 56.6908); (FD003: 65.3180). Spearman rho is therefore undefined for it and is reported as n/a - ranking that noise yields a large negative rho that means nothing. This invalidates the submitted paper's framing of the LLM as 'competitive with the LSTM baseline': the comparison was between two constant predictors, and the LLM's constant was the worse one.


## LLM configurations, corrected for multiple testing

| Dataset | Config | rho | 95% CI | p raw | p FDR | p Bonf | sig (FDR) |
|---|---|---|---|---|---|---|---|
| FD001 | ZS, n=15 | +0.294 | [+0.13, +0.44] | 0.003 | 0.038 | 0.082 | YES |
| FD001 | FS k=10, n=5 | +0.261 | [+0.06, +0.45] | 0.009 | 0.058 | 0.234 | no |
| FD001 | ZS, n=30 [run3] | +0.240 | [+0.03, +0.42] | 0.016 | 0.086 | 0.432 | no |
| FD001 | ZS, n=30 [run4] | +0.213 | [+0.04, +0.37] | 0.033 | 0.096 | 0.891 | no |
| FD001 | FS k=3, n=30 | -0.211 | [-0.39, -0.02] | 0.035 | 0.096 | 0.953 | no |
| FD001 | ZS, n=30 [run2] | +0.210 | [+0.03, +0.37] | 0.036 | 0.096 | 0.964 | no |
| FD001 | FS k=10, n=15 | +0.184 | [-0.00, +0.36] | 0.067 | 0.163 | 1.000 | no |
| FD001 | FS k=5, n=15 | +0.141 | [-0.05, +0.33] | 0.162 | 0.313 | 1.000 | no |
| FD001 | ZS, n=30 | +0.115 | [-0.10, +0.31] | 0.254 | 0.387 | 1.000 | no |
| FD001 | FS k=5, n=5 | +0.070 | [-0.15, +0.29] | 0.489 | 0.696 | 1.000 | no |
| FD001 | FS k=3, n=5 | +0.055 | [-0.13, +0.24] | 0.584 | 0.746 | 1.000 | no |
| FD001 | ZS, n=5 | +0.055 | [-0.15, +0.25] | 0.588 | 0.746 | 1.000 | no |
| FD001 | FS k=10, n=30 | +0.052 | [-0.15, +0.25] | 0.611 | 0.746 | 1.000 | no |
| FD001 | FS k=3, n=15 | +0.028 | [-0.17, +0.22] | 0.785 | 0.821 | 1.000 | no |
| FD001 | FS k=5, n=30 | +0.027 | [-0.16, +0.21] | 0.791 | 0.821 | 1.000 | no |
| FD003 | FS k=5, n=5 | +0.312 | [+0.13, +0.48] | 0.002 | 0.038 | 0.043 | YES |
| FD003 | FS k=10, n=30 | -0.284 | [-0.46, -0.09] | 0.004 | 0.038 | 0.113 | YES |
| FD003 | ZS, n=30 | +0.230 | [+0.03, +0.42] | 0.021 | 0.096 | 0.579 | no |
| FD003 | FS k=3, n=5 | +0.219 | [+0.03, +0.40] | 0.029 | 0.096 | 0.770 | no |
| FD003 | FS k=10, n=15 | +0.174 | [-0.02, +0.36] | 0.083 | 0.186 | 1.000 | no |
| FD003 | FS k=10, n=5 | +0.159 | [-0.04, +0.34] | 0.113 | 0.235 | 1.000 | no |
| FD003 | FS k=5, n=30 | -0.129 | [-0.33, +0.08] | 0.201 | 0.362 | 1.000 | no |
| FD003 | FS k=3, n=30 | -0.125 | [-0.32, +0.07] | 0.216 | 0.365 | 1.000 | no |
| FD003 | ZS, n=5 | +0.114 | [-0.10, +0.31] | 0.258 | 0.387 | 1.000 | no |
| FD003 | FS k=3, n=15 | +0.048 | [-0.16, +0.25] | 0.636 | 0.746 | 1.000 | no |
| FD003 | ZS, n=15 | +0.039 | [-0.16, +0.23] | 0.701 | 0.789 | 1.000 | no |
| FD003 | FS k=5, n=15 | -0.011 | [-0.23, +0.20] | 0.911 | 0.911 | 1.000 | no |

## Verdict

- Nominally significant (p<0.05, uncorrected): **10/27**. Expected by chance alone at this family size: 1.4.
- Surviving Benjamini-Hochberg FDR at 5%: **3/27**.
- Surviving Bonferroni at 5%: **1/27**.
- Of the FDR-surviving ones, 2 are positive and 1 are **negative** (the model ranks engines backwards). A capability cannot change sign across configurations of the same model; an artefact can.
- Largest |rho| over any LLM configuration: **0.312** (explains 9.7% of rank variance), against **0.873** for Random Forest on the same engines.
- Bootstrap 95% CI includes zero in **17/27** configurations.