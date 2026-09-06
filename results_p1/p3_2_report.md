# P3.2 - Multiplicity, the example-set association, and a common grid


## The family of rank-correlation tests

**The family is the 27 runs of the main prompting grid**: every configuration of dataset, prompting mode, number of examples and window length for which a trace file exists, including the replicate runs of the zero-shot configuration. One Spearman correlation between predicted and true RUL is computed per run, so the family is fixed by the design of the grid and not by which results turned out to be interesting.

**What was decided in advance and what was not.** The grid itself was pre-specified: the datasets, the values of k and n, and the decision to compute a rank correlation for every cell were all fixed before the runs. Nothing was added to the family afterwards. The correction across the family, and the decision to report rank correlation next to every error metric, were adopted after seeing that error alone could not separate ordering from anchoring, so they are post hoc, and the emphasis that Section 4.7 places on one particular run is post hoc by construction, since it is the run that stood out. That is why the whole family is printed here.

Of the 27 tests, **10 reach p < 0.05 uncorrected**, **3 survive Benjamini-Hochberg** at 5% and **1 survives Bonferroni**. The largest absolute correlation anywhere in the family is 0.312, whose square is 0.097. Signs are inconsistent: 22 positive and 5 negative. Benjamini-Hochberg is the primary correction, as the tests are one family of related hypotheses and false discovery is the relevant error rate; Bonferroni is given as the strict alternative.

| # | Dataset | Mode | k | n | Run | rho | 95% CI | p | p (BH) | p (Bonf.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FD003 | few-shot | 5 | 5 | run1 | +0.312 | [+0.13, +0.48] | 0.002 | 0.038 | 0.043 |
| 2 | FD001 | zero-shot | 0 | 15 | run1 | +0.294 | [+0.13, +0.44] | 0.003 | 0.038 | 0.082 |
| 3 | FD003 | few-shot | 10 | 30 | run1 | -0.284 | [-0.46, -0.09] | 0.004 | 0.038 | 0.113 |
| 4 | FD001 | few-shot | 10 | 5 | run1 | +0.261 | [+0.06, +0.45] | 0.009 | 0.058 | 0.234 |
| 5 | FD001 | zero-shot | 0 | 30 | run3 | +0.240 | [+0.03, +0.42] | 0.016 | 0.086 | 0.432 |
| 6 | FD003 | zero-shot | 0 | 30 | run1 | +0.230 | [+0.03, +0.42] | 0.021 | 0.096 | 0.579 |
| 7 | FD003 | few-shot | 3 | 5 | run1 | +0.219 | [+0.03, +0.40] | 0.029 | 0.096 | 0.770 |
| 8 | FD001 | zero-shot | 0 | 30 | run4 | +0.213 | [+0.04, +0.37] | 0.033 | 0.096 | 0.891 |
| 9 | FD001 | few-shot | 3 | 30 | run1 | -0.211 | [-0.39, -0.02] | 0.035 | 0.096 | 0.953 |
| 10 | FD001 | zero-shot | 0 | 30 | run2 | +0.210 | [+0.03, +0.37] | 0.036 | 0.096 | 0.964 |
| 11 | FD001 | few-shot | 10 | 15 | run1 | +0.184 | [-0.00, +0.36] | 0.067 | 0.163 | 1.000 |
| 12 | FD003 | few-shot | 10 | 15 | run1 | +0.174 | [-0.02, +0.36] | 0.083 | 0.186 | 1.000 |
| 13 | FD003 | few-shot | 10 | 5 | run1 | +0.159 | [-0.04, +0.34] | 0.113 | 0.235 | 1.000 |
| 14 | FD001 | few-shot | 5 | 15 | run1 | +0.141 | [-0.05, +0.33] | 0.162 | 0.313 | 1.000 |
| 15 | FD003 | few-shot | 5 | 30 | run1 | -0.129 | [-0.33, +0.08] | 0.201 | 0.362 | 1.000 |
| 16 | FD003 | few-shot | 3 | 30 | run1 | -0.125 | [-0.32, +0.07] | 0.216 | 0.365 | 1.000 |
| 17 | FD001 | zero-shot | 0 | 30 | run1 | +0.115 | [-0.10, +0.31] | 0.254 | 0.387 | 1.000 |
| 18 | FD003 | zero-shot | 0 | 5 | run1 | +0.114 | [-0.10, +0.31] | 0.258 | 0.387 | 1.000 |
| 19 | FD001 | few-shot | 5 | 5 | run1 | +0.070 | [-0.15, +0.29] | 0.489 | 0.696 | 1.000 |
| 20 | FD001 | few-shot | 3 | 5 | run1 | +0.055 | [-0.13, +0.24] | 0.584 | 0.746 | 1.000 |
| 21 | FD001 | zero-shot | 0 | 5 | run1 | +0.055 | [-0.15, +0.25] | 0.588 | 0.746 | 1.000 |
| 22 | FD001 | few-shot | 10 | 30 | run1 | +0.052 | [-0.15, +0.25] | 0.611 | 0.746 | 1.000 |
| 23 | FD003 | few-shot | 3 | 15 | run1 | +0.048 | [-0.16, +0.25] | 0.636 | 0.746 | 1.000 |
| 24 | FD003 | zero-shot | 0 | 15 | run1 | +0.039 | [-0.16, +0.23] | 0.701 | 0.789 | 1.000 |
| 25 | FD001 | few-shot | 3 | 15 | run1 | +0.028 | [-0.17, +0.22] | 0.785 | 0.821 | 1.000 |
| 26 | FD001 | few-shot | 5 | 30 | run1 | +0.027 | [-0.16, +0.21] | 0.791 | 0.821 | 1.000 |
| 27 | FD003 | few-shot | 5 | 15 | run1 | -0.011 | [-0.23, +0.20] | 0.911 | 0.911 | 1.000 |

For scale, Random Forest and the scaled-target LSTM reach rho of about +0.81 to +0.93 on the same engines, with p below 1e-24. The question is not whether any of these 27 correlations is nominally significant, but whether any is large enough to be prognostically useful, and none is.


## The example-set association, as an exploratory result

Seven example sets were run, each with k fixed at 5, so the only thing that changes between them is which examples the prompt carries. Across those seven points the mean prediction correlates with the mean RUL of the examples at **r = +0.941** (Pearson, n = 7, p = 0.0016), with a bootstrap 95% interval of [+0.823, +0.996]. Spearman's rho on the same points is +0.929 (p = 0.0025).

**This is an exploratory result and should be read as one.** Seven aggregate points is a small sample and an association of this kind does not establish that the examples cause the anchor; it establishes that the two move together across the sets that were run.

| Example set | Mean RUL of the examples | Mean prediction | r without this point |
|---|---|---|---|
| asis_seed1 | 0.0 | 26.3 | +0.924 |
| strat_seed3 | 39.4 | 31.6 | +0.955 |
| strat_seed2 | 60.0 | 55.1 | +0.955 |
| strat_seed1 | 62.8 | 43.7 | +0.953 |
| random_seed2 | 87.4 | 66.4 | +0.944 |
| random_seed1 | 99.2 | 63.8 | +0.935 |
| random_seed3 | 101.2 | 67.2 | +0.927 |

Leave-one-out, r ranges from **+0.924 to +0.955**. No single example set is carrying the association, including the one whose examples all sit at the end of life: with that point dropped, r is +0.924 on the remaining six.
Squared, r gives the share of the between-set variance in mean prediction that the mean example RUL accounts for. That is a statement about seven aggregate points, and it is not comparable with a squared rank correlation computed over 100 engines within a single run: different units of analysis, different sample sizes, different measures of association. The two should never be presented as shares of one variance.


## Concentration on a common one-cycle grid

The language models are instructed to answer with an integer, while Random Forest, XGBoost and the LSTM emit floating-point numbers, so two of their predictions are almost never exactly equal. Counting distinct values as they come therefore flatters the supervised models automatically. Here every prediction, from every model, is rounded to the nearest cycle before the concentration measures are computed. Two values count as the same when they round to the same integer.

| Dataset | Model | Engines | Distinct, raw | Distinct, 1-cycle grid | Modal share | Entropy (bits) | Ceiling |
|---|---|---|---|---|---|---|---|
| FD001 | const_trainmean | 100 | 1 | 1 | 100.0% | 0.0 | 6.64 |
| FD001 | RandomForest | 100 | 100 | 59 | 5.0% | 5.71 | 6.64 |
| FD001 | XGBoost | 100 | 100 | 63 | 5.0% | 5.781 | 6.64 |
| FD001 | LSTM_original | 100 | 4 | 1 | 100.0% | 0.0 | 6.64 |
| FD001 | LSTM_fixed | 100 | 100 | 62 | 6.0% | 5.722 | 6.64 |
| FD003 | const_trainmean | 100 | 1 | 1 | 100.0% | 0.0 | 6.64 |
| FD003 | RandomForest | 100 | 100 | 64 | 6.0% | 5.786 | 6.64 |
| FD003 | XGBoost | 100 | 100 | 69 | 5.0% | 5.917 | 6.64 |
| FD003 | LSTM_original | 100 | 1 | 1 | 100.0% | 0.0 | 6.64 |
| FD003 | LSTM_fixed | 100 | 98 | 64 | 5.0% | 5.782 | 6.64 |
| FD001 | llama3.1:8b (reference) | 100 | 2 | 2 | 85.0% | 0.61 | 6.64 |
| FD001 | llama3.1:8b-instruct-q8_0 | 100 | 5 | 5 | 80.0% | 0.993 | 6.64 |
| FD001 | mistral:7b | 100 | 4 | 4 | 50.0% | 1.234 | 6.64 |
| FD001 | qwen2.5:7b | 100 | 1 | 1 | 100.0% | 0.0 | 6.64 |
| FD001 | deepseek-r1:7b | 50 | 19 | 19 | 24.0% | 3.518 | 5.64 |

Rounding changes nothing that matters. On the common grid the supervised regressors still occupy 59 to 69 distinct values with a modal share of 5% to 6%, while the prompted models occupy 1 to 19 with a modal share of 24% to 100%. The gap is not an artefact of integer output against continuous output.

The two degenerate rows are worth keeping in view. The training-mean constant and the unscaled-target LSTM occupy one value each on this grid, which is what a collapsed predictor looks like, and the concentrated prompting runs sit next to them rather than next to the regressors.
