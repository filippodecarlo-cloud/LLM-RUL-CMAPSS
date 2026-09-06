# The k x n few-shot grid, with a correct example sampler

FD001, 100 test engines per cell, llama3.1:8b, T=0.1. Examples are windows ending at an arbitrary cycle, stratified across the three RUL bins, so their labels span 0 to 125 rather than being uniformly zero.


## The grid

| k | n | RMSE | MAE | NASA | distinct | modal (share) | entropy | rho | p |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 51.56 | 44.93 | 13,722 | 1 | 42 (100%) | 0.00 | n/a | n/a |
| 5 | 5 | 50.00 | 43.65 | 11,556 | 4 | 45 (70%) | 1.16 | +0.226 | 0.024 |
| 10 | 5 | 49.70 | 43.57 | 11,065 | 2 | 45 (98%) | 0.14 | +0.227 | 0.023 |
| 3 | 15 | 51.02 | 44.13 | 13,614 | 5 | 42 (81%) | 0.93 | -0.082 | 0.416 |
| 5 | 15 | 50.66 | 44.24 | 12,409 | 4 | 42 (52%) | 1.14 | +0.057 | 0.573 |
| 10 | 15 | 41.96 | 38.03 | 5,537 | 1 | 62 (100%) | 0.00 | n/a | n/a |
| 3 | 30 | 50.47 | 44.20 | 12,207 | 5 | 45 (57%) | 1.24 | -0.032 | 0.754 |
| 5 | 30 | 49.38 | 38.11 | 328,259 | 4 | 105 (42%) | 1.65 | +0.048 | 0.634 |
| 10 | 30 | 64.51 | 50.55 | 1,502,461 | 1 | 125 (100%) | 0.00 | n/a | n/a |

**Against the no-skill constant (RMSE 41.94):** 0/9 cells beat it.

**Collapse:** 1 to 5 distinct values over 100 engines; entropy 0.00 to 1.65 bits against a ceiling of 6.64.

## Does the number of examples do anything?

| n | k=3 | k=5 | k=10 | spread across k |
|---|---|---|---|---|
| 5 | 51.56 | 50.00 | 49.70 | 1.86 |
| 15 | 51.02 | 50.66 | 41.96 | 9.06 |
| 30 | 50.47 | 49.38 | 64.51 | 15.13 |

The largest RMSE difference between k levels at a fixed window is **15.13**. The comparable quantity measured in P1.4 is the spread produced by changing only *which* examples are drawn, with k held fixed: a range of **16.34** RMSE over the stratified seeds (standard deviation 8.25). The spread across k does not exceed the spread produced by the draw, so this grid does not establish an effect of k on RMSE.

| k | mean modal share | mean distinct values |
|---|---|---|
| 3 | 79% | 3.7 |
| 5 | 55% | 4.0 |
| 10 | 99% | 1.3 |

Concentration does not follow RMSE. Mean modal share moves 79% (k=3) to 55% (k=5) to 99% (k=10), so the largest example count in this grid is also the most concentrated: at k=10 the model returns 1.3 distinct values on average over 100 engines. Once a cell is that concentrated its error is decided by which single value is returned - here 45 (RMSE 49.70), 62 (RMSE 41.96), 125 (RMSE 64.51). The best and the worst cell of the whole grid are both at this k, which is why the RMSE spread across k should not be read as an effect of k.

Kruskal-Wallis across k: H = 1.69, p = 0.430.

## Explanations, on these traces

- Directional claims: **22** over 900 traces, about 0.02 per trace
- Agreement with canonical physics: 40.9%
- Faithfulness to the window shown: 45.5%

**These two percentages rest on 22 claims and are not worth interpreting.** The finding here is the count itself. The few-shot template carries no sentence stating which sensors rise and which fall, and without it the model almost stops naming sensor directions at all. That matches the `nocue` control arm in Section 4.5, where removing the cue cut directional claims by 74%, and it means the explanation measures in Section 4.4 are driven by the zero-shot prompts, as reported there.

## What this means for the paper

- The ablation over k can be reported as an ablation, because every level now receives the number of examples it claims to.
- On RMSE, k has no effect distinguishable from the noise of which examples are drawn: the spread across k does not exceed the spread across draws.
- On concentration the picture is different, and is the more interesting one: the largest example count is the most concentrated, at one or two distinct values in all three windows. More examples did not restore regression.
- Fixing the sampler does not lift the collapse: every cell still returns a handful of distinct values, and none of them beats a constant.