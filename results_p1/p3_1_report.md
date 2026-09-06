# P3.1 - Cluster-aware inference for the faithfulness analysis

Every inferential statement in the faithfulness sections, recomputed with the dependence between claims respected. Claims are nested in responses, responses in engines, and engines are asked about in many configurations, so a test that treats each claim as an independent observation reports a p-value that is too small. Intervals are percentile intervals from 10,000 cluster bootstrap resamples, computed once clustering on engine and once on configuration, with the wider of the two reported. Permutation p-values use 10,000 cluster-level permutations and are reported as the proportion of permutations at least as extreme, so the smallest attainable value is 0.00010.

Main analysis: 3,816 directional claims, 200 engine clusters, 25 configuration clusters.


## Cued sensors against uncued sensors

Claims about cued sensors agree with the canonical direction 94.5% of the time (n = 3,536) and claims about uncued sensors 51.4% (n = 280). The difference is 43.0 points, 95% CI [28.6, 60.7].

The published chi-square for this contrast was 599.3 with p of order 1e-132, computed as if the 3,816 claims were independent. Permuting which sensors count as cued, which is the level at which the cue is actually assigned, gives p = 0.0484. There are only seven sensors, so this design cannot produce a p below 0.048 however large the effect: with a partition this coarse the p-value carries little information and the difference in proportions with its interval is what should be reported.


## Prompts that carry the cue against prompts that do not

The zero-shot template carries the cue sentence and the few-shot template does not. Canonical agreement is 94.6% (n = 3,434) against 61.5% (n = 382), a difference of 33.1 points, 95% CI [23.8, 44.4]. Permuting the template label between the 25 configurations gives p = 0.0020.


## Faithfulness against the rate obtained by ignoring the input

Input faithfulness is 42.7% while the canonical direction is actually present in the window 42.0% of the time, so a model that ignored the input and always stated the canonical direction would score the second number. The gap is +0.7 points, 95% CI [-0.5, +2.1]. The interval contains zero, so on these claims the explanations are indistinguishable from a rule that never looks at the data. Both percentages are computed on the same 3,816 rows: over all 18,900 opportunities the same no-look rule scores 46.7%, but faithfulness is undefined wherever the model stated no direction, so that figure has a different denominator and is not the comparison to make.

## Association between the stated direction and the window, sensor by sensor

Six of the seven sensors admit a test; for s15 the model states one direction in every claim, so no association can be computed. Benjamini-Hochberg is applied across the 6 computable tests, which was missing before.

| Sensor | Claims | Cramer's V | 95% CI (cluster) | p | p (BH) |
|---|---|---|---|---|---|
| s11 | 947 | 0.054 | [0.013, 0.168] | 0.3202 | 0.6205 |
| s12 | 527 | 0.197 | [0.098, 0.313] | 0.1964 | 0.6205 |
| s14 | 877 | 0.065 | [0.014, 0.151] | 0.2449 | 0.6205 |
| s15 | 306 | not defined | not defined | not defined | not defined |
| s4 | 10 | 0.167 | [0.048, 0.845] | 0.8343 | 0.8343 |
| s7 | 270 | 0.101 | [0.026, 0.273] | 0.5170 | 0.6205 |
| s9 | 879 | 0.047 | [0.012, 0.137] | 0.4563 | 0.6205 |

After correction across the six tests, no sensor retains a detectable association. The largest association anywhere is Cramer's V of 0.20, which is weak on any reading.


## Control arms, paired by engine

- **cue removed**: canonical agreement moves +12.5 points on 48 paired claims across 45 engines, 95% CI [+0.0, +26.5], engine-level sign-flip p = 0.1481.
- **cue inverted**: canonical agreement moves -34.6 points on 283 paired claims across 100 engines, 95% CI [-42.9, -26.3], engine-level sign-flip p = 0.0001.
- **trends reversed**: canonical agreement moves -0.7 points on 298 paired claims across 100 engines, 95% CI [-2.7, +1.0], engine-level sign-flip p = 0.7554.

The published chi-square for the inverted cue was 103.1 with p of order 1e-24. The paired, engine-clustered version gives the same direction and a similar magnitude, with an interval attached and without assuming independence.


## The trend-reversal result, with an interval

Both arms make a directional claim for 298 (engine, sensor) pairs. The trend actually shown reversed for 253 of them, and the model's stated direction changed for 6, or 2.4%. Clustering on engine (100 clusters) the 95% interval is [0.4, 4.7]; the unclustered Wilson interval is [1.1, 5.1].

**On the selection.** Restricting to pairs whose shown trend reversed keeps 253 of 298 pairs, 84.9%. The restriction is necessary, since a pair whose trend did not reverse cannot test whether the claim follows the trend, but the figure describes only those pairs and should be read that way. It also bears only on what the model says about trend direction, and does not establish that no other information in the window reached the prediction.
