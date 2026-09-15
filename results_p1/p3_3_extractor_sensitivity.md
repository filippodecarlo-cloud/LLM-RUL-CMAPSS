# P3.3d - The cue comparisons without clauses about deviations, variance or levels

The reading guideline classes as unspecified a mention whose level or variability is described without a direction, and in the validation sample the author labelled every clause about deviations, variance or a level relative to normal that way. The extractor counts such a clause with the sign of the nearby word. This recomputes the comparisons of Section 4.4 with and without the clauses that use that wording (pattern `deviation|variance|fluctuat|than normal|lower values|higher values`), with the cluster bootstrap and permutation of p3_1_cluster_inference.py and its seed. The pattern is a lexical proxy for the guideline's rule, fixed after the reading; the analysis is a sensitivity check, not a replacement for the figures of record.

In the validation sample 16 pairs have a matched clause that uses that wording, and the author labelled 16 of them unspecified.

| Claims | Total | With that wording | Share |
|---|---|---|---|
| named sensors | 3,536 | 141 | 4.0% |
| s4 and s7 | 280 | 136 | 48.6% |
| zero-shot runs | 3,434 | 116 | 3.4% |
| few-shot runs | 382 | 161 | 42.1% |

## All claims (the figures of record)

3,816 claims: 3,536 about the five sensors the zero-shot template names, 280 about s4 and s7; 3,434 from zero-shot runs, 382 from few-shot runs.

| Contrast | Side A | Side B | Difference | 95% CI (cluster) | p (cluster) |
|---|---|---|---|---|---|
| Reference agreement, named sensors vs s4/s7 | 94.5% | 51.4% | 43.0 | [28.6, 60.7] | 0.0484 (floor 0.048) |
| Reference agreement, zero-shot vs few-shot | 94.6% | 61.5% | 33.1 | [23.8, 44.4] | 0.0020 |
| Faithfulness to the window, named sensors vs s4/s7 | 42.2% | 49.3% | -7.1 | [-14.4, 1.3] | |
| Faithfulness vs base rate, named sensors | 42.2% | 41.3% | +0.8 | [-0.2, +2.0] | |
| Faithfulness vs base rate, s4/s7 | 49.3% | 50.0% | -0.7 | [-12.1, +11.3] | |
| Faithfulness vs base rate, all claims | 42.7% | 42.0% | +0.7 | [-0.5, +2.1] | |

Agreement with the reference direction over all 3,816 claims: 91.3%.


## Without clauses about deviations, variance or levels

3,539 claims: 3,395 about the five sensors the zero-shot template names, 144 about s4 and s7; 3,318 from zero-shot runs, 221 from few-shot runs.

| Contrast | Side A | Side B | Difference | 95% CI (cluster) | p (cluster) |
|---|---|---|---|---|---|
| Reference agreement, named sensors vs s4/s7 | 94.4% | 78.5% | 16.0 | [8.8, 23.6] | 0.1461 (floor 0.048) |
| Reference agreement, zero-shot vs few-shot | 95.0% | 76.0% | 18.9 | [11.9, 28.2] | 0.0020 |
| Faithfulness to the window, named sensors vs s4/s7 | 42.1% | 54.9% | -12.8 | [-22.3, -2.7] | |
| Faithfulness vs base rate, named sensors | 42.1% | 41.2% | +0.9 | [-0.3, +2.0] | |
| Faithfulness vs base rate, s4/s7 | 54.9% | 54.2% | +0.7 | [-7.7, +12.7] | |
| Faithfulness vs base rate, all claims | 42.6% | 41.8% | +0.8 | [-0.3, +2.0] | |

Agreement with the reference direction over all 3,539 claims: 93.8%.
