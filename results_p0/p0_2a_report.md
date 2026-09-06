# P0.2a - Textbook agreement vs input faithfulness

All 27 trace files, 18,900 sensor-trace opportunities, 3,816 explicit directional claims.

A window counts as flat when |last - first| < 0.01 in the normalised units printed in the prompt.


## Headline

| Measure | Value |
|---|---|
| Directional claims scored | 3,816 |
| **Textbook agreement** (what the paper reported) | **91.3%** |
| **Input faithfulness** (claim matches the window shown) | **42.7%** |
| Base rate: canonical direction present, same claims | 42.0% |
| (the same rule over all 18,900 opportunities, different denominator, not comparable) | 46.7% |

## Control 1 - sensors the prompt names vs sensors it does not

The cue sentence names s9, s11, s12, s14, s15 and is silent about s4 and s7, whose canonical directions are just as established. s4 and s7 are uncued in every prompt variant, so all modes are pooled here.

| Sensor set | Directional claims | Textbook agreement | Input faithfulness |
|---|---|---|---|
| CUED (s9,s11,s12,s14,s15) | 3,536 | 94.5% | 42.2% |
| UNCUED (s4,s7) | 280 | 51.4% | 49.3% |

chi2 = 599.3, p = 2.33e-132 for the cued/uncued difference in textbook agreement. Cueing a sensor moves agreement with the canonical direction; it does not move agreement with the data.


## Control 2 - zero-shot prompt (carries the cue) vs few-shot prompt (does not)

| Prompt | Directional claims | Textbook agreement | Input faithfulness |
|---|---|---|---|
| zero-shot (cue present) | 3,434 | 94.6% | 42.1% |
| few-shot (no cue sentence) | 382 | 61.5% | 47.9% |

## Per sensor

| Sensor | In prompt? | Canonical | Claims | Textbook agr. | Input faith. | Canonical direction present in window |
|---|---|---|---|---|---|---|
| s4 | NO | increase | 10 | 60.0% | 50.0% | 60.0% |
| s7 | NO | decrease | 270 | 51.1% | 49.3% | 49.6% |
| s9 | yes | decrease | 879 | 95.6% | 27.1% | 26.5% |
| s11 | yes | increase | 947 | 98.4% | 62.1% | 62.2% |
| s12 | yes | increase | 527 | 78.7% | 47.2% | 41.9% |
| s14 | yes | decrease | 877 | 96.6% | 27.7% | 27.9% |
| s15 | yes | increase | 306 | 100.0% | 56.9% | 56.9% |

## By prompt window length

| n cycles | Claims | Textbook agr. | Input faith. |
|---|---|---|---|
| 5 | 929 | 91.1% | 41.3% |
| 15 | 1,011 | 88.0% | 44.1% |
| 30 | 1,876 | 93.2% | 42.6% |

## Distribution of claims vs reality

| Sensor | Model says increase / decrease | Window actually increase / decrease / flat |
|---|---|---|
| s4 | 6 / 4 | 1677 / 853 / 170 |
| s7 | 132 / 138 | 1071 / 1273 / 356 |
| s9 | 39 / 840 | 1460 / 707 / 533 |
| s11 | 932 / 15 | 1651 / 893 / 156 |
| s12 | 415 / 112 | 1141 / 1203 / 356 |
| s14 | 30 / 847 | 1215 / 751 / 734 |
| s15 | 306 / 0 | 1624 / 910 / 166 |

## Robustness to the flat-window threshold

The one free parameter above is the threshold below which a window counts as flat. At eps = 0 the judgement is a pure sign test with no flat class.

| eps | Input faithfulness | Base rate (always assert canonical) | Windows called flat |
|---|---|---|---|
| 0.000 | 50.1% | 49.7% | 0 |
| 0.005 | 46.9% | 46.1% | 1,087 |
| 0.010 | 42.7% | 42.0% | 2,471 |
| 0.020 | 35.7% | 34.7% | 5,112 |
| 0.050 | 23.1% | 22.1% | 10,245 |

At every threshold the explanations are no more accurate about the input than a rule that ignores the input entirely and always asserts the canonical direction. The gap never exceeds about one point in either direction, well inside the interval reported in P3.1, so the two are indistinguishable. The conclusion does not depend on the threshold.

## Does the claim carry any information about the window?

Test of independence between the direction claimed and the direction actually present, per sensor. Cramer's V is the effect size (0 = the claim tells you nothing about the data); MI is mutual information in bits.

| Sensor | Claims | chi2 | p | Cramer's V | MI (bits) | Faithfulness vs base rate |
|---|---|---|---|---|---|---|
| s4 | 10 | 0.3 | 0.870 | 0.167 | 0.0200 | 50.0% vs 60.0%  |
| s7 | 270 | 2.8 | 0.250 | 0.101 | 0.0074 | 49.3% vs 49.6%  |
| s9 | 879 | 1.9 | 0.385 | 0.047 | 0.0015 | 27.1% vs 26.5%  |
| s11 | 947 | 2.7 | 0.256 | 0.054 | 0.0017 | 62.1% vs 62.2%  |
| s12 | 527 | 20.4 | 0.000 | 0.197 | 0.0318 | 47.2% vs 41.9%  |
| s14 | 877 | 3.7 | 0.158 | 0.065 | 0.0028 | 27.7% vs 27.9%  |
| s15 | 306 | n/a | n/a | 0.000 | 0.0000 | 56.9% vs 56.9% model never varies its claim - independence is degenerate |

## Reading

- The paper's explainability figure is reproduced here as **91.3%** textbook agreement.
- Against the sensor windows actually placed in the prompts, the same claims are right **42.7%** of the time.
- The gap is the size of the prompt-leakage effect: the explanations track the canonical narrative, not the data the model was given.