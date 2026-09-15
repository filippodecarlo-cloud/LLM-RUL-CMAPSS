# P3.3 - Validating the directional-claim extractor

A stratified sample of 139 (response, sensor) pairs was labelled by the author, in shuffled order, without the extractor's verdict. Each case showed the full response with every occurrence of the sensor's name highlighted, a chart of the values of that sensor in the prompt, and the whole prompt on request. When the sensor's name did not occur in the response the screen said so, so for those cases the reader's task was to check whether the sensor was referred to in some other form. Each pair was labelled increase, decrease, stable, unspecified, absent or ambiguous following the guideline in `p3_3_reading_guideline.md`. The sample is stratified over the extractor's own verdict so that both kinds of error can be measured; the raw accuracy in the sample is therefore not the accuracy in the corpus, and the corpus figures below are reweighted by how often each verdict occurs.


## Ambiguous cases

0 of 139 were labelled ambiguous and are excluded from the scoring. They are listed so that their kind can be seen:


## Confusion matrix, sampled cases

Rows are the extractor, columns the author's reading.

| extractor \ author | increase | decrease | stable | unspecified | absent | total |
|---|---|---|---|---|---|---|
| **increase** | 34 | 1 | 0 | 5 | 0 | 40 |
| **decrease** | 4 | 31 | 2 | 3 | 0 | 40 |
| **stable** | 1 | 1 | 7 | 0 | 0 | 9 |
| **unspecified** | 9 | 0 | 0 | 21 | 0 | 30 |
| **absent** | 0 | 0 | 0 | 2 | 18 | 20 |

## Precision and recall, sampled cases

| Class | Precision | Recall | Sampled n (author) |
|---|---|---|---|
| increase | 0.850 | 0.708 | 48 |
| decrease | 0.775 | 0.939 | 33 |
| stable | 0.778 | 0.778 | 9 |
| unspecified | 0.700 | 0.677 | 31 |
| absent | 0.900 | 1.000 | 18 |

Treating increase and decrease together as **a directional claim**, precision is **0.875** and recall **0.864** in the sample. Of the 70 cases where both the extractor and the author see a directional claim, the extractor gets the sign right in 65, **92.9%**.


## Reweighted to the corpus

| Extractor verdict | Corpus count | Sampled | Agrees with the author | Implied corpus accuracy |
|---|---|---|---|---|
| increase | 1,860 | 40 | 34 | 85.0% |
| decrease | 1,956 | 40 | 31 | 77.5% |
| stable | 9 | 9 | 7 | 77.8% |
| unspecified | 1,624 | 30 | 21 | 70.0% |
| absent | 13,451 | 20 | 18 | 90.0% |

Weighting each verdict by how often it occurs, the extractor agrees with the author on **86.5%** of all 18,900 pairs. The absent class carries 71% of that weight, which is why each of its sampled responses was read in full. Since the screen said when the sensor's name did not occur, agreement on that class measures only whether the sensor was referred to in some other form.


The analysis itself uses one distinction only: whether a pair carries an increase, a decrease, or no directional claim. Stable, unspecified and absent all enter it as no claim, so a disagreement among them moves no figure in the paper. On that three-way distinction:

| Extractor verdict | Sampled | Agrees with the author | Implied corpus accuracy |
|---|---|---|---|
| increase | 40 | 34 | 85.0% |
| decrease | 40 | 31 | 77.5% |
| stable | 9 | 7 | 77.8% |
| unspecified | 30 | 21 | 70.0% |
| absent | 20 | 20 | 100.0% |

Weighted in the same way, the extractor and the author agree on whether a pair carries a directional claim, and on its sign, for **93.6%** of all 18,900 pairs.


## Missed claims

11 sampled cases the extractor counted as carrying no directional claim carry one in the author's reading (9 of 30 called unspecified, 2 of 9 called stable, 0 of 20 called absent). Scaled stratum by stratum that is roughly **489** claims, **30%** of the 1,633 pairs called unspecified or stable, not counted against the 3,816 that were. Of the 11, **11** state the reference direction and 3 match the direction in the supplied window. The cases:

- id 1, s11, extractor stable, author increase, window increase: "... (trend: 0.217 -> 0.280), indicating potential wear on the engine's components. Conversely, degradation-related sensors s11 (trend: 0.257 -> 0.309) and s12 (trend: 0.253 -> 0.395) exhibit a steady increase in values, "
- id 5, s15, extractor unspecified, author increase, window increase: "The sensor readings indicate a gradual increase in degradation-related sensors such as s11, s12, and s15, suggesting advanced wear. Additionally, efficiency-related sensors like s9 and s14 show decreasing values, further"
- id 39, s15, extractor unspecified, author increase, window decrease: "...ith increasing values in degradation-related sensors such as s11 (trend: 0.613 -> 0.762 -> 0.696 -> 0.714 -> 0.762) and s15 (trend: 0.167 -> 0.593 -> 0.549 -> 0.810). Conversely, efficiency-related sensors like s9 and"
- id 53, s12, extractor unspecified, author increase, window decrease: "... showing signs of advanced wear, with increasing values in degradation-related sensors such as s11 (0.651 to 0.594) and s12 (0.147 to 0.133). Conversely, efficiency-related sensors like s9 (0.077 to 0.114) and s14 (0."
- id 67, s15, extractor unspecified, author increase, window decrease: "The sensor readings indicate a gradual increase in degradation-related sensors such as s11, s12, and s15, suggesting advanced wear. Additionally, there is a noticeable decrease in efficiency-related sensors like s9 and s"
- id 83, s12, extractor unspecified, author increase, window decrease: "...is showing signs of advanced wear, with increasing values in degradation-related sensors such as s11 (0.667 at t-3) and s12 (0.475 at t-3). Conversely, efficiency-related sensors like s9 (0.271 at t-3) and s14 (0.305 "
- id 115, s9, extractor stable, author decrease, window increase: "... and s12 (0.792 to 0.820) suggests advanced wear, while decreasing or stable values for efficiency-related sensors like s9 (0.437 to 0.469) and s14 (0.431 to 0.431) indicate a degradation pattern. The relatively high "
- id 123, s11, extractor unspecified, author increase, window decrease: "...s s9 and s14 show a consistent decrease in values, indicating advanced wear. Meanwhile, the degradation-related sensors s11, s12, and s15 exhibit increasing values, suggesting accelerated degradation. These trends ind"
- id 124, s12, extractor unspecified, author increase, window stable: "The recent trend shows increasing values for s11 (0.291 at t-2, up from 0.240 at t-5) and s12 (0.223 at t-2, up from 0.191 at t-5), indicating advanced wear in these degradation-related sensors. In contrast, efficiency-r"
- id 131, s12, extractor unspecified, author increase, window increase: "... showing signs of advanced wear, with increasing values in degradation-related sensors such as s11 (0.613 to 0.613) and s12 (0.505 to 0.456), indicating a trend towards failure. Conversely, efficiency-related sensors "
- id 134, s15, extractor unspecified, author increase, window decrease: "The sensor readings show a gradual increase in values for degradation-related sensors such as s11, s12, and s15, indicating advanced wear. In contrast, efficiency-related sensors like s9 and s14 exhibit decreasing values"

## Directional calls the author did not count

Of the 80 sampled directional calls, **10** carry no directional claim in the author's reading (8 read as unspecified, 2 read as stable), which scales to roughly **477** of the 3,816 claims counted. Of the 10, **9** carry the reference direction and 5 match the direction in the supplied window. The cases:

- id 23, s9, extractor decrease, author unspecified, window increase: "...s12, and s15, suggesting advanced wear. Additionally, there is a noticeable decrease in efficiency-related sensors like s9 and s14, further indicating impending failure."
- id 28, s7, extractor increase, author unspecified, window decrease: "The engine exhibits a pattern of gradual wear, with increasing deviations in efficiency sensors (s7 and s8) over the last five cycles. This is consistent with the reference example of advanced wear, suggesting that the e"
- id 48, s12, extractor increase, author unspecified, window stable: "The sensor readings indicate a gradual increase in degradation-related sensors such as s11, s12, and s15, suggesting advanced wear. Conversely, efficiency-related sensors like s9 and s14 show decreasing values, consisten"
- id 85, s9, extractor decrease, author unspecified, window decrease: "...ation-related sensors such as s11, s12, and s15, indicating advanced wear. In contrast, efficiency-related sensors like s9 and s14 exhibit lower values, suggesting decreased performance. The monotonic drift of these t"
- id 96, s14, extractor decrease, author stable, window decrease: "...elated sensors such as s11, s12, and s15, suggesting advanced wear. In contrast, efficiency-related sensors like s9 and s14 show relatively stable or decreasing values, indicating a potential loss of performance. The "
- id 100, s15, extractor increase, author unspecified, window increase: "The recent trend shows increasing values for s11, s12, and s15, indicating advanced wear. Conversely, s9 and s14 show decreasing values, suggesting a decline in efficiency. Additionally, the variance of these sensors has"
- id 102, s7, extractor decrease, author unspecified, window increase: "...dings, it appears that the engine is experiencing advanced wear, particularly in the efficiency sensors. The values for s7 and s8 are consistently lower than normal, indicating a potential issue with the engine's comb"
- id 125, s7, extractor decrease, author stable, window decrease: "...gradual degradation, with increasing deviations from normal operating conditions. Specifically, the efficiency sensors (s7 and s8) show a steady decline in readings over time, indicating advanced wear on the engine's "
- id 130, s11, extractor increase, author unspecified, window increase: "...s s9 and s14 show a consistent decrease in values, indicating advanced wear. Meanwhile, the degradation-related sensors s11 and s12 exhibit increasing variance and monotonic drift, suggesting an approaching failure. T"
- id 137, s11, extractor increase, author unspecified, window decrease: "The engine exhibits a pattern of gradual wear, with increasing deviations in efficiency sensors (s8 and s11) over the last five cycles. This is consistent with the reference example of advanced wear, suggesting that the "

## Wrong sign

Of the 80 sampled directional calls, **5** carry the opposite sign to the author's reading, one in 16.
Correcting them would make 2 faithful to the window and 2 unfaithful, with 1 on a flat window. The errors therefore do not lean towards making the model look less faithful than it is.

The cases:

- id 2, s14, extractor increase, author decrease, window stable: "...ase in s11, s12, and s15, indicating advanced wear and potential failure. Conversely, efficiency-related sensors s9 and s14 exhibit decreasing values, further supporting the trend of increasing degradation. Additional"
- id 55, s12, extractor decrease, author increase, window decrease: "...engine is showing signs of advanced wear, with increasing values in degradation-related sensors such as s11 (0.339) and s12 (0.789), indicating a decline in performance. Conversely, efficiency-related sensors like s9 "
- id 80, s12, extractor decrease, author increase, window increase: "...engine is showing signs of advanced wear, with increasing values in degradation-related sensors such as s11 (0.327) and s12 (0.738), indicating a decline in performance. Conversely, efficiency-related sensors like s9 "
- id 110, s12, extractor decrease, author increase, window decrease: "...engine is showing signs of advanced wear, with increasing values in degradation-related sensors such as s11 (0.536) and s12 (0.488), indicating a decline in performance. Conversely, efficiency-related sensors like s9 "
- id 113, s12, extractor decrease, author increase, window increase: "...engine is showing signs of advanced wear, with increasing values in degradation-related sensors such as s11 (0.488) and s12 (0.420), indicating a decline in performance. Conversely, efficiency-related sensors like s9 "

## What the errors do to the faithfulness figures

Each figure is estimated twice from the same sample, once with the extractor's verdicts and once with the author's labels, weighting every stratum by its corpus frequency. The first column is therefore the sample's own estimate of what the corpus shows, and the difference is what the extractor's errors do to it. Intervals are 95% percentile intervals from 10,000 bootstrap resamples within strata, paired between the two readings.

| Figure | Manuscript (corpus) | Extractor, sample | Author, sample | Difference | 95% interval |
|---|---|---|---|---|---|
| Directional claims in the corpus | 3,816 | 3,816 | 3,828 | +12 | -377 to +385 |
| Agreement with the reference direction | 91.3% | 92.4% | 100.0% | +7.6 | +2.5 to +13.8 |
| Faithfulness to the supplied window | 42.7% | 49.5% | 45.9% | -3.6 | -11.0 to +3.9 |
| Base rate on the same claims | 42.0% | 50.7% | 45.9% | -4.8 | -10.2 to +0.7 |
| Faithfulness minus base rate | +0.7 | -1.2 | 0.0 | +1.2 | -3.9 to +7.4 |
| Agreement with the empirical benchmark direction, scorable claims | 47.4% | 48.4% | 39.9% | -8.5 | -19.0 to +1.5 |
| Conflict set: share stating the reference direction | 89.5% | 84.3% | 100.0% | +15.7 | +3.7 to +29.1 |

In the author's reading **81 of the 81** sampled directional claims state the reference direction, including 32 of the 32 about a sensor whose own sub-dataset has it the other way round. Of the 6 sampled calls in which the extractor finds a departure from the reference, the author reads 0 the same way. Where every claim states the reference direction, faithfulness to the window equals the base rate by construction.

The clauses about deviations, variance or levels, which the extractor counts and the guideline does not, are concentrated where the prompt names no direction; their effect on the cue comparisons of Section 4.4 is computed in `p3_3_extractor_sensitivity.md`.


## Limits of this validation

One reader, the author, so no inter-annotator statistic is available and the numbers measure the extractor against one careful reading. The reading is the one step of the analysis that cannot be rerun from the code: a replication has to repeat it, on the released sample and with the released guideline. The sample is stratified rather than random, so only the reweighted figures describe the corpus, and with 80 sampled directional calls every corpus-level estimate here carries the sampling error shown in the intervals.
