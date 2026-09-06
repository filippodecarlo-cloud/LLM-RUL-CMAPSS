# P3.3 - Validating the directional-claim extractor

A stratified sample of 139 (response, sensor) pairs was read by hand against the same local context the extractor sees, and labelled increase, decrease, stable, unspecified or absent. The sample is stratified over the extractor's own verdict so that both kinds of error can be measured, which means the raw accuracy in the sample is not the accuracy in the corpus; the corpus figures below are reweighted by how often each verdict actually occurs.

**5 of 139 cases were labelled ambiguous** and are excluded from the scoring. They are all of one kind: the model hedges, as in "relatively stable or decreasing values", and no single label is correct. The extractor resolves such phrases inconsistently, calling the same construction stable in one response and decrease in another.


## Confusion matrix, sampled cases

Rows are the extractor, columns the human reading.

| extractor \ human | increase | decrease | stable | unspecified | absent | total |
|---|---|---|---|---|---|---|
| **increase** | 36 | 1 | 0 | 3 | 0 | 40 |
| **decrease** | 4 | 34 | 0 | 1 | 0 | 39 |
| **stable** | 1 | 0 | 4 | 0 | 0 | 5 |
| **unspecified** | 9 | 0 | 0 | 21 | 0 | 30 |
| **absent** | 0 | 0 | 0 | 0 | 20 | 20 |

## Precision and recall, sampled cases

| Class | Precision | Recall | F1 | Sampled n (human) |
|---|---|---|---|---|
| increase | 0.900 | 0.720 | 0.800 | 50 |
| decrease | 0.872 | 0.971 | 0.919 | 35 |
| stable | 0.800 | 1.000 | 0.889 | 4 |
| unspecified | 0.700 | 0.840 | 0.764 | 25 |
| absent | 1.000 | 1.000 | 1.000 | 20 |

Treating increase and decrease together as **a directional claim**, which is the distinction the faithfulness analysis rests on, precision is 0.949 and recall 0.882 in the sample. Of the 75 cases where both the extractor and the reader see a directional claim, the extractor gets the sign right in 70, 93.3%.


## Reweighted to the corpus

| Extractor verdict | Corpus count | Sampled | Correct in sample | Implied corpus accuracy |
|---|---|---|---|---|
| increase | 1,860 | 40 | 36 | 90.0% |
| decrease | 1,956 | 39 | 34 | 87.2% |
| stable | 9 | 5 | 4 | 80.0% |
| unspecified | 1,624 | 30 | 21 | 70.0% |
| absent | 13,451 | 20 | 20 | 100.0% |

Weighting each verdict by how often it occurs, the extractor agrees with the reader on **95.1%** of all 18,900 (response, sensor) pairs in the corpus.


## What the errors do to the headline percentages

Two error modes matter and they pull in opposite directions.

- **Missed claims.** 10 of the 35 sampled cases the extractor called unspecified or stable do carry a directional claim, 29%. Scaled to the corpus that is roughly 467 claims not counted, against the 3,816 that were. Almost all are the same construction: the direction is stated once for a list of sensors and the 60-character window before the sensor token does not reach back to the verb.
- **Wrong sign.** Of the sampled cases the extractor called increase, 90% are read as increase; of those it called decrease, 87% are read as decrease. The failure is a clause such as "increasing values in s11 and s12, indicating a decline in performance", where the trailing word belongs to the consequence and not to the sensor.

Both modes are conservative for the paper's argument rather than favourable to it. The missed claims are overwhelmingly canonical statements of the kind the prompt cues, so counting them would raise agreement with the textbook, which is already the high number. The sign errors are a few per cent and are not systematically aligned with the window, so they add noise to input faithfulness rather than bias. The gap between 91% and 43% is far larger than either.


## Limits of this validation

One annotator, no second reader, so no inter-rater statistic is available and the numbers measure the extractor against one careful reading. The annotator saw the extractor's verdict while labelling, which is a source of anchoring. The sample is stratified rather than random, so only the reweighted figures describe the corpus. The residual ambiguity is concentrated in two constructions: hedged disjunctions, and claims about a level or a variance rather than a trend, where "values consistently lower than normal" and "increasing deviations" are not statements about the direction of the series.
