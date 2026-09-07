# P3.5 - The direction the prompt asserts against the direction in the data

The zero-shot prompt states which sensors rise and which fall with wear, and the faithfulness analysis scored the model against that same set of directions. That measures whether the model repeats the prompt. It assumes the prompt is right about the benchmark, which is a separate question and is settled here by measurement.


## What the benchmark does

Share of training engines in which the sensor is higher at the end than at the start, at four scales. The last column is what the prompt asserts.

| Dataset | Sensor | Symbol | Whole life | Last 30 | Last 15 | Last 5 | Measured | Asserted |
|---|---|---|---|---|---|---|---|---|
| FD001 | s4 | T50 | 100% | 97% | 94% | 70% | increase | increase |
| FD001 | s7 | P30 | 0% | 4% | 13% | 31% | decrease | decrease |
| FD001 | s9 | Nc | 70% | 66% | 65% | 57% | increase | decrease |
| FD001 | s11 | Ps30 | 100% | 99% | 89% | 70% | increase | increase |
| FD001 | s12 | phi | 0% | 2% | 12% | 28% | decrease | increase |
| FD001 | s14 | NRc | 57% | 58% | 57% | 55% | increase | decrease |
| FD001 | s15 | BPR | 100% | 92% | 81% | 62% | increase | increase |
| FD003 | s4 | T50 | 100% | 96% | 91% | 73% | increase | increase |
| FD003 | s7 | P30 | 44% | 44% | 53% | 63% | decrease | decrease |
| FD003 | s9 | Nc | 83% | 80% | 79% | 71% | increase | decrease |
| FD003 | s11 | Ps30 | 100% | 100% | 97% | 69% | increase | increase |
| FD003 | s12 | phi | 44% | 45% | 46% | 61% | decrease | increase |
| FD003 | s14 | NRc | 76% | 75% | 76% | 78% | increase | decrease |
| FD003 | s15 | BPR | 56% | 52% | 52% | 61% | increase | increase |

A direction counts as attested when at least 65% of engines show it. The two methods agree everywhere: the sign of the median rank correlation with engine age gives the same direction as the endpoint comparison for every sensor and both sub-datasets.

At the 65% attestation threshold the seven scored sensors fall into three groups, and the grouping is not the same in the two sub-datasets:

| Sub-dataset | Prompt direction is the one the data shows | Prompt direction is the opposite of it | No consistent direction |
|---|---|---|---|
| FD001 | s4, s7, s11, s15 | **s9, s12** | s14 |
| FD003 | s4, s11 | **s9, s14** | s7, s12, s15 |

  - **s9** (Nc, Physical core speed) on FD001: the prompt says decrease, the data shows increase in 70% of engines.
  - **s12** (phi, Ratio of fuel flow to Ps30) on FD001: the prompt says increase, the data shows decrease in 100% of engines.
  - s14 (NRc) on FD001: majority trend increase in only 57% of engines, below the threshold, so the sensor has no direction to be scored against here.
  - **s9** (Nc, Physical core speed) on FD003: the prompt says decrease, the data shows increase in 83% of engines.
  - **s14** (NRc, Corrected core speed) on FD003: the prompt says decrease, the data shows increase in 76% of engines.
  - s7 (P30) on FD003: majority trend decrease in only 56% of engines, below the threshold, so the sensor has no direction to be scored against here.
  - s12 (phi) on FD003: majority trend decrease in only 56% of engines, below the threshold, so the sensor has no direction to be scored against here.
  - s15 (BPR) on FD003: majority trend increase in only 56% of engines, below the threshold, so the sensor has no direction to be scored against here.

Two of the five sensors the prompt names are the wrong way round in each single-condition sub-dataset, but not the same two: s9 and s12 on FD001, s9 and s14 on FD003.


## What the model follows

The first and third rows are defined for every claim. The second is defined only where the sub-dataset attests a direction, which is 2,754 of the 3,816 claims.

| Reference | Claims | Agreement |
|---|---|---|
| The direction the prompt asserts | 3,816 | **91.3%** |
| The direction measured in the benchmark | 2,754 | **47.4%** |
| The direction in the window actually shown | 3,816 | 42.7% |

On the same 2,754 claims, agreement with the direction the prompt asserts is 89.9%, so the gap is not an artefact of the restriction.


| Sensor | Sub-dataset | Prompt vs data | Claims | Follows prompt | Follows data | Follows window |
|---|---|---|---|---|---|---|
| s4 (T50) | FD001 | agree | 3 | 66.7% | 66.7% | 33.3% |
| s7 (P30) | FD001 | agree | 170 | 40.0% | 40.0% | 45.9% |
| s9 (Nc) | FD001 | **conflict** | 578 | 95.8% | 4.2% | 28.5% |
| s11 (Ps30) | FD001 | agree | 612 | 99.2% | 99.2% | 65.8% |
| s12 (phi) | FD001 | **conflict** | 309 | 65.7% | 34.3% | 46.9% |
| s14 (NRc) | FD001 | no consistent direction | 581 | 96.7% | n/a | 28.2% |
| s15 (BPR) | FD001 | agree | 143 | 100.0% | 100.0% | 64.3% |
| s4 (T50) | FD003 | agree | 7 | 57.1% | 57.1% | 57.1% |
| s7 (P30) | FD003 | no consistent direction | 100 | 70.0% | n/a | 55.0% |
| s9 (Nc) | FD003 | **conflict** | 301 | 95.0% | 5.0% | 24.3% |
| s11 (Ps30) | FD003 | agree | 335 | 97.0% | 97.0% | 55.2% |
| s12 (phi) | FD003 | no consistent direction | 218 | 97.2% | n/a | 47.7% |
| s14 (NRc) | FD003 | **conflict** | 296 | 96.3% | 3.7% | 26.7% |
| s15 (BPR) | FD003 | no consistent direction | 163 | 100.0% | n/a | 50.3% |

**The decisive subset.** Where the attested direction is the opposite of the asserted one, a claim cannot follow both. Over the 1,484 claims about such a sensor in such a sub-dataset the model follows the prompt 89.5% of the time and the benchmark 10.5%. Where the two references part company, the model goes with the prompt.

Reported separately because it cannot be scored: on FD001 the prompt asserts decrease for s14 and the model states it in 96.7% of its 581 claims, while the benchmark supports no direction either way.
Reported separately because it cannot be scored: on FD003 the prompt asserts decrease for s7 and the model states it in 70.0% of its 100 claims, while the benchmark supports no direction either way.
Reported separately because it cannot be scored: on FD003 the prompt asserts increase for s12 and the model states it in 97.2% of its 218 claims, while the benchmark supports no direction either way.
Reported separately because it cannot be scored: on FD003 the prompt asserts increase for s15 and the model states it in 100.0% of its 163 claims, while the benchmark supports no direction either way.


## What this changes

The headline figure is unchanged as a number and changes as a claim. Agreement of 91.3% is agreement with the direction the prompt supplies, and it was described as agreement with established degradation physics. On this benchmark those are not the same thing: measured against the data the same claims are right 47.4% of the time, on the claims where the benchmark attests a direction at all. It also explains why input faithfulness sits at the level a rule that ignores the input would reach, since the reference the model is echoing is itself uninformative about the sensor windows for the sensors it gets backwards.
