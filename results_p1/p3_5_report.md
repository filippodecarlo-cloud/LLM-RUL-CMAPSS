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

On FD001, the reference sub-dataset:

- The prompt is right about 4 of the 7 scored sensors.
- It asserts the **opposite** of what the data shows for s9, s12, s14.

  - **s9** (Nc, Physical core speed): the prompt says decrease, the data shows increase in 70% of engines.
  - **s12** (phi, Ratio of fuel flow to Ps30): the prompt says increase, the data shows decrease in 100% of engines.
  - **s14** (NRc, Corrected core speed): the prompt says decrease, the data shows increase in 57% of engines.

FD003 carries two fault modes and its directions are correspondingly less consistent, which is worth noting but does not change the picture on FD001.


## What the model follows

| Reference | Agreement over 3,816 directional claims |
|---|---|
| The direction the prompt asserts | **91.3%** |
| The direction measured in the benchmark | **41.0%** |
| The direction in the window actually shown | 42.7% |

| Sensor | Prompt vs data | Claims | Follows prompt | Follows data | Follows window |
|---|---|---|---|---|---|
| s4 (T50) | agree | 10 | 60.0% | 60.0% | 50.0% |
| s7 (P30) | agree | 270 | 51.1% | 51.1% | 49.3% |
| s9 (Nc) | **conflict** | 879 | 95.6% | 4.4% | 27.1% |
| s11 (Ps30) | agree | 947 | 98.4% | 98.4% | 62.1% |
| s12 (phi) | **conflict** | 527 | 78.7% | 21.3% | 47.2% |
| s14 (NRc) | **conflict** | 877 | 96.6% | 3.4% | 27.7% |
| s15 (BPR) | agree | 306 | 100.0% | 100.0% | 56.9% |

**The decisive subset.** On the 3 sensors where the prompt and the benchmark disagree (s9, s12, s14), a claim cannot follow both. Over the 2,283 claims about them the model follows the prompt 92.1% of the time and the benchmark 7.9%. Where the two references part company, the model goes with the prompt.


## What this changes

The headline figure is unchanged as a number and changes as a claim. Agreement of 91.3% is agreement with the direction the prompt supplies, and it was described as agreement with established degradation physics. On this benchmark those are not the same thing: measured against the data the same claims are right 41.0% of the time. It also explains why input faithfulness sits at the level a rule that ignores the input would reach, since the reference the model is echoing is itself uninformative about the sensor windows for three of the five sensors the prompt names.
