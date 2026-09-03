# P0.4 - Zero-shot collapses harder than few-shot

If stratified few-shot examples caused the categorical collapse, removing them would relieve it. They do not: prompts with no examples produce the *narrowest* output sets in the whole grid.


## By prompting mode (both datasets pooled)

| Mode | Runs | Distinct values (mean, range) | Modal share | Output entropy (bits) |
|---|---|---|---|---|
| zero-shot (no examples) | 9 | 2.3 (2-3) | 74% | 0.81 |
| few-shot (k=3,5,10) | 18 | 5.8 (3-11) | 54% | 1.68 |

Mann-Whitney U = 9, p = 0.0001 for the one-sided hypothesis that zero-shot yields FEWER distinct values than few-shot.

For scale: 100 engines with a genuine regressor give up to 6.64 bits of output entropy; Random Forest produces 100 distinct values. Zero-shot averages 0.81 bits.


## Every zero-shot run

| Dataset | n | Run | Distinct | Modal value | Modal share | Entropy (bits) |
|---|---|---|---|---|---|---|
| FD001 | 5 | run1 | 3 | 23 | 68% | 0.97 |
| FD001 | 15 | run1 | 2 | 23 | 84% | 0.63 |
| FD001 | 30 | run1 | 2 | 23 | 85% | 0.61 |
| FD001 | 30 | run2 | 2 | 23 | 81% | 0.70 |
| FD001 | 30 | run3 | 2 | 23 | 81% | 0.70 |
| FD001 | 30 | run4 | 3 | 23 | 80% | 0.78 |
| FD003 | 5 | run1 | 2 | 23 | 61% | 0.96 |
| FD003 | 15 | run1 | 3 | 23 | 71% | 0.93 |
| FD003 | 30 | run1 | 2 | 23 | 51% | 1.00 |

## Reading

- Zero-shot prompts contain no examples, no RUL labels and no stratification, yet return a median of 2 distinct values across 100 engines, against 5 for few-shot.
- The narrowest run in the entire grid is zero-shot: 2 distinct values, modal share 85%.
- Stratified few-shot examples therefore cannot be the cause of the collapse. They widen the output set slightly, by supplying values to copy, but the collapse is already present without them.
- What the examples do change is WHICH value the model settles on. In few-shot the modal value tracks the window length (n=5 -> 23, n=15 -> 45, n=30 -> 95); in zero-shot it is 23 for every window length and both datasets. The window length moves the anchor only when examples are present to supply candidate values - an anchoring effect, not improved temporal modelling (reviewer R1.7).