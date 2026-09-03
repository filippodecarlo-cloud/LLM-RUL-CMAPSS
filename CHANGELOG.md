# Changelog

## 2.0.0 — 2026-09-03

New study on the same benchmark, with an evaluation protocol as the main contribution, and
corrections to two defects present in 1.0.0.

### Added

- `p0/` — the evaluation protocol: no-skill baselines and rank correlation for all 27
  configurations, BH correction and bootstrap intervals, the two explanation measures over
  3,816 directional claims, the four paired control arms, and the demonstration of the
  few-shot defect.
- `p1/` — model sweep across Mistral 7B, Qwen2.5 7B, DeepSeek-R1 7B and Llama 3.1 8B at two
  quantisation levels; repaired supervised baselines with XGBoost added; few-shot resampling
  under three conditions and three seeds; FD002 and FD004 with per-regime scaling; generators
  for the figures, for the key numbers, and for the manuscript number audit.
- `results_p0/`, `results_p1/` — all outputs, including the new inference traces.
- `figures_v14/` — figures as native Excel charts, editable rather than bitmaps, plus the
  methodology diagram as PowerPoint shapes.
- `numeri_chiave.md` — every number in the paper, regenerated from the data.

### Fixed

- **LSTM baseline did not converge.** It emitted one value for every test engine (56.69 on
  FD001, 65.32 on FD003), worse than the training mean. Causes: unscaled 0-125 target under
  MSE with default Adam; `validation_split` cutting a contiguous tail of overlapping windows
  from the same engines, so validation loss never measured generalisation; no early stopping.
  Corrected in `p1/p2_1_baselines.py`: RMSE 43.89 to 13.58 on FD001, 39.87 to 13.81 on FD003.

- **Few-shot example selection did not stratify.** The pool was the last cycle of each training
  engine, where the piecewise RUL target is 0 by construction, so all engines fell in one bin.
  The prompt received one example for k=3 and k=5, and three for k=10, all labelled
  `True RUL: 0` and described as advanced wear. Demonstrated in `p1/p1_0_fewshot_bug.py`.

### Withdrawn

- **The ablation over the number of few-shot examples.** Two of its three levels used an
  identical prompt, so the comparison was not an ablation over k.

- **The explanation accuracy figure as previously framed.** Agreement with canonical
  degradation physics (91.3%) was measured while the prompt itself stated the expected sensor
  directions. Reported here alongside agreement with the data actually supplied (42.7%), which
  is below the rate obtained by ignoring the input.

### Note on the numbers

Neither defect explains the categorical collapse. Zero-shot prompts contain no examples and
collapse harder than few-shot ones; the collapse reproduces across four models, two
quantisation levels and all four sub-datasets. Correcting the defects widens the gap between
prompting and supervised baselines.

## 1.0.0 — 2026-06-23

First release. Llama 3.1 8B on FD001 and FD003, zero-shot and few-shot, against Random Forest
and LSTM baselines. 27 inference traces. Permanently available at its own version DOI.
