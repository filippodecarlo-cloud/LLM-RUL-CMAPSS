# Changelog

## 2.2.2 - 2026-09-07

A third external review found that the manuscript declared one sensor unscorable
and then counted its claims in the aggregate that scores against the measured
direction. Checking that turned up a larger problem of the same kind, which is
what most of this release is about.

### Fixed

- **The measured direction was taken from FD001 and applied to claims made on
  FD003.** The corpus of 3,816 directional claims spans both sub-datasets, 2,396
  and 1,420, and the two do not attest the same directions: FD003 carries two
  fault modes, and there s14 is attested while s7, s12 and s15 are not. Every
  benchmark comparison is now computed within the sub-dataset the claim was made
  in. `p1/p3_5_prompt_vs_benchmark.py`, `p1/make_figures_xlsx.py` and
  `p1/audit_numbers.py` all changed; the audit computes it independently of p3_5,
  so the two would have to drift together to agree wrongly.
- **A sensor with no attested direction was being scored against one.** s14 rises
  in 57% of FD001 engines, below the 65% threshold the same script uses, so it
  has no direction to be right or wrong about there. Aggregate agreement with the
  measured direction is now reported over the 2,754 claims a sub-dataset can
  score, and is 47.4%, against 89.9% agreement with the reference direction on
  those same claims. The previously reported 41.0% carried both defects.
- **The conflicting-sensor result is unchanged in substance and now spans both
  sub-datasets.** Two of the five sensors the prompt names run the wrong way in
  each single-condition sub-dataset, and they are not the same two: s9 and s12 on
  FD001, s9 and s14 on FD003. Over the 1,484 claims about them the model follows
  the prompt 89.5% of the time and the benchmark 10.5%, against 89.3% and 10.7%
  on the narrower FD001-only subset reported before.
- `p1/p3_4_sensor_directions.py` used an 80% attestation threshold where
  `p3_5` used 65%, so the two scripts disagreed about s14 on FD003. Aligned to
  65%. The endpoint comparison and the per-engine rank correlation then classify
  every sensor and sub-dataset identically, which is a check worth having.
- `numeri_chiave.md` still reported the base rate over all 18,900 opportunities,
  46.7%, which is not the denominator faithfulness is measured on. It reads 42.0%
  now, matching the manuscript, and carries the benchmark-agreement line.
- `README.md` carried the superseded reading of three conflicting sensors, an
  unterminated bold marker in the title, and a causal claim that explicit
  reasoning fixes output collapse, which the single-run comparison cannot support.
- `p0/p0_2a_faithfulness.py` still named the earlier peer review in its
  docstring. The questions are unchanged; the attribution is gone.
- The 2.2.0 entry below has been corrected for the same defect and carries a note
  saying so.

## 2.2.1 - 2026-09-06

Corrections after a second external review of the manuscript. The analysis
changed in one place and the conclusion did not.

### Fixed

- **The per-sensor permutation test was wrong.** It collapsed each engine to the
  first window direction observed for it, but the direction depends on the
  window length and n takes three values: for s11, 99 of 200 engines carry more
  than one. The null was built from a quantity that does not exist in the data.
  The test now builds an engine x configuration table from every opportunity,
  not only from the cells where a claim was made, and permutes engines within a
  dataset so an engine's whole profile moves together. An intermediate version
  built the table from the claims alone; at 15% to 28% occupancy most permuted
  lookups missed and fell back to the observed value, which pushed p toward 1
  instead. Both the intermediate and the final version give the same answer.
- **Intervals and multiplicity in the same test.** Per-sensor intervals are now
  clustered on engine and on configuration with the wider reported, as the
  method always claimed, and Benjamini-Hochberg is applied across the six
  computable tests, which was missing. s12's uncorrected p moves from 0.014 to
  0.20 and after correction no sensor shows a detectable association between the
  stated direction and the window.
- **s14 was counted as contradicting the prompt.** It rises in 57% of engines,
  below the 65% this code uses to call a direction attested, so it has no
  consistent direction and cannot contradict anything. The comparison is now on
  s9 and s12: 1,406 claims, 89.3% following the asserted direction against 10.7%
  following the benchmark, with s14 reported separately.
- **Stale outputs.** `p1_model_summary.csv` carried a partial row from an
  interrupted sweep, Mistral at n = 2 against the 100 responses in its trace
  file; `p1_comparison.csv` predated the DeepSeek run. Both rebuilt from the
  traces.

### Added

- `p1/p3_8_write_zotero_fields.py`, which applies the Crossref lookup of
  `p3_7` to the local Zotero database. It fills empty fields only, refuses to
  run while Zotero is open, backs the database up first, and skips any record
  whose Crossref match `p3_7` flagged as approximate.

### Changed

- Figure 6 labels both series and shows the share, since 6 pairs against 253 was
  invisible on a shared axis. Figure 7 labels its seven points.

## 2.2.0 - 2026-09-06

Answers the statistical objections to the faithfulness analysis, and reports one
finding that came out of answering them.

### Added

- `p1/p3_1_cluster_inference.py` - every inferential claim in the faithfulness
  sections recomputed at the cluster level. The 3,816 directional claims are
  nested in responses, engines and configurations, so the published chi-squares
  reported p-values that were too small. Cluster bootstrap for intervals, cluster
  permutation for p-values, clustering once on engine and once on configuration
  with the wider interval kept.
- `p1/p3_2_multiplicity_examples_grid.py` - all 27 rank tests with the family
  defined and pre-specification stated; the example-set association reported as
  exploratory with n, p, a bootstrap interval and leave-one-out; and
  concentration recomputed on a common one-cycle grid, since counting distinct
  values favours a continuous model over one instructed to answer with an
  integer. On that grid the supervised regressors still occupy 59 to 69 values
  against 1 to 19 for the prompted ones.
- `p1/p3_3_extractor_sample.py` and `p1/p3_3_extractor_validate.py` - the claim
  extractor validated against 139 hand-read cases. Directional precision 0.95,
  recall 0.88, sign correct in 93%, 95% agreement over the corpus once
  reweighted. Both failure modes are conservative for the argument.
- `p1/p3_4_sensor_directions.py` and `p1/p3_5_prompt_vs_benchmark.py` - the
  degradation direction of each scored sensor measured on the training set
  instead of asserted, and compared with what the prompt claims.
- `p1/p3_6_word_layout.py`, `p1/p3_7_reference_metadata.py` - manuscript layout
  fixes, and a Crossref lookup for the bibliographic fields that were missing.

### The finding

The zero-shot prompt states which sensors rise and which fall with wear, and in
each single-condition sub-dataset two of the five it names are the wrong way
round. On FD001 the ratio of fuel flow to static pressure falls in 100 of 100
training engines where the prompt says it rises, and physical core speed rises in
70% where the prompt says it falls; on FD003 the pair is physical core speed and
corrected core speed. Two independent methods agree and the pattern holds at
every window length. On the 1,484 claims about a sensor the prompt has backwards
in its own sub-dataset, the model follows the prompt 89.5% of the time and the
benchmark 10.5%. Over the 2,754 claims a sub-dataset can score, agreement with
the reference direction is 89.9% and with the measured one 47.4%.

(The figures in this entry were corrected in 2.2.2. As first published they read
three sensors, 2,283 claims and 92.1% against 7.9%, counting corrected core speed
as conflicting on FD001 when it does not reach the attestation threshold there,
and scoring FD003 claims against FD001 directions.)

### Fixed

- `p0/p0_2a_faithfulness.py` computed the base rate over all 18,900 opportunities
  while faithfulness was computed over the 3,816 directional claims. On matched
  denominators the base rate is 42.0%, not 46.7%, so faithfulness is
  indistinguishable from a rule that ignores the input rather than below it. True
  at every threshold tested. The same mismatch was in the figure 5 data.
- `analyze_explainability.py` labelled s4 as the HPC outlet temperature. Under the
  Saxena et al. column order s4 is T50, the LPT outlet; T30 is s3. Verified by
  requiring the constant columns to be the ambient and demanded quantities.
- Two chart series set `border` twice in the same dict, so the outline meant to
  separate bars in greyscale was being discarded.

### Changed

- Axis and legend text from 10 to 11 pt, titles to 12 pt.
- Figure 1: the model box overflowed its border; enlarged and re-flowed.
- Figure 4 redrawn on a single linear axis. It carried three quantities on two
  axes with a logarithmic scale that existed only to fit 2 against 259.
- Figure 5 now shows four series on matched denominators, including agreement
  with the measured direction.
- Terminology throughout: the measure is agreement with the direction the prompt
  asserts, not with canonical physics, since on this benchmark those differ.

## 2.1.0 - 2026-09-06

Brings the repository into line with the manuscript. Release 2.0.0 held 4,657 of the
model responses the paper reports; this one holds all 5,557.

### Added

- `results_p1/traces_p1_grid_FD001_fs_k*_n*.json` - the nine cells of the k x n grid
  re-run with the stratified example sampler, 900 responses, with
  `p1_grid_fixed_summary.csv` and `p1_grid_report.md`.
- `p1/p1_grid_analyze.py` - the analysis behind that grid.
- `p1/verify_tables.py` - recomputes every table cell in the manuscript from the traces
  and compares it with what the manuscript says. Needs the .docx; set `PAPER_DOCX`.
- `p1/export_figures_png.py` - renders the figures from the Excel and PowerPoint sources.
- `p1/make_manifest.py`, `manifest.json`, `manifest.csv` - one record per inference run:
  which experiment it belongs to, its configuration, its SHA-256, and the metrics
  recomputed from the file rather than copied from a report.
- `REPRODUCIBILITY.md` - Ollama version, model tags and digests, hardware, generation
  settings, retry and timeout behaviour, baseline hyperparameters, seeds, and an
  explicit statement of what cannot be reproduced exactly.
- `figures_png/` - the eight figures as they appear in the paper.

### Changed

- `requirements.txt` now pins the versions actually used.
- `figures/` renamed to `figures_original_grid/`. Those matplotlib figures belong to the
  first release and their numbering does not correspond to the paper's figures, which
  was a live source of confusion while they shared a directory name with the current
  ones.
- `figures_v14/` updated to the renumbered figures, so the workbook matches the paper.
- README rewritten: every claim is now scoped to the experiment it holds for, since the
  model sweep is FD001 only and the four sub-datasets were run with the reference model
  only.

### Fixed

- `analyze_explainability.py` counted the nine verdicts of "stable" as wrong-direction
  claims, and therefore inside the denominator of "directional mentions". Its report
  read 3,825 claims and 91.1% agreement where the paper reads 3,816 and 91.3% on the
  same traces. A claim that names no direction is now reported on its own line, and the
  regenerated `results/explainability_report.md` agrees with the manuscript. The
  underlying classifier is unchanged, so nothing else moves.
- `p1/p1_grid_analyze.py` compared a range across k with a standard deviation across
  example draws, which understates the noise, and concluded from it that k has no
  effect. The comparison is now range against range (15.13 against 16.34), the
  conclusion is scoped to RMSE, and the report states separately what k does to
  concentration: mean modal share moves 79% to 55% to 99% as k goes 3, 5, 10.

### Note

The 2.0.0 changelog listed the k ablation as withdrawn. It is no longer withdrawn: the
grid was re-run with a corrected sampler and is reported in the paper. The original
27 traces are unchanged and are kept as the first of two sampling conditions.

Entries below this one are left as written. Two of their wordings would not be used
today: the 2.0.0 note says the concentration reproduces "across four models, two
quantisation levels and all four sub-datasets", where in fact the model sweep is FD001
only and the four sub-datasets were run with the reference model only; and it treats the
quantisation comparison as showing no effect, where Q4 to Q8 leaves the concentration at
85% and 80% but does move the error and the number of distinct values.

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
