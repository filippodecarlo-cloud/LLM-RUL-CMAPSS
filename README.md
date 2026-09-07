# LLM-RUL-CMAPSS

Code, inference traces and analysis for:

> **Evaluating large language models for remaining useful life prediction: a protocol and evidence from C-MAPSS**
> Filippo De Carlo, University of Florence.

This repository evaluates open-weight language models prompted **without fine-tuning** for
Remaining Useful Life (RUL) prediction on the **NASA CMAPSS** turbofan benchmark, against
Random Forest, XGBoost, LSTM and constant predictors.

The contribution is a **four-part evaluation protocol** and what it finds when applied:

1. **No-skill baselines.** Compare against a predictor that ignores its input and always
   returns one number.
2. **Rank correlation.** Spearman's rho with bootstrap intervals and correction for multiple
   testing, to separate ordering ability from point accuracy.
3. **Output entropy.** Distinct predicted values, modal share and Shannon entropy, which make
   categorical collapse measurable.
4. **Explanation faithfulness.** Whether a stated sensor direction matches the reference
   direction is a different question from whether it matches the window actually supplied. Three
   controls separate them: remove the prompt cue, invert it, and reverse the sensor trends in
   the data.

## What the protocol finds

**Error.** No configuration in the original 27-run grid beats a constant predictor fitted on
the training set (best 45.60 against 41.94 RMSE on FD001), and none of the nine cells of the
grid re-run with a corrected example sampler does either. One run in the example-set
experiment does, reaching RMSE 38.68; it is reported in full because it looks like a
counterexample, and because its squared rank correlation of 0.07 against 0.66 for Random
Forest is the clearest argument for the protocol's second part.

**Concentration.** In the four non-reasoning configurations on FD001, predictions fall on 1 to
5 distinct values over 100 engines. The reference model returns the same modal value, 23, on
all four sub-datasets, over 100 to 259 engines, under different operating conditions, fault
modes and normalisation.

**Explanations.** Over 18,900 sensor-trace opportunities the reference model makes 3,816
explicit directional claims. They agree with the reference direction 91.3% of the
time and with the direction in the supplied window 42.7%, against 42.0% for a rule that never
looks at the input on the same claims, so no detectable difference was observed.

**The prompt is not right about the benchmark.** In each single-condition sub-dataset two of
the five sensors the zero-shot template names run opposite to the direction it asserts, and they
are not the same two: on FD001 the ratio of fuel flow to static pressure falls in 100 of 100
training engines while the prompt says it rises, and physical core speed rises in 70% while the
prompt says it falls; on FD003 the pair is physical core speed and corrected core speed. A
direction counts only where at least 65% of the training engines show it, which excludes s14 on
FD001 and s7, s12 and s15 on FD003. On the 1,484 claims about a sensor the prompt has backwards
in its own sub-dataset, the model follows the prompt 89.5% of the time and the benchmark 10.5%.
Over the 2,754 claims a sub-dataset can score, agreement with the reference direction is 89.9%
and agreement with the measured one is 47.4%.
Reversing every trend in the input changes 2.4% of the stated directions. Across seven example
sets, the mean prediction tracks the mean RUL of the examples with r = +0.941 over those seven
aggregate points.

**Reasoning models are a separate case.** DeepSeek-R1 7B is far less concentrated, producing 19
distinct values over 50 engines rather than 1 to 5. It is not more predictive: RMSE 67.31
against 39.93 for a constant on the same engines, and rho = -0.12. Output collapse and absence
of prognostic capability are therefore separate failures. Which property of this model accounts
for the difference is not identified here, since it differs from the others in training data and
architecture as well as in reasoning explicitly.

Scope matters when reading any of the above. The model sweep is FD001 only; the four
sub-datasets were run with the reference model only; the quantisation comparison is Llama 3.1
8B at Q4_K_M against Q8_0. `manifest.csv` labels every run with the experiment it belongs to.

---

## Repository structure

| Path | Contents |
|---|---|
| `experiment.py`, `config.py` | Original experiment: data loading, baselines, Llama inference, metrics |
| `p0/` | The evaluation protocol: no-skill baselines, rank correlation, faithfulness, control arms |
| `p1/` | Model sweep, few-shot resampling, corrected grid, repaired baselines, multi-condition datasets, figure and audit generators |
| `results/` | The 27 traces of the original grid (engine id, prediction, ground truth, full reasoning) |
| `results_p0/` | Protocol outputs: no-skill tables, rank-correlation audit, faithfulness analysis, control arms |
| `results_p1/` | Model sweep, corrected grid, repaired baselines, example-set experiment, FD002/FD004 |
| `figures_v14/` | The paper's figures as **native Excel charts** and an editable PowerPoint diagram |
| `figures_png/` | The same eight figures rendered, as they appear in the paper |
| `figures_original_grid/` | Matplotlib figures from the first release, kept for reference; they are **not** the paper's figures and their numbering does not correspond |
| `manifest.json`, `manifest.csv` | One record per inference run: configuration, experiment, SHA-256, and metrics recomputed from the file |
| `numeri_chiave.md` | Every number in the paper, regenerated from the data |
| `REPRODUCIBILITY.md` | Versions, model digests, hardware, generation settings, hyperparameters, seeds |

### The scripts that matter

| Script | What it does |
|---|---|
| `p0/p0_1_noskill_and_rank.py` | Constant predictors and Spearman rho for all 27 configurations |
| `p0/p0_1b_rank_significance.py` | BH correction, bootstrap intervals, baselines as a reference scale |
| `p0/p0_2a_faithfulness.py` | Textbook agreement vs input faithfulness over 3,816 directional claims |
| `p0/p0_23_control_prompts.py` | The four paired control arms (replication, cue removed, cue inverted, trends reversed) |
| `p0/p0_4_zeroshot_collapse.py` | Concentration by prompting mode |
| `p1/p3_1_cluster_inference.py` | Every faithfulness contrast redone at the cluster level: claims are nested in engines and configurations |
| `p1/p3_2_multiplicity_examples_grid.py` | All 27 rank tests with the family stated, the example-set association as an exploratory result, and a common one-cycle grid for comparing concentration across model families |
| `p1/p3_3_extractor_sample.py`, `p1/p3_3_extractor_validate.py` | Draw and score a hand-annotated sample validating the claim extractor |
| `p1/p3_5_prompt_vs_benchmark.py` | Whether the degradation directions the prompt asserts are the ones the benchmark exhibits |
| `p1/p1_model_sweep.py` | Same configuration, different model; robust parsing, raw responses kept |
| `p1/p1_34_fewshot_sampling.py` | The example-set experiment: k fixed, only the examples change |
| `p1/p1_grid_fixed.py`, `p1/p1_grid_analyze.py` | The k x n grid re-run with a stratified sampler, and its analysis |
| `p1/p2_1_baselines.py` | Random Forest, XGBoost, and the LSTM under both training regimes |
| `p1/p2_2_multicondition.py` | FD002 and FD004 with per-regime scaling |
| `p1/make_manifest.py` | Rebuilds `manifest.json` and `manifest.csv` with fresh checksums |
| `p1/make_key_numbers.py` | Regenerates `numeri_chiave.md` from the data |
| `p1/audit_numbers.py`, `p1/verify_tables.py` | Check the manuscript against the data: a named list of quantities, and every table cell |

Every inference script is resumable: it writes after each engine and skips completed ones on
restart.

---

## Two things worth knowing before reading the code

**Two few-shot sampling conditions, not one.** The paper compares two ways of building the
example pool, and both are in this repository. The first draws each example from the final
cycle of a training engine, where the piecewise RUL target is zero by construction. Every
engine therefore falls in the lowest of the three RUL bins, and since the sampler takes
`max(1, k // 3)` engines from each bin, a requested k of 3 or 5 places **one** example in the
prompt and a requested k of 10 places **three**, all labelled `True RUL: 0`. That is the
condition behind `results/` and it is why `manifest.csv` records the requested k rather than
an effective one. The second draws stratified windows ending at an arbitrary cycle, so labels
span 0 to 125; that is `p1/p1_grid_fixed.py` and `p1/p1_34_fewshot_sampling.py`. Fixing the
sampler moves the anchor and does not lift the collapse.

**Two scaling paths, on purpose.** `experiment.py` fits one `MinMaxScaler` on the training
set. That is correct for FD001 and FD003, which have a single operating condition.
`p1/p2_2_multicondition.py` fits one scaler per operating regime, because on FD002 and FD004
a global scaler buries degradation under regime switching. Neither is a leftover; see
`REPRODUCIBILITY.md`.

---

## Requirements

- Python 3.13 (3.10+ should work)
- [Ollama](https://ollama.com) running locally:
  ```bash
  ollama serve
  ollama pull llama3.1:8b
  ```
  Other models used: `mistral:7b`, `qwen2.5:7b`, `deepseek-r1:7b`,
  `llama3.1:8b-instruct-q8_0`. Exact digests are in `REPRODUCIBILITY.md`.
- Packages: `pip install -r requirements.txt` (versions pinned to those used for the paper)

## Data

The CMAPSS data is public NASA data and is not redistributed here. Use `download_data.py`, or
place the `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` files in `cmapss_data/`.

## Reproducing the analysis

Most results are recomputed from the stored traces and need no inference:

```bash
python p0/p0_1_noskill_and_rank.py      # no-skill baselines and rank correlation
python p0/p0_2a_faithfulness.py         # the two explanation measures
python p1/p1_grid_analyze.py            # the corrected k x n grid
python p1/make_key_numbers.py           # regenerate every number in the paper
```

Two scripts train models rather than reading traces, and take several minutes on CPU:
`p1/p2_1_baselines.py`, which fits Random Forest, XGBoost and both LSTM configurations,
and `p0/p0_1b_rank_significance.py`, which fits the supervised reference used as a scale
for the rank correlations. TensorFlow does not use the GPU on native Windows, so both
run on CPU there.

New inference requires Ollama and takes hours per configuration on consumer hardware. No
sampling seed is sent to the model, so a re-run will not reproduce the stored traces
exactly; `REPRODUCIBILITY.md` explains what this does and does not affect.

## Hardware limits, stated plainly

Everything ran on a laptop with a 4 GB GPU and 15.7 GB of RAM. Consequences: fp16 weights for
an 8B model do not fit, so the quantisation comparison is Q4_K_M against Q8_0; DeepSeek-R1 was
run on 50 engines rather than 100 because of its inference cost; and Gemma 4 could not be
evaluated at all, since it crashes the runtime on the 30-cycle prompt.

## Relation to release 1.0.0

Release 1.0.0 was an earlier version of this codebase and remains available at its own version
DOI. Two things changed in the code between the releases, and anyone who downloaded 1.0.0
should know about both.

The **LSTM baseline in 1.0.0 had not converged**: it emitted a single value for every test
engine, worse than the training mean, because the 0-125 target was left unscaled under MSE,
`validation_split` cut a contiguous tail of overlapping windows from the same engines, and
there was no early stopping. `p1/p2_1_baselines.py` runs the same architecture under both
regimes; trained properly, FD001 RMSE goes from 43.89 to 13.58.

The **few-shot example pool in 1.0.0 was built from the last cycle of each training engine**,
so every example carried the label zero and the number of examples reaching the prompt was
not the requested k. The 27 traces in `results/` were produced that way and are unchanged;
`p1/p1_grid_fixed.py` re-runs the grid with a stratified sampler so the two conditions can be
compared, and `p1/p1_0_fewshot_bug.py` demonstrates the behaviour of the original one.

Neither changes the central finding. Zero-shot prompts contain no examples and concentrate
harder than few-shot ones, and the concentration reproduces across models, across the two
quantisation levels tested, and on all four sub-datasets. Correcting both widens the gap
between prompting and the supervised baselines rather than narrowing it.

## Licence

MIT, see `LICENSE`.
