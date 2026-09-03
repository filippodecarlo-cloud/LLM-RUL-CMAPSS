# LLM-RUL-CMAPSS

Code, inference traces and analysis for:

> **Evaluating large language models for remaining useful life prediction: a protocol, and what it reveals on CMAPSS**
> Filippo De Carlo, University of Florence.

This repository evaluates open-weight language models prompted **without fine-tuning** for
Remaining Useful Life (RUL) prediction on the **NASA CMAPSS** turbofan benchmark, against
Random Forest, XGBoost, LSTM and constant predictors.

The contribution is a **four-part evaluation protocol** and what it finds when applied:

1. **No-skill baselines.** Compare against a predictor that ignores its input and always
   answers one number.
2. **Rank correlation.** Spearman's rho with bootstrap intervals and correction for multiple
   testing, to separate ordering ability from point accuracy.
3. **Output entropy.** Distinct predicted values, modal share and Shannon entropy, which make
   categorical collapse measurable.
4. **Explanation faithfulness.** Whether a stated sensor direction matches the canonical
   physics is a different question from whether it matches the window actually supplied. Three
   controls separate them: remove the prompt cue, invert it, and reverse the sensor trends in
   the data.

**What the protocol finds.** No prompting configuration beats a constant predictor. Predictions
concentrate on 1 to 5 distinct values over 100 to 259 test engines, with the same modal value
on all four sub-datasets, across four models and two quantisation levels. Explanations agree
with canonical degradation physics 91.3% of the time and with the data actually supplied 42.7%,
below the 46.7% obtained by ignoring the input entirely. Reversing every trend in the input
changes 2.4% of the stated directions, while the mean prediction is 89% explained by the mean
RUL of the examples placed in the prompt.

A reasoning model (DeepSeek-R1 7B) breaks the categorical collapse, producing 19 distinct
values, but has the worst error of any model tested and no ability to order engines. Output
collapse and absence of prognostic capability are separate failures.

---

## Corrections to the previous release

Release 1.0 of this repository accompanied earlier work on the same benchmark. While rebuilding
the baselines for the present study, two defects were found in that code. Both are corrected
here and documented in Appendix A of the paper.

**The LSTM baseline had not converged.** It emitted a single value for every test engine (56.69
on FD001), worse than the training mean. Three causes: an unscaled 0-125 target under MSE
with default Adam, a `validation_split` that cut a contiguous tail of overlapping windows from
the same engines, and no early stopping. Retrained correctly, RMSE goes from 43.89 to **13.58**
on FD001, better than Random Forest and XGBoost. See `p1/p2_1_baselines.py`.

**The few-shot procedure did not stratify.** The example pool was built from the last cycle of
each training engine, where the piecewise RUL target is 0 by construction. All engines
therefore fell in one bin, and the prompt received one example for k=3 and k=5, and three for
k=10, all labelled `True RUL: 0`. The ablation over k in the previous release is withdrawn: two
of its three levels are the same condition. See `p1/p1_0_fewshot_bug.py` for the
demonstration, static and empirical.

**Neither defect explains the central finding.** Zero-shot prompts contain no examples and
collapse harder than few-shot ones, and the collapse reproduces across models, quantisations
and all four sub-datasets. Correcting the defects widens the gap between prompting and
supervised baselines rather than narrowing it.

The previous version remains permanently available at its own DOI.

---

## Repository structure

| Path | Contents |
|---|---|
| `experiment.py`, `config.py` | Original experiment: data loading, baselines, Llama inference, metrics |
| `p0/` | The evaluation protocol: no-skill baselines, rank correlation, faithfulness, control arms |
| `p1/` | Model sweep, few-shot resampling, repaired baselines, multi-condition datasets, figure and audit generators |
| `results/` | 27 trace files from the original grid (engine id, prediction, ground truth, full reasoning) |
| `results_p0/` | Protocol outputs: no-skill tables, rank-correlation audit, faithfulness analysis, control arms |
| `results_p1/` | Model sweep, repaired baselines, few-shot sampling, FD002/FD004 |
| `figures_v14/` | Figures as **native Excel charts** and an editable PowerPoint diagram |
| `numeri_chiave.md` | Every number in the paper, regenerated from the data |

### The scripts that matter

| Script | What it does |
|---|---|
| `p0/p0_1_noskill_and_rank.py` | Constant predictors and Spearman rho for all 27 configurations |
| `p0/p0_1b_rank_significance.py` | BH correction, bootstrap intervals, baselines as a reference scale |
| `p0/p0_2a_faithfulness.py` | Textbook agreement vs input faithfulness over 3,816 directional claims |
| `p0/p0_23_control_prompts.py` | The four paired control arms (replication, cue removed, cue inverted, trends reversed) |
| `p1/p1_0_fewshot_bug.py` | Demonstrates the few-shot defect, statically and on the published traces |
| `p1/p1_model_sweep.py` | Same configuration, different model; robust parsing, raw responses kept |
| `p1/p2_1_baselines.py` | Diagnoses and repairs the LSTM, adds XGBoost |
| `p1/p2_2_multicondition.py` | FD002 and FD004 with per-regime scaling |
| `p1/make_key_numbers.py` | Regenerates `numeri_chiave.md` from the data |
| `p1/audit_numbers.py` | Checks every number in the manuscript against the result files |

Every inference script is resumable: it writes after each engine and skips completed ones on
restart.

---

## Requirements

- Python 3.13 (3.10+ should work)
- [Ollama](https://ollama.com) running locally:
  ```bash
  ollama serve
  ollama pull llama3.1:8b
  ```
  Other models used: `mistral:7b`, `qwen2.5:7b`, `deepseek-r1:7b`,
  `llama3.1:8b-instruct-q8_0`.
- Packages: see `requirements.txt`

## Data

The CMAPSS data is public NASA data and is not redistributed here. Use `download_data.py`, or
place the `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` files in `cmapss_data/`.

## Reproducing the analysis

Most results are recomputed from the stored traces and need no inference:

```bash
python p0/p0_1_noskill_and_rank.py      # no-skill baselines and rank correlation
python p0/p0_2a_faithfulness.py         # the two explanation measures
python p1/p1_0_fewshot_bug.py           # the few-shot defect
python p1/p2_1_baselines.py             # repaired baselines (trains RF, XGBoost, LSTM)
python p1/make_key_numbers.py           # regenerate every number in the paper
```

New inference requires Ollama and takes hours per configuration on consumer hardware. Timings
and the exact machine are reported in the paper.

## Hardware limits, stated plainly

Everything ran on a laptop with a 4 GB GPU and 15.7 GB of RAM. Consequences: fp16 weights for
an 8B model do not fit, so the quantisation comparison is Q4_K_M against Q8_0; DeepSeek-R1 was
run on 50 engines rather than 100 because of its inference cost; and Gemma 4 could not be
evaluated at all, since it crashes the runtime on the 30-cycle prompt.

## Licence

MIT, see `LICENSE`.
