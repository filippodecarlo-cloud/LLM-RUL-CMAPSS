# Reproducibility

Everything needed to re-derive the paper's numbers, and an honest account of what
cannot be reproduced exactly. The version strings, digests and hardware below were
read off the machine that ran the experiments, not written from memory.

## What can be reproduced exactly, and what cannot

The analysis is deterministic. Every table and figure in the paper is recomputed from
the stored traces by the scripts listed at the end of this file, and repeated runs give
identical output. That includes the supervised baselines: re-running
`p1/p2_1_baselines.py` on the machine described below reproduces
`p2_1_baseline_summary.csv`, `p2_1_baseline_predictions.csv` and
`p2_1_lstm_diagnostics.csv` byte for byte, seed 42. The one exception is
`experiment.run_lstm`, which sets no seed; `p0/p0_1b_rank_significance.py` calls it for
a reference scale, so its LSTM row moves by roughly 0.1 RMSE between runs while
remaining a single-valued predictor. Every LSTM figure quoted in the paper comes from
the seeded `p1/p2_1_baselines.py`.

Inference is not. No sampling seed is sent to the model: the request carries
`temperature` and nothing else, so `top_p`, `top_k`, `num_predict` and the seed take
Ollama's model defaults, and a fresh run of the same configuration will not reproduce
the stored traces token for token. This is a limitation of how the experiments were
run, and it is why the paper reports four replicates of the reference configuration
(FD001, zero-shot, n = 30) rather than a single run: across them RMSE spans 61.5 to
62.8 and the modal share 80% to 85%. That spread is the scale on which any single-run
difference in this paper should be read.

## Software

| Component | Version |
|---|---|
| Ollama | 0.33.3 |
| Python | 3.13.7 (CPython, AMD64) |
| numpy | 2.4.1 |
| pandas | 3.0.0 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| tensorflow | 2.21.0 |
| xgboost | 3.4.1 |
| openai (client library) | 2.33.0 |
| matplotlib | 3.10.9 |
| openpyxl | 3.1.5 |
| python-docx | 1.2.0 |

`requirements.txt` pins these versions. The analysis scripts are not sensitive to
minor version drift in numpy or pandas, but the LSTM result depends on TensorFlow's
initialisation, so a different TensorFlow may move it by a fraction of a cycle.

## Models

Tags and digests as reported by `ollama list` on that machine. `deepseek-r1:7b` is a
distillation with a Qwen2 architecture, which is worth knowing when reading it beside
`qwen2.5:7b`.

| Tag | Digest | Parameters | Quantisation | Architecture |
|---|---|---|---|---|
| llama3.1:8b | 46e0c10c039e | 8.0B | Q4_K_M | llama |
| llama3.1:8b-instruct-q8_0 | b158ded76fa0 | 8.0B | Q8_0 | llama |
| mistral:7b | 6577803aa9a0 | 7.2B | Q4_K_M | llama |
| qwen2.5:7b | 845dbda0ea48 | 7.6B | Q4_K_M | qwen2 |
| deepseek-r1:7b | 755ced02ce7b | 7.6B | Q4_K_M | qwen2 |

`llama3.1:8b` is the reference model throughout. `gemma4:e4b-it-qat` was pulled and
excluded: it aborts the runtime on the 30-cycle prompt.

## Hardware

| Item | Value |
|---|---|
| CPU | Intel Core i7-1360P, 12 cores / 16 threads |
| RAM | 16,815,702,016 bytes (15.7 GB) |
| GPU | NVIDIA RTX A500 Laptop, 4096 MiB, driver 573.44 |
| OS | Windows 11, build 10.0.22631 |

Two consequences are load-bearing for the paper. Full-precision weights for an 8B
model do not fit in this much memory, so the quantisation comparison is Q4_K_M
against Q8_0 rather than against fp16. And TensorFlow does not use the GPU on native
Windows, so all baselines were trained on CPU.

## Generation settings

All inference goes through the OpenAI-compatible endpoint Ollama exposes at
`http://localhost:11434/v1`, with the API key string `ollama`, which the local server
ignores.

| Setting | Value |
|---|---|
| temperature | 0.1 in every run |
| top_p, top_k, num_predict, seed | not sent; Ollama model defaults apply |
| max_tokens | not set |
| request timeout | 600 s in `p1/p1_model_sweep.py`; the OpenAI client default elsewhere |
| retries | 3 attempts per engine |
| backoff | 10/20/30 s in `experiment.py`; 5/10/15 s in the `p0/` and `p1/` runners |
| pause between calls | 0.5 s in `experiment.py` |

**On exhausted retries.** `experiment.py` records a prediction of 75 and the reasoning
string `API error`. That branch never fired: no prediction among the 2,700 published
traces equals 75, and the parse rate over all 5,557 responses is 100%. The `p0/` and
`p1/` runners skip the engine instead, and no run is missing an engine. Every
inference script writes after each engine and skips completed ones on restart, so an
interrupted run resumes without repeating calls.

## Data handling

| Item | Value |
|---|---|
| RUL target | piecewise linear, capped at 125 cycles |
| Sensors used | the 14 non-constant channels: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21 |
| Sensors dropped | s1, s5, s6, s10, s16, s18, s19, constant across the benchmark |
| Scaling, FD001 and FD003 | one `MinMaxScaler` fitted on train, applied to test |
| Scaling, FD002 and FD004 | one `MinMaxScaler` per operating regime, each fitted on train |
| Operating regimes | k-means, 6 clusters, `random_state=42`, fitted on the train settings and used to assign test rows |
| Baseline window | 30 cycles, zero-padded at the front for shorter engines |

FD001 and FD003 have a single operating condition, so a global scaler is the right
choice there and `experiment.py` uses one. FD002 and FD004 have six, where a global
scaler buries degradation under regime switching, so `p1/p2_2_multicondition.py` fits
one scaler per regime. The two scripts differ deliberately; neither is a leftover.

## Baseline hyperparameters

**Random Forest** (`experiment.py`): `n_estimators=200`, `random_state=42`, all other
scikit-learn defaults. Features are the mean, standard deviation and net change of
each sensor over the window.

**XGBoost** (`p1/p2_1_baselines.py`): `n_estimators=600`, `max_depth=6`,
`learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.
Same features as the Random Forest.

**LSTM** (`p1/p2_1_baselines.py`). Both rows in the paper's baseline table come from
one architecture: `LSTM(64, return_sequences=True)`, `Dropout(0.2)`, `LSTM(32)`,
`Dropout(0.2)`, `Dense(1)`, over 30-cycle windows of the 14 sensors, MSE loss,
`tf.keras.utils.set_random_seed(42)`. They differ in how the target is scaled and how
training is supervised.

| | Unscaled target | Scaled target |
|---|---|---|
| target | raw 0-125 | divided by 125 |
| optimiser | `adam` defaults | Adam, lr 1e-3, `ReduceLROnPlateau` (patience 5, factor 0.5) |
| validation | `validation_split=0.1`, a contiguous tail of overlapping windows from the same engines | 15% of *engines*, drawn with `default_rng(42)` |
| epochs | 30, run to completion | up to 100, `EarlyStopping` patience 10, best weights restored |
| batch size | 256 | 256 |

The first configuration converges to a constant. It is reported because a recurrent
baseline that fails this way still records an RMSE better than any prompting
configuration in the original grid, which is the paper's point about error metrics.

## Statistics

| Quantity | Setting |
|---|---|
| Bootstrap resamples | 10,000, `default_rng(20260901)`, resampled over engines |
| Random-uniform reference | mean of the metrics of 1,000 complete U(0,125) prediction vectors, `default_rng(42)`, not the metric of an averaged prediction |
| Multiple testing | Benjamini-Hochberg over the family of 27 rank tests, with Bonferroni reported alongside |
| Flat-window threshold | net change below 0.01 in the normalised units printed in the prompt |
| Spearman tests | two-sided |

## Selection of the reported configurations

The paper's headline table reports, for each sub-dataset, the configuration with the
lowest RMSE and the configuration with the lowest NASA score among the 27 published
runs. Both are selected on the test set and are therefore optimistic. They are
reported that way deliberately: the argument is that even a test-set-optimal
prompting configuration does not beat a constant fitted on the training set alone.
All 27 runs are in `results_p0/p0_1_llm_configs.csv` and
`results_p0/p0_1b_rank_significance.csv`, so nothing is hidden behind the selection.

## Reproducing the analysis without running inference

```bash
pip install -r requirements.txt
python download_data.py                 # or place the CMAPSS files in cmapss_data/
python p0/p0_1_noskill_and_rank.py      # constant baselines, rank correlation
python p0/p0_1b_rank_significance.py    # BH/Bonferroni, bootstrap intervals
python p0/p0_2a_faithfulness.py         # textbook agreement vs input faithfulness
python p0/p0_23_analyze.py              # the four control arms
python p0/p0_4_zeroshot_collapse.py     # collapse by prompting mode
python p1/p1_analyze.py                 # model sweep against the baselines
python p1/p1_34_analyze.py              # the example-set experiment
python p1/p1_grid_analyze.py            # the corrected k x n grid
python p1/p2_1_baselines.py             # trains RF, XGBoost and both LSTMs
python p1/make_key_numbers.py           # regenerates numeri_chiave.md
```

`p1/p2_1_baselines.py` and `p0/p0_1b_rank_significance.py` train models and take several
minutes each on CPU; everything else reads the stored traces and finishes in seconds.

Two scripts need something this repository does not contain.
`p1/verify_tables.py` recomputes every table cell and compares it with the
manuscript, so it needs the `.docx`; set `PAPER_DOCX` to its path.
`p1/export_figures_png.py` renders the charts by driving Excel and PowerPoint over
COM, so it needs Windows with both installed. The chart data itself is in
`figures_v14/figures_v14.xlsx`, one `DATI_figN` sheet per figure, readable anywhere.

## Re-running inference

```bash
ollama serve
ollama pull llama3.1:8b
python experiment.py --datasets FD001 --modes zero_shot --cycles 30 --suffix n30
```

On the hardware above a configuration of 100 engines takes roughly one to two hours.
DeepSeek-R1 needs about seven minutes per engine, which is why it was run on 50 engines
rather than 100. Because no seed is sent, a re-run produces different traces; compare
distributions, not individual predictions.
