# LLM-RUL-CMAPSS

Code and experimental results for the paper:

> **Large Language Models as Implicit Classifiers in Predictive Maintenance: A Diagnostic Study on Turbofan Remaining Useful Life**
> Filippo De Carlo, University of Florence.

This repository reproduces a diagnostic evaluation of **Llama 3.1 8B** (local inference via [Ollama](https://ollama.com), no fine-tuning, no architectural modification) for Remaining Useful Life (RUL) prediction on the **NASA CMAPSS** turbofan benchmark, compared against Random Forest and LSTM baselines.

**Central finding.** The LLM does not behave as a regressor: across all FD001 configurations it maps the 100 test engines onto only **2–11 distinct predicted values** (up to 85% on a single integer) — *categorical collapse* / *implicit-classifier* behaviour. The RMSE improvement with longer prompt windows is a cluster-selection effect, not genuine regression. Meanwhile the model's natural-language reasoning correctly identifies the expected sensor-degradation direction in **>90% of directional mentions** across 2,700 traces, exposing a systematic decoupling between textual reasoning and numerical output. The full reasoning traces in [`results/`](results/) are the primary evidence for these claims.

---

## Repository structure

| Path | Contents |
|---|---|
| `experiment.py` | Main script: data loading, Random Forest, LSTM, Llama inference, metrics |
| `config.py` | Centralised parameters (datasets, k, n, RUL cap, sensors) |
| `analyze_explainability.py` | Semi-quantitative analysis of the 2,700 reasoning traces |
| `generate_figures.py`, `generate_figures_n30.py`, `generate_fig1_methodology.py` | Figure generation |
| `check_table8.py` | Verifies the predicted-value distribution tables against the raw traces |
| `download_data.py` | Helper to fetch the NASA CMAPSS dataset (see below) |
| `collect_system_info.py`, `test_api.py` | Hardware/info utilities |
| `results/` | 27 JSON trace files (engine_id, predicted RUL, true RUL, full reasoning) + result CSVs + explainability report |
| `figures/` | Paper figures (PNG 300 dpi + PDF) |

> The raw **CMAPSS data is not redistributed here** (it is public NASA data). See *Data* below.

---

## Requirements

- Python 3.13 (3.10+ should work)
- [Ollama](https://ollama.com) running locally with the model pulled:
  ```bash
  ollama serve
  ollama pull llama3.1:8b
  ```
- Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## Data

The experiments use the NASA CMAPSS Turbofan Engine Degradation Simulation dataset (sub-datasets FD001 and FD003), which is publicly available from the NASA Prognostics Center of Excellence Data Repository. Place the `train_FDxxx.txt`, `test_FDxxx.txt`, and `RUL_FDxxx.txt` files in a `cmapss_data/` folder next to the scripts (or run `python download_data.py`).

## Reproducing the results

```bash
# Baselines (RF, LSTM) + LLM zero/few-shot at the default n=5 window
python experiment.py

# Window-size ablation (n = 5, 15, 30)
python experiment.py --cycles 15 --suffix n15 --skip-baselines
python experiment.py --cycles 30 --suffix n30 --skip-baselines

# Run-to-run variance (zero-shot, FD001, n=30, repeated)
python experiment.py --cycles 30 --suffix n30_run2 --datasets FD001 --modes zero_shot --skip-baselines

# Explainability analysis over all trace files
python analyze_explainability.py

# Figures
python generate_figures.py
python generate_figures_n30.py
python generate_fig1_methodology.py
```

Inference is fully local: temperature is fixed at `T=0.1` and the model is served at `http://localhost:11434/v1`.

## Mapping results → paper

- `results/traces_FD001_*.json`, `results/traces_FD003_*.json` → Tables 3, 5, 6, 7 and the predicted-value distribution tables (verify with `check_table8.py`).
- `results/all_results*.csv` → aggregate RMSE / MAE / NASA Score per configuration.
- `results/explainability_report.md` → the per-sensor explainability table.

## Citation

If you use this code or data, please cite the paper (DOI to be added on acceptance) and this repository. A `CITATION.cff` is included.

## License

Code released under the MIT License (see `LICENSE`). The NASA CMAPSS dataset is subject to its own (public) terms from the NASA Prognostics Center of Excellence.
