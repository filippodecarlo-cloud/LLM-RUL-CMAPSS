# ============================================================
# CONFIGURATION
# ============================================================
# NOTE: inference runs locally via Ollama (no cloud API key required).
# The optional variable below is read from the environment only if some
# external provider is ever used; it is NOT needed to reproduce the paper.
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # optional, unused for local Ollama runs

# Sub-datasets to evaluate (FD001 and FD003 — single operating condition)
# Add "FD002", "FD004" for the multi-condition extension.
DATASETS = ["FD001", "FD003"]

# RUL cap for the piecewise-linear target
RUL_CAP = 125

# Number of recent cycles shown to the LLM in each prompt
# Window-size ablation: vary via the --cycles CLI flag (e.g. 5, 15, 30)
N_CYCLES_IN_PROMPT = 5

# Few-shot example counts to test
FEW_SHOT_K = [3, 5, 10]

# Seconds between local Ollama calls (kept for compatibility)
API_DELAY = 5.0

# Max test engines per dataset (None = all 100)
MAX_TEST_ENGINES = None

# Informative sensors (constant sensors removed: s1,s5,s6,s10,s16,s18,s19)
SENSOR_NAMES = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
