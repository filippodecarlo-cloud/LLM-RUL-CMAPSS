"""
Collect hardware and runtime information for the paper's
computational resources section. Run from Anaconda Prompt:
    python collect_system_info.py
"""

import platform
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

info = {}

# ── OS & Python ──────────────────────────────────────────────
info["os"]             = platform.system()
info["os_version"]     = platform.version()
info["python_version"] = platform.python_version()
info["architecture"]   = platform.machine()

# ── CPU ──────────────────────────────────────────────────────
try:
    import psutil
    info["cpu_name"]         = platform.processor()
    info["cpu_physical_cores"]  = psutil.cpu_count(logical=False)
    info["cpu_logical_cores"]   = psutil.cpu_count(logical=True)
    info["cpu_freq_max_mhz"]    = round(psutil.cpu_freq().max, 1) if psutil.cpu_freq() else "N/A"
    info["ram_total_gb"]        = round(psutil.virtual_memory().total / 1e9, 2)
    info["ram_available_gb"]    = round(psutil.virtual_memory().available / 1e9, 2)
except ImportError:
    print("psutil not found — install with: pip install psutil")
    import multiprocessing
    info["cpu_logical_cores"] = multiprocessing.cpu_count()

# ── GPU ──────────────────────────────────────────────────────
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        info["gpu"] = result.stdout.strip()
    else:
        info["gpu"] = "None (CPU-only)"
except Exception:
    info["gpu"] = "None (CPU-only)"

# ── Ollama model info ─────────────────────────────────────────
try:
    result = subprocess.run(
        ["ollama", "show", "llama3.1:8b", "--modelfile"],
        capture_output=True, text=True, timeout=10)
    info["llm_model"] = "Llama 3.1 8B (Q4_K_M quantization via Ollama)"
except Exception:
    info["llm_model"] = "Llama 3.1 8B via Ollama"

# ── Experiment timing (from results files) ───────────────────
results_dir = Path("results")
traces = sorted(results_dir.glob("traces_*.json"))

timing = {}
for f in traces:
    timing[f.stem] = {
        "completed_at": datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S"),
        "file_size_kb": round(f.stat().st_size / 1024, 1)
    }
    with open(f) as fh:
        data = json.load(fh)
    timing[f.stem]["n_engines"] = len(data)

info["experiment_timing"] = timing

# ── Throughput estimate ───────────────────────────────────────
# Based on FD001 zero-shot (50 engines, ~79 min)
info["avg_seconds_per_engine_zero_shot"] = round(79 * 60 / 50, 1)
info["avg_seconds_per_engine_few_shot"]  = round(69 * 60 / 50, 1)

# ── Print report ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("COMPUTATIONAL RESOURCES REPORT")
print("=" * 55)
for k, v in info.items():
    if k == "experiment_timing":
        print(f"\nExperiment timing:")
        for name, t in v.items():
            print(f"  {name}: {t['n_engines']} engines, completed {t['completed_at']}")
    else:
        print(f"{k:35s}: {v}")

# ── Save to JSON ─────────────────────────────────────────────
out = Path("results") / "system_info.json"
with open(out, "w") as f:
    json.dump(info, f, indent=2, default=str)
print(f"\nSaved to: {out.absolute()}")
