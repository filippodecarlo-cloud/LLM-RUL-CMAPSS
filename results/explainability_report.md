### Explainability — Semi-Quantitative Analysis (v100 (100 engines))

Aggregated across **2700 reasoning traces** from all LLM configurations (zero-shot + few-shot k=3,5,10) and both sub-datasets.

| Sensor | Physical Quantity | Expected Trend | Correctly Identified | Wrong Direction | Mentioned but Unspecified | Not Mentioned |
|---|---|---|---|---|---|---|
| s4 | Total temperature at LPT outlet | ↑ (increase) | 6 (0%) | 4 (0%) | 29 (1%) | 2661 (99%) |
| s7 | Total pressure at HPC outlet | ↓ (decrease) | 138 (5%) | 132 (5%) | 631 (23%) | 1799 (67%) |
| s9 | Physical core speed | ↓ (decrease) | 840 (31%) | 45 (2%) | 54 (2%) | 1761 (65%) |
| s11 | Static pressure at HPC outlet | ↑ (increase) | 932 (35%) | 16 (1%) | 245 (9%) | 1507 (56%) |
| s12 | Ratio of fuel flow to Ps30 | ↑ (increase) | 415 (15%) | 112 (4%) | 471 (17%) | 1702 (63%) |
| s14 | Corrected core speed | ↓ (decrease) | 847 (31%) | 32 (1%) | 15 (1%) | 1806 (67%) |
| s15 | Bypass ratio | ↑ (increase) | 306 (11%) | 0 (0%) | 179 (7%) | 2215 (82%) |

**Summary across the 7 informative sensors:**
- Directional claims (increase or decrease): **3816** (20.2% of 18900 sensor-trace opportunities).
- Of these, **3484** (91.3%) name the canonical degradation direction and **332** (8.7%) name the opposite one.
- A further **9** mentions call the sensor stable; they state no direction and are excluded from the two percentages above.

Agreement with the canonical direction is not a measure of whether the explanation describes the input. For that, see `p0/p0_2a_faithfulness.py` and `results_p0/p0_2a_report.md`, which score the same claims against the sensor window actually shown in each prompt.