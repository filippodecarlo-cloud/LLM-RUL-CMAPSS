# P0.2b / P0.3 - control arms: does the model follow the prompt or the data?

Paired design: the same 100 FD001 test engines, the same model (llama3.1:8b, T=0.1), the same zero-shot template, n=30. One thing changes per arm.


## 1. The predicted numbers barely move

| Arm | RMSE | MAE | NASA | distinct values | modal value | modal share |
|---|---|---|---|---|---|---|
| repl (original prompt, real data) | 61.57 | 50.94 | 44,023 | 3 | 23 | 80% |
| nocue (cue removed) | 59.13 | 49.62 | 32,450 | 4 | 23 | 69% |
| invcue (cue asserts opposite physics) | 62.32 | 52.24 | 44,512 | 5 | 23 | 84% |
| revdata (original cue, window reversed in time) | 61.21 | 51.45 | 43,688 | 4 | 23 | 77% |

The replication arm lands inside the band of the four existing runs of this exact configuration (RMSE 61.5-62.8, modal value 23 at 80-85%), so the harness reproduces `experiment.py`.

**Every arm still collapses onto 23.** Removing the cue, reversing the cue, and reversing every trend in the data all leave the output distribution essentially unchanged: 3-5 distinct values over 100 engines, 69-84% on the single integer 23. The number the model emits is a property of the prompt format, not of the sensor evidence or of the physics it is told.


## 2. The explanations follow the cue (P0.2b)

| Arm | Directional claims | Agreement with canonical physics | Agreement with what THIS prompt asserts |
|---|---|---|---|
| repl (original prompt, real data) | 346 | 94.2% | 94.2% |
| nocue (cue removed) | 89 | 95.5% | n/a (no cue) |
| invcue (cue asserts opposite physics) | 357 | 62.2% | 37.8% |
| revdata (original cue, window reversed in time) | 338 | 95.6% | 95.6% |

- **Removing the cue does not lower canonical agreement** (94.2% -> 95.5%). What it changes is how much the model talks about directions at all: directional claims fall from **346 to 89** (74% fewer). Unprompted, the model largely stops naming sensor trends; when it still names one it stays canonical. So the cue is not the sole source of the 91% - the canonical degradation narrative is also in the model's priors.
- **Reversing the cue does move it**, from 94.2% to **62.2%**, with **37.8%** of claims now following the asserted (physically wrong) direction. Told that worn engines behave the opposite way, the model reverses roughly a third of its claims about the very same unchanged sensor data. The prompt exerts a large causal effect on the explanations; the prior resists the rest.
- repl vs invcue: chi2 = 103.1, p = 3.26e-24.

## 3. The explanations ignore the data (P0.3)

| Arm | Directional claims | Faithfulness to the window actually shown |
|---|---|---|
| repl (original prompt, real data) | 347 | 43.5% |
| revdata (original cue, window reversed in time) | 339 | 42.8% |

**The paired test.** Among the (engine, sensor) pairs where the model made a directional claim in both arms and the trend in the window genuinely reversed (**n = 253**), the claim changed in **6 cases (2.4%)**.

The data was turned upside down and the explanation stayed the same. Whatever the reasoning text is describing, it is not the sensor window in the prompt.


## 4. Per sensor, repl vs revdata

| Sensor | In cue? | Claims increase/decrease (repl) | Claims increase/decrease (revdata) |
|---|---|---|---|
| s4 | NO | 0 / 0 | 0 / 0 |
| s7 | NO | 1 / 0 | 1 / 0 |
| s9 | yes | 1 / 98 | 1 / 94 |
| s11 | yes | 99 / 0 | 99 / 0 |
| s12 | yes | 20 / 18 | 20 / 12 |
| s14 | yes | 1 / 95 | 2 / 94 |
| s15 | yes | 14 / 0 | 16 / 0 |

## Conclusion

- The **numbers** are unmoved by removing the cue, by reversing the cue, and by reversing the data. Every arm collapses onto 23.
- The **explanations** are driven by the prompt and by the model's canonical prior, in that order, and by the sensor window not at all: when the trends are reversed the claims follow only 2.4% of the time.
- Nuance worth keeping in the paper: the reported 91% is **not purely** prompt leakage. Deleting the cue leaves canonical agreement intact while suppressing most directional claims, so part of it comes from the model's priors. Contradicting the cue does move agreement by 32 points, which establishes the prompt's causal role.
- Either way the metric never measured what the paper claimed: under every arm, including the untouched original, the explanations match the data given to the model roughly 43% of the time - below the rate obtained by ignoring the data entirely (P0.2a).