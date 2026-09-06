# P1.0 - The published few-shot procedure does not match the paper


## 1. Static analysis of `experiment.run_llm`

- The example pool is the last cycle of each training engine: **100 engines, RUL min 0, max 0** - every one of them is zero, because the piecewise target is 0 at failure.
- Stratifying that pool by RUL bin gives **{0: 100}**: bins 1 and 2 are empty.
- So the loop over the three bins can only draw from bin 0, and the prompt receives `per_bin = max(1, k//3)` examples rather than `k`:

| k requested | per_bin | examples actually in the prompt | engines | RUL shown |
|---|---|---|---|---|
| 3 | 1 | **1** | [1] | [0] |
| 5 | 1 | **1** | [1] | [0] |
| 10 | 3 | **3** | [1, 2, 3] | [0, 0, 0] |

Every example is labelled `True RUL: 0` and, since 0 <= 40, described by `_build_example_block` as *"advanced wear; significant deviation in efficiency sensors"*. The model is shown only end-of-life engines and asked to extrapolate to healthy ones.


## 2. Empirical check on the published traces

Reference for *same prompt, T=0.1*: 5 replicate zero-shot n=30 runs, 10 pairs. Mean observed agreement 79.6%, chance 69.6%, **excess +10.0 points**.

| Dataset | n | k3 vs k5 (same prompt predicted) | k3 vs k10 (different prompt) |
|---|---|---|---|
| FD001 | 5 | 67% vs 64% chance (**+2.6**) | 50% vs 45% chance (**+4.6**) |
| FD001 | 15 | 57% vs 46% chance (**+11.2**) | 23% vs 24% chance (**-0.7**) |
| FD001 | 30 | 39% vs 26% chance (**+13.0**) | 32% vs 24% chance (**+7.5**) |
| FD003 | 5 | 53% vs 39% chance (**+13.8**) | 3% vs 1% chance (**+1.7**) |
| FD003 | 15 | 67% vs 58% chance (**+9.4**) | 24% vs 23% chance (**+1.4**) |
| FD003 | 30 | 37% vs 19% chance (**+17.7**) | 28% vs 20% chance (**+8.4**) |

Mean excess agreement: **k3-k5 +11.3 points**, in line with the +10.0 of true replicates of one prompt; **k3-k10 +3.8 points**, clearly lower. The pattern matches the static analysis: k=3 and k=5 are the same single-example prompt, k=10 is a different three-example one.


## 3. What this invalidates

- **The k ablation is not an ablation over k.** Every 'k=3' and 'k=5' result in the paper comes from one example; 'k=10' from three. Any statement of the form 'performance improves with more examples' is unsupported - two of the three levels are the same condition.
- **The stratification described in the methods never happened.** All examples sit in one RUL bin, and it is the bin at failure.
- **This is a plausible mechanism for the low anchors.** The model is shown engines with RUL 0 described as badly worn, then asked about a test set whose mean RUL is 74.5. Few-shot predictions collapse onto 23, 45 and 95 - all far below the mean. This is a testable hypothesis rather than speculation, and P1.3 tests it by supplying genuinely stratified examples.
- **The released code and the described procedure diverge.** Anything that reports a k must therefore say which of the two samplers produced it.

> Note: the categorical collapse itself does **not** depend on this defect. Zero-shot prompts contain no examples at all and collapse harder (P0.4), and the collapse reproduces on mistral, qwen2.5 and llama-q8_0 (P1.2). What the defect invalidates is specifically the k ablation and the stratification claim.