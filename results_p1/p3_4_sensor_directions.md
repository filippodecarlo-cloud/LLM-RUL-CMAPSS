# P3.4 - Sensor identities and measured degradation directions

## The mapping from column to measurement

Column order follows the variable list of Saxena et al. (2008). The check is that the columns which carry ambient or demanded quantities, and therefore cannot vary in a fixed-condition simulation, are exactly the columns that are constant in the data.

- Constant in FD001: s1, s5, s10, s16, s18, s19
- Predicted constant under this mapping: s1, s5, s10, s16, s18, s19
- **Match: yes**


## Direction under degradation, measured on the training set

Per training engine, the Spearman correlation between the sensor and the remaining useful life; the direction is the sign of the median across engines. A negative correlation with RUL means the sensor rises as the engine wears.

| Sensor | Saxena symbol | Measurement | Dataset | Median rho with RUL | Engines agreeing | Measured direction |
|---|---|---|---|---|---|---|
| s4 | T50 | Total temperature at LPT outlet | FD001 | -0.789 | 100% of 100 | increase |
| s7 | P30 | Total pressure at HPC outlet | FD001 | +0.783 | 100% of 100 | decrease |
| s9 | Nc | Physical core speed | FD001 | -0.744 | 71% of 100 | increase |
| s11 | Ps30 | Static pressure at HPC outlet | FD001 | -0.819 | 100% of 100 | increase |
| s12 | phi | Ratio of fuel flow to Ps30 | FD001 | +0.801 | 100% of 100 | decrease |
| s14 | NRc | Corrected core speed | FD001 | -0.665 | 60% of 100 | increase |
| s15 | BPR | Bypass ratio | FD001 | -0.726 | 100% of 100 | increase |
| s4 | T50 | Total temperature at LPT outlet | FD003 | -0.760 | 100% of 100 | increase |
| s7 | P30 | Total pressure at HPC outlet | FD003 | +0.714 | 56% of 100 | decrease |
| s9 | Nc | Physical core speed | FD003 | -0.844 | 84% of 100 | increase |
| s11 | Ps30 | Static pressure at HPC outlet | FD003 | -0.806 | 100% of 100 | increase |
| s12 | phi | Ratio of fuel flow to Ps30 | FD003 | +0.733 | 56% of 100 | decrease |
| s14 | NRc | Corrected core speed | FD003 | -0.837 | 76% of 100 | increase |
| s15 | BPR | Bypass ratio | FD003 | -0.668 | 56% of 100 | increase |

The two sub-datasets agree on 7 of 7 sensors.

Not every direction is attested. These show the measured direction in fewer than 65% of engines, the threshold used in p3_5, so they have no consistent direction to be scored against and are excluded from the benchmark-agreement figures there:

- s14 (NRc) on FD001: 60% of engines, median rho -0.665
- s7 (P30) on FD003: 56% of engines, median rho +0.714
- s12 (phi) on FD003: 56% of engines, median rho +0.733
- s15 (BPR) on FD003: 56% of engines, median rho -0.668

These clear the threshold without being strongly attested, and are kept:

- s9 (Nc) on FD001: 71% of engines, median rho -0.744
- s14 (NRc) on FD003: 76% of engines, median rho -0.837


## Consequence for the faithfulness analysis

- **s4** (T50, Total temperature at LPT outlet), not named in the prompt: measured direction increase, attested in 100 to 100% of engines.
- **s7** (P30, Total pressure at HPC outlet), not named in the prompt: measured direction decrease, attested in 56 to 100% of engines.
- **s9** (Nc, Physical core speed), named in the prompt: measured direction increase, attested in 71 to 84% of engines.
- **s11** (Ps30, Static pressure at HPC outlet), named in the prompt: measured direction increase, attested in 100 to 100% of engines.
- **s12** (phi, Ratio of fuel flow to Ps30), named in the prompt: measured direction decrease, attested in 56 to 100% of engines.
- **s14** (NRc, Corrected core speed), named in the prompt: measured direction increase, attested in 60 to 76% of engines.
- **s15** (BPR, Bypass ratio), named in the prompt: measured direction increase, attested in 56 to 100% of engines.

The canonical direction used for scoring is the measured one. Where the two sub-datasets or the engines within one of them disagree, that is stated rather than smoothed over.
