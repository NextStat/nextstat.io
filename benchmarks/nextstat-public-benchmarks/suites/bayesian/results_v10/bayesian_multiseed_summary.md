# Bayesian suite (multi-seed summary)

- Seeds: `42, 0, 123`
- Backends: `cmdstanpy,nextstat`
- Config: `chains=4`, `warmup=1000`, `samples=2000`, `max_treedepth=10`, `target_accept=0.8`, `init_jitter_rel=0.1`

Metrics are aggregated across seeds as mean ± std (where available).

## Aggregate table

| Case | Backend | Statuses | min ESS_bulk/s | Wall (s) | min ESS_bulk | max R-hat |
|---|---|---|---:|---:|---:|---:|
| eight_schools_non_centered | cmdstanpy | `ok,ok,ok` | 16783 ± 2254 | 0.305 ± 0.00687 | 5115 ± 582 | 1.001 ± 0.000393 |
| eight_schools_non_centered | nextstat | `ok,ok,ok` | 28510 ± 1591 | 0.209 ± 0.00416 | 5972 ± 443 | 1.002 ± 0.000269 |
| glm_logistic_regression | cmdstanpy | `ok,ok,ok` | 15280 ± 713 | 0.526 ± 0.0296 | 8029 ± 128 | 1.001 ± 0.000713 |
| glm_logistic_regression | nextstat | `ok,ok,ok` | 14640 ± 328 | 0.547 ± 0.0124 | 8000 ± 0 | 1.001 ± 0.000254 |
| hier_random_intercept_non_centered | cmdstanpy | `ok,ok,ok` | 504 ± 89.8 | 4.637 ± 0.0755 | 2332 ± 388 | 1.002 ± 0.000208 |
| hier_random_intercept_non_centered | nextstat | `ok,ok,ok` | 1680 ± 40.0 | 2.024 ± 0.151 | 3397 ± 225 | 1.002 ± 0.000989 |
| histfactory_simple_8p | cmdstanpy | `ok,ok,ok` | 24391 ± 2994 | 0.155 ± 0.00858 | 3775 ± 511 | 1.001 ± 0.00018 |
| histfactory_simple_8p | nextstat | `ok,ok,ok` | 25794 ± 2758 | 0.178 ± 0.0117 | 4569 ± 201 | 1.001 ± 0.000384 |

## Health Summary

| Case | Backend | Worst divergence | Worst treedepth hit rate | Worst R-hat | Worst min E-BFMI | Worst min ESS_tail |
|---|---|---:|---:|---:|---:|---:|
| eight_schools_non_centered | cmdstanpy | 0 | 0 | 1.002 | — | 3110 |
| eight_schools_non_centered | nextstat | 0 | 0 | 1.002 | 0.903 | 3993 |
| glm_logistic_regression | cmdstanpy | 0 | 0 | 1.002 | — | 5233 |
| glm_logistic_regression | nextstat | 0 | 0 | 1.002 | 1.018 | 5307 |
| hier_random_intercept_non_centered | cmdstanpy | 0 | 0 | 1.002 | — | 1936 |
| hier_random_intercept_non_centered | nextstat | 0 | 0 | 1.003 | 0.7 | 3592 |
| histfactory_simple_8p | cmdstanpy | 0 | 0 | 1.001 | — | 2125 |
| histfactory_simple_8p | nextstat | 0 | 0 | 1.002 | 0.905 | 2721 |

## Notes

- If some seeds produced `warn`/`failed`, inspect the per-seed `bayesian_suite.json` under each `seed_*` directory.
- `--reuse-existing` regenerates the summary from existing `seed_*` artifacts without rerunning the suite.
- Publishable snapshots should pin toolchains and report exact versions; this summary is meant for quick stability checks.

