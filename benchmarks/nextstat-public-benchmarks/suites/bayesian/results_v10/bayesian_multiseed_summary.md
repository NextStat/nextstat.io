# Bayesian suite (multi-seed summary)

- Seeds: `42, 0, 123`
- Backends: `nextstat,cmdstanpy`
- Config: `chains=4`, `warmup=1000`, `samples=2000`, `max_treedepth=10`, `target_accept=0.8`, `init_jitter_rel=0.1`

Metrics are aggregated across seeds as mean ± std (where available).

## Aggregate table

| Case | Backend | Statuses | min ESS_bulk/s | Wall (s) | min ESS_bulk | max R-hat |
|---|---|---|---:|---:|---:|---:|
| eight_schools_non_centered | cmdstanpy | `ok,ok,ok` | 26644 ± 2221 | 0.192 ± 0.015 | 5115 ± 582 | 1.001 ± 0 |
| eight_schools_non_centered | nextstat | `ok,ok,ok` | 54083 ± 4902 | 0.103 ± 0.002 | 5559 ± 425 | 1.002 ± 0.001 |
| glm_logistic_regression | cmdstanpy | `ok,ok,ok` | 29600 ± 1971 | 0.272 ± 0.016 | 8029 ± 128 | 1.001 ± 0.001 |
| glm_logistic_regression | nextstat | `ok,ok,ok` | 29895 ± 668 | 0.268 ± 0.006 | 8000 ± 0 | 1.001 ± 0 |
| hier_random_intercept_non_centered | cmdstanpy | `ok,ok,ok` | 1015 ± 188 | 2.304 ± 0.058 | 2332 ± 388 | 1.002 ± 0 |
| hier_random_intercept_non_centered | nextstat | `ok,ok,ok` | 3255 ± 360 | 0.981 ± 0.061 | 3192 ± 404 | 1.003 ± 0.001 |
| histfactory_simple_8p | cmdstanpy | `warn,warn,warn` | — | 0 ± 0 | — | — |
| histfactory_simple_8p | nextstat | `ok,ok,ok` | 65047 ± 7374 | 0.071 ± 0.008 | 4609 ± 88.0 | 1.001 ± 0 |

## Notes

- If some seeds produced `warn`/`failed`, inspect the per-seed `bayesian_suite.json` under each `seed_*` directory.
- Publishable snapshots should pin toolchains and report exact versions; this summary is meant for quick stability checks.

