# MAMS suite (multi-seed summary)

- Seeds: `42, 0, 123`
- Backends: `nextstat_mams,nextstat_nuts`
- Config: `chains=4`, `warmup=3500`, `samples=2000`, `dataset_seed=12345`, `target_accept=0.985`, `parity_warn_z=8.0`, `parity_fail_z=12.0`

Metrics are aggregated across sampler seeds as mean ± std (where available).
`dataset_seed` stays fixed so repeatability reflects sampler variation, not regenerated data variation.

## Aggregate table

| Case | Backend | Statuses | ESS/grad | ESS/s | Warm ESS/s | Wall (s) | min ESS_bulk | max R-hat |
|---|---|---|---:|---:|---:|---:|---:|---:|
| eight_schools | nextstat_mams | `ok,ok,ok` | 0.138 ± 0.000795 | 28994 ± 4012 | 49513 ± 1673 | 0.153 ± 0.0121 | 4413 ± 284 | 1.007 ± 0.000852 |
| eight_schools | nextstat_nuts | `ok,ok,ok` | 0.0305 ± 0.00374 | 14678 ± 809 | 17516 ± 1175 | 0.354 ± 0.0312 | 5188 ± 306 | 1.001 ± 0.000257 |
| glm_logistic | nextstat_mams | `ok,ok,ok` | 0.077 ± 0.00662 | 1206 ± 102 | 1246 ± 77.0 | 1.532 ± 0.00453 | 1848 ± 159 | 1.005 ± 0.00095 |
| glm_logistic | nextstat_nuts | `ok,ok,ok` | 0.0748 ± 0.00316 | 4149 ± 239 | 4263 ± 273 | 1.829 ± 0.0258 | 7583 ± 328 | 1.001 ± 0.000525 |
| neal_funnel_2d | nextstat_mams | `ok,ok,ok` | 0.000539 ± 0.000208 | 1733 ± 860 | 1869 ± 1037 | 0.829 ± 0.434 | 1211 ± 173 | 1.006 ± 0.00284 |
| neal_funnel_2d | nextstat_nuts | `ok,ok,ok` | 2.52e-05 ± 1.04e-05 | 12.4 ± 2.328 | 14.9 ± 1.733 | 1.022 ± 0.185 | 12.4 ± 1.372 | 1.335 ± 0.0541 |
| std_normal_10d | nextstat_mams | `ok,ok,ok` | 0.276 ± 0.0096 | 51581 ± 2595 | 92179 ± 836 | 0.129 ± 0.00392 | 6630 ± 230 | 1.008 ± 0.00116 |
| std_normal_10d | nextstat_nuts | `ok,ok,ok` | 0.0711 ± 0.000737 | 32700 ± 3862 | 44004 ± 657 | 0.247 ± 0.0305 | 8000 ± 0 | 1.002 ± 0.00145 |

## Health Summary

| Case | Backend | Worst ESS/s | Worst min ESS_bulk | Worst min ESS_tail | Worst R-hat | Worst accept rate |
|---|---|---:|---:|---:|---:|---:|
| eight_schools | nextstat_mams | 24649 | 4116 | 1770 | 1.008 | 0.979 |
| eight_schools | nextstat_nuts | 14067 | 4875 | 3121 | 1.002 | 0.984 |
| glm_logistic | nextstat_mams | 1118 | 1715 | 2355 | 1.006 | 0.982 |
| glm_logistic | nextstat_nuts | 3990 | 7375 | 3537 | 1.002 | 0.978 |
| neal_funnel_2d | nextstat_mams | 1098 | 1093 | 491 | 1.009 | 0.973 |
| neal_funnel_2d | nextstat_nuts | 9.845 | 11.5 | 12.7 | 1.392 | 0.484 |
| std_normal_10d | nextstat_mams | 50029 | 6384 | 6021 | 1.009 | 0.982 |
| std_normal_10d | nextstat_nuts | 28477 | 8000 | 4951 | 1.004 | 0.98 |

## Parity Summary

| Case | Statuses | max z | Worst max z |
|---|---|---:|---:|
| eight_schools | `ok,ok,ok` | 2.421 ± 0.462 | 2.946 |
| glm_logistic | `ok,ok,ok` | 1.588 ± 0.602 | 2.024 |
| neal_funnel_2d | `ok,ok,ok` | 3.22 ± 0.888 | 3.949 |
| std_normal_10d | `ok,ok,ok` | 0.75 ± 0.107 | 0.871 |

## Notes

- If some seeds produced `warn`/`failed`, inspect the per-seed `mams_suite.json` under each `seed_*` directory.
- `--reuse-existing` regenerates the summary from existing `seed_*` artifacts without rerunning the suite.
- Parity rows aggregate the tracked NextStat MAMS vs NextStat NUTS posterior mean z-score comparison from each per-seed suite run.

