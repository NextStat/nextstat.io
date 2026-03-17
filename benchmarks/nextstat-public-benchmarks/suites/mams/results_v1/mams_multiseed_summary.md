# MAMS suite (multi-seed summary)

- Seeds: `42, 0, 123`
- Backends: `nextstat_mams,nextstat_nuts`
- Config: `chains=4`, `warmup=3500`, `samples=2000`, `dataset_seed=12345`, `target_accept=0.985`, `parity_warn_z=8.0`, `parity_fail_z=12.0`

Metrics are aggregated across sampler seeds as mean ± std (where available).
`dataset_seed` stays fixed so repeatability reflects sampler variation, not regenerated data variation.

## Aggregate table

| Case | Backend | Statuses | ESS/grad | ESS/s | Warm ESS/s | Wall (s) | min ESS_bulk | max R-hat |
|---|---|---|---:|---:|---:|---:|---:|---:|
| eight_schools | nextstat_mams | `ok,ok,ok` | 0.138 ± 0.000795 | 30538 ± 3676 | 50894 ± 2314 | 0.145 ± 0.00849 | 4413 ± 284 | 1.007 ± 0.000852 |
| eight_schools | nextstat_nuts | `ok,ok,ok` | 0.0305 ± 0.00374 | 14378 ± 128 | 17397 ± 1076 | 0.361 ± 0.0226 | 5188 ± 306 | 1.001 ± 0.000257 |
| glm_logistic | nextstat_mams | `ok,ok,ok` | 0.077 ± 0.00662 | 1208 ± 123 | 1276 ± 106 | 1.532 ± 0.053 | 1848 ± 159 | 1.005 ± 0.00095 |
| glm_logistic | nextstat_nuts | `ok,ok,ok` | 0.0748 ± 0.00316 | 4234 ± 188 | 4264 ± 295 | 1.791 ± 0.03 | 7583 ± 328 | 1.001 ± 0.000525 |
| neal_funnel_2d | nextstat_mams | `ok,ok,ok` | 0.000539 ± 0.000208 | 1756 ± 840 | 2007 ± 1142 | 0.812 ± 0.424 | 1211 ± 173 | 1.006 ± 0.00284 |
| neal_funnel_2d | nextstat_nuts | `ok,ok,ok` | 2.52e-05 ± 1.04e-05 | 12.5 ± 2.233 | 15.0 ± 1.87 | 1.005 ± 0.173 | 12.4 ± 1.372 | 1.335 ± 0.0541 |
| std_normal_10d | nextstat_mams | `ok,ok,ok` | 0.276 ± 0.0096 | 50037 ± 914 | 93362 ± 5637 | 0.133 ± 0.007 | 6630 ± 230 | 1.008 ± 0.00116 |
| std_normal_10d | nextstat_nuts | `ok,ok,ok` | 0.0711 ± 0.000737 | 32080 ± 3229 | 43753 ± 86.0 | 0.251 ± 0.0267 | 8000 ± 0 | 1.002 ± 0.00145 |

## Health Summary

| Case | Backend | Worst ESS/s | Worst min ESS_bulk | Worst min ESS_tail | Worst R-hat | Worst accept rate |
|---|---|---:|---:|---:|---:|---:|
| eight_schools | nextstat_mams | 26637 | 4116 | 1770 | 1.008 | 0.979 |
| eight_schools | nextstat_nuts | 14239 | 4875 | 3121 | 1.002 | 0.984 |
| glm_logistic | nextstat_mams | 1080 | 1715 | 2355 | 1.006 | 0.982 |
| glm_logistic | nextstat_nuts | 4056 | 7375 | 3537 | 1.002 | 0.978 |
| neal_funnel_2d | nextstat_mams | 1118 | 1093 | 491 | 1.009 | 0.973 |
| neal_funnel_2d | nextstat_nuts | 10.1 | 11.5 | 12.7 | 1.392 | 0.484 |
| std_normal_10d | nextstat_mams | 49206 | 6384 | 6021 | 1.009 | 0.982 |
| std_normal_10d | nextstat_nuts | 28392 | 8000 | 4951 | 1.004 | 0.98 |

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

