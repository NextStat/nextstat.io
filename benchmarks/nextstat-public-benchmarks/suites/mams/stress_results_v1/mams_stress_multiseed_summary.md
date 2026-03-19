# MAMS Stress Multi-Seed Summary

Config: seeds=`42,0,123`, backends=`nextstat_mams,nextstat_nuts`, n_chains=`4`, warmup=`3500`, samples=`2000`, dataset_seed=`12345`, target_accept=`0.985`

Seed semantics: `config.seed` / `config.benchmark_seed` is the requested benchmark seed; cold start uses that seed, warm start uses `seed+1`, and reported posterior/diagnostic metrics come from `config.reported_draws_seed`.

## Case Catalog

| Case | Tier | Parity scope | Description |
|---|---|---|---|
| neal_funnel_10d_centered | pathological_control | informational | Centered 10D funnel hard-geometry control. |
| neal_funnel_ncp_10d | supported | required | Non-centered 10D funnel supported repeatability case. |
| hier_random_intercept_non_centered | supported | required | Hierarchical logistic random intercept stress case. |

## Aggregate Cases

| Case | Tier | Backend | Statuses | Config overrides | ESS/s mean ± sd | min ESS_bulk worst | max R-hat worst |
|---|---|---|---|---|---:|---:|---:|
| hier_random_intercept_non_centered | supported | nextstat_mams | `ok,ok,ok` | init_l=2.0 | 643 ± 191 | 600 | 1.007 |
| hier_random_intercept_non_centered | supported | nextstat_nuts | `ok,ok,ok` | — | 426 ± 36.4 | 3258 | 1.003 |
| neal_funnel_10d_centered | pathological_control | nextstat_mams | `ok,ok,ok` | — | 3101 ± 2125 | 1419 | 1.861 |
| neal_funnel_10d_centered | pathological_control | nextstat_nuts | `ok,ok,ok` | — | 5.689 ± 3.415 | 5.691 | 2.046 |
| neal_funnel_ncp_10d | supported | nextstat_mams | `ok,ok,ok` | — | 48454 ± 1931 | 6335 | 1.007 |
| neal_funnel_ncp_10d | supported | nextstat_nuts | `ok,ok,ok` | — | 28973 ± 2362 | 8000 | 1.002 |

## Parity Summary

| Case | Tier | Scope | Parity statuses | max z mean ± sd | worst max z |
|---|---|---|---|---:|---:|
| hier_random_intercept_non_centered | supported | required | `ok,ok,ok` | 3.352 ± 1.141 | 4.663 |
| neal_funnel_10d_centered | pathological_control | informational | `ok,ok,ok` | 4.183 ± 1.385 | 5.177 |
| neal_funnel_ncp_10d | supported | required | `ok,ok,ok` | 1.143 ± 0.115 | 1.272 |
