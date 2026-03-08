# MAMS Stress Multi-Seed Summary

Config: seeds=`42,0,123`, backends=`nextstat_mams,nextstat_nuts`, n_chains=`4`, warmup=`3500`, samples=`2000`, dataset_seed=`12345`, target_accept=`0.985`

## Case Catalog

| Case | Tier | Parity scope | Description |
|---|---|---|---|
| neal_funnel_10d_centered | pathological_control | informational | Centered 10D funnel hard-geometry control. |
| neal_funnel_ncp_10d | supported | required | Non-centered 10D funnel supported repeatability case. |
| hier_random_intercept_non_centered | supported | required | Hierarchical logistic random intercept stress case. |

## Aggregate Cases

| Case | Tier | Backend | Statuses | ESS/s mean ± sd | min ESS_bulk worst | max R-hat worst |
|---|---|---|---|---:|---:|---:|
| hier_random_intercept_non_centered | supported | nextstat_mams | `ok,ok,ok` | 799 ± 81.2 | 2041 | 1.016 |
| hier_random_intercept_non_centered | supported | nextstat_nuts | `ok,ok,ok` | 457 ± 34.3 | 3258 | 1.003 |
| neal_funnel_10d_centered | pathological_control | nextstat_mams | `ok,ok,ok` | 3561 ± 2491 | 1419 | 1.861 |
| neal_funnel_10d_centered | pathological_control | nextstat_nuts | `ok,ok,ok` | 6.176 ± 3.771 | 5.691 | 2.046 |
| neal_funnel_ncp_10d | supported | nextstat_mams | `ok,ok,ok` | 62010 ± 8216 | 6335 | 1.007 |
| neal_funnel_ncp_10d | supported | nextstat_nuts | `ok,ok,ok` | 34535 ± 2470 | 8000 | 1.002 |

## Parity Summary

| Case | Tier | Scope | Parity statuses | max z mean ± sd | worst max z |
|---|---|---|---|---:|---:|
| hier_random_intercept_non_centered | supported | required | `ok,ok,ok` | 3.797 ± 0.292 | 4.09 |
| neal_funnel_10d_centered | pathological_control | informational | `ok,ok,ok` | 4.183 ± 1.385 | 5.177 |
| neal_funnel_ncp_10d | supported | required | `ok,ok,ok` | 1.143 ± 0.115 | 1.272 |
