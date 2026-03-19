# MAMS Stress Assessment

Source summary: `mams_stress_multiseed_summary.json`

## Stress readiness

- status: `passed`
- supported repeatability: `passed`
- pathological controls: `passed`

## Supported repeatability gate

- status: `passed`
- backend: `nextstat_mams`
- max_r_hat threshold: `1.01`
- failing cases: `—`

| Metric | Worst case | Observed | Policy |
|---|---|---:|---:|
| max_r_hat | hier_random_intercept_non_centered | 1.0075 | 1.01 |
| min_ess_bulk | hier_random_intercept_non_centered | 600 | — |
| min_ess_tail | hier_random_intercept_non_centered | 777 | — |
| ess_per_sec | hier_random_intercept_non_centered | 433 | — |
| accept_rate | hier_random_intercept_non_centered | 0.9792 | — |
| parity_max_z | hier_random_intercept_non_centered | 4.6635 | 8 |

| Case | Statuses | Worst max R-hat | Worst min ESS_bulk | Worst ESS/s |
|---|---|---:|---:|---:|
| hier_random_intercept_non_centered | `ok,ok,ok` | 1.0075 | 600 | 433 |
| neal_funnel_ncp_10d | `ok,ok,ok` | 1.0066 | 6,335 | 46,668 |

| Case | Parity statuses | Worst max z |
|---|---|---:|
| hier_random_intercept_non_centered | `ok,ok,ok` | 4.6635 |
| neal_funnel_ncp_10d | `ok,ok,ok` | 1.2717 |

## Pathological controls

- status: `passed`

| Case | Statuses | Worst max R-hat | Worst min ESS_bulk | Worst ESS/s |
|---|---|---:|---:|---:|
| neal_funnel_10d_centered | `ok,ok,ok` | 1.8611 | 1,419 | 663 |
