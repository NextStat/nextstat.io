# MAMS Stress Assessment

Source summary: `mams_stress_multiseed_summary.json`

## Stress readiness

- status: `failed`
- supported repeatability: `failed`
- pathological controls: `passed`

## Supported repeatability gate

- status: `failed`
- backend: `nextstat_mams`
- max_r_hat threshold: `1.01`
- failing cases: `hier_random_intercept_non_centered`

| Metric | Worst case | Observed | Policy |
|---|---|---:|---:|
| max_r_hat | hier_random_intercept_non_centered | 1.0158 | 1.01 |
| min_ess_bulk | hier_random_intercept_non_centered | 2,041 | — |
| min_ess_tail | hier_random_intercept_non_centered | 863 | — |
| ess_per_sec | hier_random_intercept_non_centered | 710 | — |
| accept_rate | hier_random_intercept_non_centered | 0.9801 | — |
| parity_max_z | hier_random_intercept_non_centered | 4.0899 | 8 |

| Case | Statuses | Worst max R-hat | Worst min ESS_bulk | Worst ESS/s |
|---|---|---:|---:|---:|
| hier_random_intercept_non_centered | `ok,ok,ok` | 1.0158 | 2,041 | 710 |
| neal_funnel_ncp_10d | `ok,ok,ok` | 1.0066 | 6,335 | 56,540 |

| Case | Parity statuses | Worst max z |
|---|---|---:|
| hier_random_intercept_non_centered | `ok,ok,ok` | 4.0899 |
| neal_funnel_ncp_10d | `ok,ok,ok` | 1.2717 |

### Supported repeatability failures

- `hier_random_intercept_non_centered`: `max_r_hat_exceeds_threshold` (observed `1.0158`, threshold `1.01`)

## Pathological controls

- status: `passed`

| Case | Statuses | Worst max R-hat | Worst min ESS_bulk | Worst ESS/s |
|---|---|---:|---:|---:|
| neal_funnel_10d_centered | `ok,ok,ok` | 1.8611 | 1,419 | 706 |
