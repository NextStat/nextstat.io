# MAMS Multi-Seed Assessment

Source summary: `mams_multiseed_summary.json`

## Repeatability gate

- status: `passed`
- backend: `nextstat_mams`
- max_r_hat threshold: `1.01`
- reviewed cases: `4`
- reviewed parity cases: `4`
- failing cases: `—`

| Metric | Worst case | Observed | Policy |
|---|---|---:|---:|
| max_r_hat | neal_funnel_2d | 1.0087 | 1.01 |
| min_ess_bulk | neal_funnel_2d | 1,093 | — |
| min_ess_tail | neal_funnel_2d | 491 | — |
| ess_per_sec | glm_logistic | 1,080 | — |
| accept_rate | neal_funnel_2d | 0.9726 | — |
| parity_max_z | neal_funnel_2d | 3.9493 | 8 |

| Case | Statuses | Worst max R-hat | Worst min ESS_bulk | Worst min ESS_tail | Worst ESS/s | Worst accept rate |
|---|---|---:|---:|---:|---:|---:|
| eight_schools | `ok,ok,ok` | 1.0083 | 4,116 | 1,770 | 26,637 | 0.9788 |
| glm_logistic | `ok,ok,ok` | 1.0061 | 1,715 | 2,355 | 1,080 | 0.9819 |
| neal_funnel_2d | `ok,ok,ok` | 1.0087 | 1,093 | 491 | 1,118 | 0.9726 |
| std_normal_10d | `ok,ok,ok` | 1.0086 | 6,384 | 6,021 | 49,206 | 0.9819 |

| Case | Parity statuses | Worst max z |
|---|---|---:|
| eight_schools | `ok,ok,ok` | 2.9462 |
| glm_logistic | `ok,ok,ok` | 2.0236 |
| neal_funnel_2d | `ok,ok,ok` | 3.9493 |
| std_normal_10d | `ok,ok,ok` | 0.871 |
