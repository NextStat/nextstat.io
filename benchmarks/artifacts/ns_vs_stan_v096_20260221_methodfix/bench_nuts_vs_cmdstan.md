# NUTS v11: NextStat vs CmdStan

Seeds: [42, 123, 777] | Chains: 4 | Warmup: 1000 | Samples: 1000

GLM config: N=1000, P=10

Environment:
- Python: 3.12.3 (main, Nov  6 2025, 13:44:16) [GCC 13.3.0]
- Platform: Linux 6.5.0-35-generic (x86_64)
- Git commit: None
- NextStat: 0.9.6
- CmdStan path: /workspace/nextstat.io/.internal/cmdstan/cmdstan-2.38.0

Target parity:
- Eight Schools: non-centered in both engines
- GLM logistic: alpha~N(0,5), beta_j~N(0,2.5) in both engines

| Model | NS ESS_bulk/s | Stan ESS_bulk/s | Ratio | NS R-hat | Stan R-hat | NS div% | Stan div% |
|-------|--------------|-----------------|-------|----------|------------|---------|-----------|
| std_normal_10d | 89989 | 79274 | 1.14x | 1.0040 | 1.0035 | 0.0% | 0.0% |
| eight_schools | 44191 | 19540 | 2.26x | 1.0023 | 1.0028 | 0.0% | 0.0% |
| glm_logistic | 2929 | 4004 | 0.73x | 1.0020 | 1.0026 | 0.0% | 0.0% |


## std_normal_10d

- NS wall time (median): 0.04s
- NS min ESS_bulk/s: 89989
- NS min ESS_tail/s: 30387
- NS metric: diagonal
- Stan wall time (median): 0.09s
- Stan min ESS_bulk/s: 79274
- Stan min ESS_tail/s: 29460

## eight_schools

- NS wall time (median): 0.06s
- NS min ESS_bulk/s: 44191
- NS min ESS_tail/s: 31271
- NS metric: diagonal
- Stan wall time (median): 0.12s
- Stan min ESS_bulk/s: 19540
- Stan min ESS_tail/s: 15325

## glm_logistic

- NS wall time (median): 1.01s
- NS min ESS_bulk/s: 2929
- NS min ESS_tail/s: 2217
- NS metric: diagonal
- Stan wall time (median): 0.79s
- Stan min ESS_bulk/s: 4004
- Stan min ESS_tail/s: 3001
