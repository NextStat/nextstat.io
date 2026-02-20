# NUTS v11: NextStat vs CmdStan

Seeds: [42, 123, 777] | Chains: 4 | Warmup: 1000 | Samples: 1000

GLM config: N=1000, P=10

Environment:
- Python: 3.12.3 (main, Jan 22 2026, 20:57:42) [GCC 13.3.0]
- Platform: Linux 6.8.0-90-generic (x86_64)
- Git commit: e4c6c9f5d49623a9dcf2d6b3967682a62676ab48
- NextStat: 0.9.6
- CmdStan path: /root/.cmdstan/cmdstan-2.38.0

Target parity:
- Eight Schools: non-centered in both engines
- GLM logistic: alpha~N(0,5), beta_j~N(0,2.5) in both engines

| Model | NS ESS_bulk/s | Stan ESS_bulk/s | Ratio | NS R-hat | Stan R-hat | NS div% | Stan div% |
|-------|--------------|-----------------|-------|----------|------------|---------|-----------|
| std_normal_10d | 162893 | 99327 | 1.64x | 1.0037 | 1.0035 | 0.0% | 0.0% |
| eight_schools | 67399 | 27118 | 2.49x | 1.0035 | 1.0028 | 0.0% | 0.0% |
| glm_logistic | 6599 | 4452 | 1.48x | 1.0026 | 1.0026 | 0.0% | 0.0% |


## std_normal_10d

- NS wall time (median): 0.02s
- NS min ESS_bulk/s: 162893
- NS min ESS_tail/s: 90354
- NS metric: diagonal
- Stan wall time (median): 0.07s
- Stan min ESS_bulk/s: 99327
- Stan min ESS_tail/s: 37956

## eight_schools

- NS wall time (median): 0.04s
- NS min ESS_bulk/s: 67399
- NS min ESS_tail/s: 46575
- NS metric: diagonal
- Stan wall time (median): 0.08s
- Stan min ESS_bulk/s: 27118
- Stan min ESS_tail/s: 19734

## glm_logistic

- NS wall time (median): 0.45s
- NS min ESS_bulk/s: 6599
- NS min ESS_tail/s: 5241
- NS metric: diagonal
- Stan wall time (median): 0.71s
- Stan min ESS_bulk/s: 4452
- Stan min ESS_tail/s: 3285
