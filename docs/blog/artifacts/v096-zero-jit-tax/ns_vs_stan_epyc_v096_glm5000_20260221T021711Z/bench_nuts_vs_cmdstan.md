# NUTS v11: NextStat vs CmdStan

Seeds: [42, 123, 777] | Chains: 4 | Warmup: 1000 | Samples: 1000

GLM config: N=5000, P=20

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
| std_normal_10d | 127058 | 92755 | 1.37x | 1.0037 | 1.0035 | 0.0% | 0.0% |
| eight_schools | 57348 | 26154 | 2.19x | 1.0035 | 1.0028 | 0.0% | 0.0% |
| glm_logistic | 462 | 364 | 1.27x | 1.0025 | 1.0025 | 0.0% | 0.0% |


## std_normal_10d

- NS wall time (median): 0.03s
- NS min ESS_bulk/s: 127058
- NS min ESS_tail/s: 67631
- NS metric: diagonal
- Stan wall time (median): 0.07s
- Stan min ESS_bulk/s: 92755
- Stan min ESS_tail/s: 36549

## eight_schools

- NS wall time (median): 0.05s
- NS min ESS_bulk/s: 57348
- NS min ESS_tail/s: 47629
- NS metric: diagonal
- Stan wall time (median): 0.09s
- Stan min ESS_bulk/s: 26154
- Stan min ESS_tail/s: 20513

## glm_logistic

- NS wall time (median): 2.72s
- NS min ESS_bulk/s: 462
- NS min ESS_tail/s: 661
- NS metric: diagonal
- Stan wall time (median): 3.85s
- Stan min ESS_bulk/s: 364
- Stan min ESS_tail/s: 570
