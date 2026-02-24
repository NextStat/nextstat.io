# EPYC NS vs Stan (v0.9.6) — 3-seed Summary

Date: 2026-02-21
Host: AMD EPYC 7502P (32C/64T)

Methodology:
- Seeds: 42, 123, 777
- Chains: 4
- Warmup/Sampling: 1000 / 1000
- NextStat metric: diagonal
- Target parity:
  - Eight Schools: non-centered in both engines
  - GLM logistic: alpha~N(0,5), beta_j~N(0,2.5) in both engines

## Run A: GLM n=1000, p=10
Artifact:
- `ns_vs_stan_epyc_v096_glm1000_20260221T021540Z/bench_nuts_vs_cmdstan.json`

| Model | NS ESS_bulk/s | Stan ESS_bulk/s | NS/Stan | NS wall (median) | Stan wall (median) | NS R-hat | Stan R-hat | NS div% | Stan div% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| std_normal_10d | 162,893 | 99,327 | 1.64x | 0.025s | 0.071s | 1.0037 | 1.0035 | 0.00 | 0.00 |
| eight_schools | 67,399 | 27,118 | 2.49x | 0.037s | 0.084s | 1.0035 | 1.0028 | 0.03 | 0.00 |
| glm_logistic | 6,599 | 4,452 | 1.48x | 0.445s | 0.707s | 1.0026 | 1.0026 | 0.00 | 0.00 |

## Run B: GLM n=5000, p=20
Artifact:
- `ns_vs_stan_epyc_v096_glm5000_20260221T021711Z/bench_nuts_vs_cmdstan.json`

| Model | NS ESS_bulk/s | Stan ESS_bulk/s | NS/Stan | NS wall (median) | Stan wall (median) | NS R-hat | Stan R-hat | NS div% | Stan div% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| std_normal_10d | 127,058 | 92,755 | 1.37x | 0.031s | 0.072s | 1.0037 | 1.0035 | 0.00 | 0.00 |
| eight_schools | 57,348 | 26,154 | 2.19x | 0.049s | 0.087s | 1.0035 | 1.0028 | 0.03 | 0.00 |
| glm_logistic | 462 | 364 | 1.27x | 2.719s | 3.852s | 1.0025 | 1.0025 | 0.00 | 0.00 |
