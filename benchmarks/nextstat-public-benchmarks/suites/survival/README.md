# Survival Analysis Suite (Seed)

This directory contains the **survival analysis benchmark suite** for the standalone public benchmarks repo.

Scope (seed):
- Cox Proportional Hazards (Efron ties) with coefficient parity
- Kaplan-Meier survival curves with confidence intervals
- Log-rank test for group comparison
- Weibull AFT parametric model

This suite produces **publishable** JSON artifacts under pinned schemas:
- `nextstat.survival_benchmark_result.v1` (per case)
- `nextstat.survival_benchmark_suite_result.v1` (suite index)

## Cases

| Case | Kind | n | p | Description |
|------|------|---|---|-------------|
| `cox_ph_1k_5p` | cox_ph | 1000 | 5 | Cox PH, small model |
| `cox_ph_10k_10p` | cox_ph | 10000 | 10 | Cox PH, medium model |
| `kaplan_meier_1k` | kaplan_meier | 1000 | - | KM curves + log-rank (2 groups) |
| `weibull_aft_1k` | weibull_aft | 1000 | - | Parametric Weibull AFT |

## Run

```bash
# Full suite
python3 suites/survival/suite.py --deterministic --out-dir out/survival

# Single case
python3 suites/survival/run.py --case cox_ph_1k_5p --kind cox_ph --n 1000 --p 5 --out out/survival/cox_ph_1k_5p.json
```

## Parity Baselines

Optional baselines (skipped with `status="warn"` if missing):
- `lifelines` for Cox PH, Kaplan-Meier, and Weibull AFT
- `scikit-survival` for Cox PH (secondary, coefficients only)

Install baselines:
```bash
pip install lifelines scikit-survival pandas
```

## Parity Metrics

- **Cox PH**: `coef_max_abs_diff`, `coef_max_rel_diff`, `partial_loglik_diff` vs lifelines; optional `sksurv_coef_max_abs_diff` vs scikit-survival
- **Kaplan-Meier**: `survival_max_abs_diff` at common time points vs lifelines
- **Weibull AFT**: raw parameter comparison (parameterisations differ between libraries)
