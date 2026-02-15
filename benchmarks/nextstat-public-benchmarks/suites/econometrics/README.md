# Econometrics Suite (Seed)

This directory is reserved for the **econometrics benchmark suite** in the standalone public benchmarks repo.

Scope (Phase 12 baseline):
- panel fixed effects (within estimator) + 1-way / 2-way cluster SE
- DiD TWFE baseline
- DiD TWFE wild cluster bootstrap (Webb 6-point)
- staggered-adoption DiD baseline (group-time ATT)
- event study TWFE baseline
- IV / 2SLS baseline (HC1 + HAC/Newey-West)
- AIPW baseline

This suite produces **publishable** JSON artifacts under pinned schemas:
- `nextstat.econometrics_benchmark_result.v1` (per case)
- `nextstat.econometrics_benchmark_suite_result.v1` (suite index)

Related spec in the main repo:
- `/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/suites/econometrics.md`

## Run

```bash
python3 suites/econometrics/suite.py --deterministic --out-dir out/econometrics
```

Optional baselines (skipped with `status="warn"` if missing):
- `statsmodels` for OLS-derived cases (panel FE, DiD, event study)
- `linearmodels` for IV/2SLS parity
- `pyfixest` + `wildboottest` for staggered DiD and wild cluster bootstrap parity
