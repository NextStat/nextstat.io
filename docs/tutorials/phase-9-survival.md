---
title: "Phase 9: Survival Analysis (Parametric + Cox PH) — Tutorial"
status: draft
---

# Phase 9: Survival Analysis (Parametric + Cox PH)

This tutorial documents the baseline survival analysis support in NextStat:

- Parametric right-censoring models: exponential, Weibull, log-normal AFT
- Cox proportional hazards (Cox PH) via partial likelihood with explicit ties policy

The implementation goal is a dependency-light, reproducible baseline with deterministic behavior
and contract tests.

## What is implemented

Core models (low-level; use `nextstat.fit(...)` for MLE):

- `nextstat.ExponentialSurvivalModel(times, events)`
- `nextstat.WeibullSurvivalModel(times, events)`
- `nextstat.LogNormalAftModel(times, events)`
- `nextstat.CoxPhModel(times, events, x, ties=...)`

Notes:

- `events[i] = False` means right-censored at `times[i]`.
- Cox PH uses **partial likelihood** (no baseline hazard) and does not include an intercept.
- Cox PH ties policy is explicit: `ties="breslow"` or `ties="efron"`.

## Prerequisites

From the repo root:

```bash
cargo test -p ns-inference
./.venv/bin/maturin develop --release -m bindings/ns-py/Cargo.toml
```

## Quick start: Cox PH (Python)

```python
import math
import nextstat

# Example with ties and censoring.
times = [2.0, 1.0, 1.0, 0.5, 0.5, 0.2]
events = [True, True, False, True, False, False]
x = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, -1.0],
    [0.0, -1.0],
    [0.5, 0.5],
]

fit = nextstat.survival.cox_ph.fit(times, events, x, ties="efron", robust=True)

beta_hat = [float(v) for v in fit.coef]
print("converged:", fit.converged, "nll:", fit.nll)
print("beta_hat:", beta_hat)

# Hazard ratios (per 1-unit increase in each covariate).
print("hazard ratios:", fit.hazard_ratios())

# Wald confidence intervals.
print("beta CI:", fit.confint(level=0.95, robust=True))
print("HR CI:", fit.hazard_ratio_confint(level=0.95, robust=True))
```

### Choosing a ties policy

If your data has repeated event times (common with discretized time), you must choose a ties
approximation:

- `ties="breslow"`: simpler, widely used
- `ties="efron"`: often more accurate with many ties

Both are deterministic; the choice is part of the model contract.

## Quick start: parametric right-censoring models

Parametric models are intercept-only baselines (no covariates yet). Use `nextstat.fit(...)`.

```python
import nextstat

times = [0.5, 1.2, 0.7, 2.0, 0.9]
events = [True, False, True, False, True]

fit_exp = nextstat.survival.exponential.fit(times, events)
print("exp nll:", fit_exp.nll, "params:", fit_exp.params)

m_w = nextstat.WeibullSurvivalModel(times, events)
fit_w = nextstat.fit(m_w)
print("weibull nll:", fit_w.nll, "params:", fit_w.bestfit)
```

## Validation and regression tests

NextStat keeps runtime tests dependency-light. For Cox PH, the Python regression suite includes a
pure-Python reference implementation (Breslow/Efron) and checks:

- NLL matches reference (tight tolerance)
- analytic gradient matches finite differences (smoke-level tolerance)
- invariance to row permutations

Run:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q \
  tests/python/test_survival_contract.py
```

### Optional parity (statsmodels)

If you have `statsmodels` installed, NextStat includes an optional parity test against
`statsmodels.duration.hazard_regression.PHReg`:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q \
  tests/python/test_survival_cox_statsmodels_parity.py
```

## Notes and limitations (baseline)

- Cox PH uses partial likelihood; the baseline hazard is not modeled, so absolute survival curves
  are not produced in this baseline.
- No robust SE / sandwich variance yet for Cox PH.
- Parametric models currently do not include covariates (intercept-only baselines).
- Left truncation and time-varying covariates are out of scope for this baseline.
