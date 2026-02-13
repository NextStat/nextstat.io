---
title: "Phase 9: Survival Analysis — Tutorial"
status: draft
---

# Phase 9: Survival Analysis

This tutorial documents the survival analysis support in NextStat:

- **Non-parametric**: Kaplan-Meier estimator, log-rank test
- **Parametric** right-censoring models: exponential, Weibull, log-normal AFT
- **Semi-parametric**: Cox proportional hazards (Cox PH) via partial likelihood with explicit ties policy

The implementation goal is a dependency-light, reproducible baseline with deterministic behavior
and contract tests.

## What is implemented

**Non-parametric estimators** (no model fitting required):

- `nextstat.kaplan_meier(times, events, conf_level=0.95)` — Kaplan-Meier survival curve with Greenwood variance and log-log CIs
- `nextstat.log_rank_test(times, events, groups)` — Mantel-Cox log-rank test (2+ groups)

**Parametric models** (low-level; use `nextstat.fit(model)` for MLE):

- `nextstat.ExponentialSurvivalModel(times, events)`
- `nextstat.WeibullSurvivalModel(times, events)`
- `nextstat.LogNormalAftModel(times, events)`
- `nextstat.CoxPhModel(times, events, x, ties="efron")`

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

## Quick start: Kaplan-Meier (Python)

```python
import nextstat

# Right-censored data (True = event, False = censored).
times  = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 12, 12, 15, 23, 27]
events = [True, True, True, False, True, True, False, True, False,
          True, True, True, False, False, True, True]

km = nextstat.kaplan_meier(times, events, conf_level=0.95)
print("n:", km["n"], "events:", km["n_events"])
print("median survival:", km["median"])

# Step-wise survival curve.
for t, s, lo, hi in zip(km["time"], km["survival"], km["ci_lower"], km["ci_upper"]):
    print(f"  t={t:5.1f}  S={s:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
```

## Quick start: Kaplan-Meier (CLI)

```bash
# Input: JSON with "times" and "events" arrays.
nextstat survival km --input km_data.json
nextstat survival km --input km_data.json --conf-level 0.90
```

## Quick start: Log-rank test (Python)

```python
import nextstat

# R's aml dataset: Maintained vs Nonmaintained chemotherapy.
times  = [9,13,13,18,23,28,31,34,45,48,161, 5,5,8,8,12,16,23,27,30,33,43,45]
events = [True,True,False,True,True,False,True,True,False,True,False,
          True,True,True,True,True,False,True,True,True,True,True,True]
groups = [1]*11 + [2]*12  # 1=Maintained, 2=Nonmaintained

lr = nextstat.log_rank_test(times, events, groups)
print(f"chi² = {lr['chi_squared']:.2f}, df = {lr['df']}, p = {lr['p_value']:.4f}")
for g, o, e in zip(lr["group_ids"], lr["observed"], lr["expected"]):
    print(f"  group {g}: observed={o:.0f}, expected={e:.2f}")
```

## Quick start: Log-rank test (CLI)

```bash
# Input: JSON with "times", "events", and "groups" arrays.
nextstat survival log-rank-test --input logrank_data.json
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

# Baseline survival curve (for a given covariate vector x0).
grid = [0.0, 0.5, 1.0, 2.0]
print("S(t | x0):", fit.predict_survival([[0.0, 0.0]], times=grid)[0])
```

## Quick start: Cox PH (CLI)

The CLI uses a JSON in / JSON out contract. Example input:

```json
{
  "times": [2.0, 1.0, 1.0, 0.5, 0.5, 0.2],
  "events": [true, true, false, true, false, false],
  "x": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0], [0.5, 0.5]],
  "groups": [1, 1, 2, 2, 3, 3]
}
```

Run:

```bash
nextstat survival cox-ph-fit --input cox.json --ties efron
```

Notes:
- If `groups` are provided, robust SE are **cluster-robust** by default.
- Use `--no-robust` to disable sandwich SE, and `--no-cluster-correction` to disable the small-sample factor `G/(G-1)`.

### Choosing a ties policy

If your data has repeated event times (common with discretized time), you must choose a ties
approximation:

- `ties="breslow"`: simpler, widely used
- `ties="efron"`: often more accurate with many ties

Both are deterministic; the choice is part of the model contract.

## Quick start: parametric right-censoring models

Parametric models are intercept-only baselines (no covariates yet). Use the high-level helpers:

```python
import nextstat

times = [0.5, 1.2, 0.7, 2.0, 0.9]
events = [True, False, True, False, True]

fit_exp = nextstat.survival.exponential.fit(times, events)
print("exp nll:", fit_exp.nll, "params:", fit_exp.params)

fit_w = nextstat.survival.weibull.fit(times, events)
print("weibull nll:", fit_w.nll, "params:", fit_w.params)

fit_ln = nextstat.survival.lognormal_aft.fit(times, events)
print("lognormal_aft nll:", fit_ln.nll, "params:", fit_ln.params)
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

JSON runner (Apex2-style; records skipped if missing deps):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
  tests/apex2_survival_statsmodels_report.py --out tmp/apex2_survival_statsmodels_report.json
```

### Cluster-robust SE (optional)

If your data has correlated observations within groups (e.g. subjects, sites), you can request
cluster-robust (sandwich) standard errors by passing `groups=subject_ids` to the fit helper. This does
not change the MLE coefficients; it changes the uncertainty estimate.

```python
fit = nextstat.survival.cox_ph.fit(times, events, x, ties="efron", groups=subject_ids, robust=True)
print("robust kind:", fit.robust_kind)  # "cluster"
print("cluster-robust SE:", fit.robust_se)
```

### PH diagnostics (Schoenfeld residuals)

Schoenfeld residuals are a standard diagnostic for the proportional hazards assumption.
NextStat exposes a dependency-light residual calculator (event rows only):

```python
sr = nextstat.survival.cox_ph_schoenfeld(times, events, x, ties="efron", coef=fit.coef)
print("corr(log time):", sr.corr_log_time())
print("slope(log time):", sr.slope_log_time())
print("PH test:", nextstat.survival.cox_ph_ph_test(times, events, x, ties="efron", coef=fit.coef))
```

## Notes and limitations (baseline)

- Cox PH uses partial likelihood; the baseline hazard is not modeled, so absolute survival curves
  require a baseline hazard estimate. NextStat outputs a baseline cumulative hazard estimate
  (Breslow/Efron increments) to enable `S(t | x)` prediction in the baseline implementation.
- Robust (sandwich) SE are supported (HC0 or cluster-robust when `groups` are provided).
- Parametric models currently do not include covariates (intercept-only baselines).
- Left truncation and time-varying covariates are out of scope for this baseline.
