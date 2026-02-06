---
title: "Phase 12: Econometrics (Panel FE Baseline)"
---

# Panel fixed effects (within estimator)

NextStat includes a minimal panel linear regression baseline with fixed effects (FE) using the **within estimator**:

- Model: `y_it = alpha_i + x_it' beta + eps_it`
- FE `alpha_i` captures time-invariant unobserved heterogeneity per entity (e.g. user, firm, region).
- Estimation is done by demeaning within each entity and running OLS on transformed data.

## When to use FE

Use fixed effects when:
- you suspect **entity-specific confounders** correlated with regressors `x_it`, and
- you have repeated observations per entity over time.

Avoid 1-way FE when:
- your regressors are time-invariant within entities (they will be absorbed / not identifiable), or
- you need random-effects style partial pooling (use hierarchical models instead).

## API

Two entry points:

```python
import nextstat

# 1) Raw X/y
fit = nextstat.econometrics.panel_fe_fit(
    x=[[...], [...], ...],
    y=[...],
    entity=[...],
    time=[...],          # optional, required for cluster="time"
    cluster="entity",    # "entity" | "time" | "none"
)

# 2) Formula + tabular data
fit = nextstat.econometrics.panel_fe_from_formula(
    "y ~ 1 + x1 + x2",
    data,                # dict-of-columns / list-of-dicts / pandas.DataFrame (if installed)
    entity="entity_id",
    time="t",            # optional
    cluster="entity",
)

print(fit.column_names)
print(fit.coef)
print(fit.standard_errors)
```

## Standard errors

This baseline supports **1-way cluster-robust** standard errors:
- `cluster="entity"`: cluster by entity (default)
- `cluster="time"`: cluster by time (requires `time=...`)

Two-way clustering is not implemented yet.

# Difference-in-Differences (DiD) and event study (TWFE)

NextStat also provides minimal DiD and event-study helpers using a **two-way fixed effects (TWFE)** baseline:

- DiD regressor: `treat_i * post_t`
- Event study regressors: `treat_i * 1[rel_time == k]` for a window of relative times, with a reference bin omitted.

## Parallel trends

All DiD-style estimators rely on a parallel-trends assumption. Use event-study plots as a baseline diagnostic:
- coefficients for negative relative times (pre-periods) should be near 0
- large pre-trends are a red flag for identification

## API

```python
import nextstat

# DiD via TWFE
did = nextstat.econometrics.did_twfe_from_formula(
    "y ~ 1 + x1",  # optional controls
    data,
    entity="entity_id",
    time="t",
    treat="treated",
    post="post",
    cluster="entity",
)
print(did.att, did.att_se)

# Event study via TWFE (relative time = t - policy_time)
es = nextstat.econometrics.event_study_twfe_from_formula(
    "y ~ 1 + x1",
    data,
    entity="entity_id",
    time="t",
    treat="treated",
    event_time=10,       # scalar policy time, or a column name for per-row event time
    window=(-4, 4),
    reference=-1,
    cluster="entity",
)
print(es.rel_times)
print(es.coef)
print(es.standard_errors)
```

# Instrumental variables (IV) / 2SLS baseline

NextStat includes a minimal **instrumental variables** estimator using **two-stage least squares (2SLS)** for linear models.

- Structural equation: `y = d * beta + x' gamma + u` where `d` is endogenous.
- Instruments: `z` affects `d`, but is excluded from the structural equation.

## API

```python
import nextstat

fit = nextstat.econometrics.iv_2sls_from_formula(
    "y ~ 1 + x",      # exogenous regressors (and intercept)
    data,
    endog="d",        # endogenous regressor(s)
    instruments=["z"],  # excluded instruments
    cov="hc1",        # "homoskedastic" | "hc1" | "cluster"
)

print(fit.column_names)
print(fit.coef)
print(fit.standard_errors)
print(fit.diagnostics.excluded_instruments)
print(fit.diagnostics.first_stage_f)
print(fit.diagnostics.first_stage_partial_r2)
```
