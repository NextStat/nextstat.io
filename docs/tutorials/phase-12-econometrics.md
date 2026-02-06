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

