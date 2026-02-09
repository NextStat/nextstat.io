---
title: "Phase 12: Panel Fixed Effects (Within Estimator) — Tutorial"
status: draft
---

# Panel fixed effects (within estimator)

NextStat includes a minimal panel linear regression baseline with **entity fixed effects** (FE)
via the within estimator:

- Model: `y_it = alpha_i + x_it' beta + eps_it`
- Estimation: demean within each entity, then run OLS on transformed data
- SE: 1-way cluster-robust (`cluster="entity"` or `cluster="time"`)

## Quick start

```python
import nextstat

data = {
    "y": [10.0, 11.0, 12.0, -3.0, -2.0, -1.0],
    "x1": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    "entity": ["a", "a", "a", "b", "b", "b"],
    "t": [0, 1, 2, 0, 1, 2],
}

fit = nextstat.econometrics.panel_fe_from_formula(
    "y ~ 1 + x1",
    data,
    entity="entity",
    time="t",
    cluster="entity",
)

print(fit.column_names)
print(fit.coef)
print(fit.standard_errors)
```

Example output (approx):

```text
['x1']
[1.00...]
[0.00...]
```

## Notes and limitations

- The intercept is absorbed by entity FE.
- Any regressor that is constant within an entity is absorbed (not identifiable).
- Only 1-way clustering is implemented.

