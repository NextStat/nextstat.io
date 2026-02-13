---
title: "Phase 12: Doubly-Robust AIPW (ATE/ATT) — Tutorial"
status: draft
---

# Doubly-robust AIPW (ATE/ATT)

NextStat includes a minimal baseline **augmented inverse probability weighting (AIPW)**
estimator for binary treatments:

- `estimand="ate"`: average treatment effect
- `estimand="att"`: average treatment effect on the treated

Nuisance models (baseline):
- propensity score via logistic regression (`nextstat.causal.propensity.fit`)
- outcome regression via linear regression fitted separately per arm

This is a convenience layer for applied workflows, not a claim of causal identification.

## Quick start

```python
import math
import random

import nextstat


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


rng = random.Random(0)
n = 800
tau = 2.0

x = []
y = []
t = []

for _ in range(n):
    x1 = rng.gauss(0.0, 1.0)
    p = sigmoid(-0.2 + 1.2 * x1)
    ti = 1 if rng.random() < p else 0
    ui = 0.2 * rng.gauss(0.0, 1.0)
    yi = tau * float(ti) + 0.7 * x1 + ui
    x.append([x1])
    t.append(ti)
    y.append(yi)

fit = nextstat.causal.aipw.aipw_fit(x, y, t, estimand="ate", trim_eps=1e-6)
print(fit.estimate, fit.standard_error)
print("warnings:", fit.propensity_diagnostics.warnings)
```

Here, `trim_eps` clips propensity scores into `(trim_eps, 1-trim_eps)` to avoid infinite weights.

Example output (approx):

```text
2.0001430275996452 0.0185778607636736
warnings: []
```

## Hooks (precomputed nuisances)

For sensitivity analysis or integration with external models, you can pass precomputed
`propensity_scores`, `mu0`, `mu1` to `aipw_fit(...)`.

## Sensitivity helpers

Baseline helpers:
- `nextstat.causal.aipw.e_value_rr(rr)` (risk-ratio scale)
- `nextstat.causal.aipw.rosenbaum_bounds(y_treated, y_control, *, gammas=None) -> RosenbaumBoundsResult` (Rosenbaum-style bounds for matched designs; returns `gammas`, `p_upper`, `p_lower`, `gamma_critical`)

Example (matched pairs):

```python
import nextstat

y_treated = [10.0, 12.0, 15.0, 11.0, 13.0]
y_control = [5.0, 6.0, 7.0, 5.5, 6.5]

rb = nextstat.causal.aipw.rosenbaum_bounds(y_treated, y_control, gammas=[1.0, 1.5, 2.0, 3.0, 5.0])
print("gamma_critical:", rb.gamma_critical)
```

## Limitations

- AIPW relies on overlap/positivity and correct nuisance modeling (or at least one nuisance correct).
- This baseline uses simple logistic/linear models; it does not implement cross-fitting.
