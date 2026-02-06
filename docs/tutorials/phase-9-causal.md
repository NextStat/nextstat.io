---
title: "Phase 9: Causal Helpers (Propensity Scores) — Tutorial"
status: draft
---

# Phase 9: Causal Helpers (Propensity Scores)

This tutorial documents a small set of **causal workflow helpers** focused on:

- propensity score estimation
- overlap diagnostics
- balance diagnostics (standardized mean differences)
- inverse-probability weighting (IPW)

These helpers are a convenience layer, not a claim of causal identification.

## What is implemented

`nextstat.causal.propensity.fit(...)`:

- Fits a propensity score model (logistic regression with a small ridge penalty by default).
- Returns clipped propensity scores in `(eps, 1-eps)` to avoid infinite weights.

`nextstat.causal.propensity.ipw_weights(...)`:

- Computes IPW weights for common estimands (`ate`, `att`, `atc`).

`nextstat.causal.propensity.diagnostics(...)`:

- Overlap checks (min/max propensity, group ranges, warnings).
- Balance checks via standardized mean differences (SMD), unweighted and optionally weighted.
- Effective sample size (ESS) under weights.

## Quick start (Python)

```python
import math
import random

import nextstat


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


rng = random.Random(0)
n = 1000
x = []
t = []

for _ in range(n):
    x1 = rng.gauss(0.0, 1.0)
    x2 = rng.gauss(0.0, 1.0)
    # Treatment depends on covariates (confounding in observational settings).
    p = sigmoid(1.2 * x1 - 0.7 * x2)
    t_i = 1 if rng.random() < p else 0
    x.append([x1, x2])
    t.append(t_i)

ps = nextstat.causal.propensity.fit(x, t, l2=1.0)
weights = nextstat.causal.propensity.ipw_weights(
    t,
    ps.propensity_scores,
    estimand="ate",
    stabilized=True,
    max_weight=50.0,
)

diag0 = nextstat.causal.propensity.diagnostics(x, t, ps.propensity_scores)
diagw = nextstat.causal.propensity.diagnostics(x, t, ps.propensity_scores, weights=weights)

print("propensity range:", diag0.propensity_min, diag0.propensity_max)
print("max |SMD| unweighted:", diag0.max_abs_smd_unweighted)
print("max |SMD| weighted:", diagw.max_abs_smd_weighted)
print("warnings:", diagw.warnings)
```

## Notes and limitations

- **No magic causality**: propensity scoring and weighting do not, by themselves, identify causal
  effects. Causal conclusions depend on study design and assumptions (e.g., unconfoundedness,
  positivity, correct specification).
- **Overlap matters**: extreme propensities imply unstable weights and poor identification. Prefer
  trimming, clipping, or redesigning the study when overlap is weak.
- **Modeling choices**: the propensity model is intentionally simple and dependency-light; it does
  not replace domain-specific modeling or sensitivity analyses.

