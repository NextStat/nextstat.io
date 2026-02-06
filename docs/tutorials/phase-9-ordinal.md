---
title: "Phase 9: Ordinal Outcomes (Ordered Logit) — Tutorial"
status: draft
---

# Phase 9: Ordinal Outcomes (Ordered Logit)

This tutorial documents the baseline support for **ordinal outcomes** via
ordered logistic regression (proportional odds model).

## What is implemented

`nextstat.ordinal.ordered_logit.fit(...)`:

- Ordered logistic regression (K levels, y in `{0..K-1}`)
- MLE fit through the standard `nextstat.fit(...)` surface
- Prediction: per-level probabilities via `predict_proba(...)`

## Prerequisites

From the repo root:

```bash
cargo test -p ns-inference
./.venv/bin/maturin develop --release -m bindings/ns-py/Cargo.toml
```

Optional parity check against statsmodels:

```bash
./.venv/bin/pip install "statsmodels>=0.14" "numpy>=2.0"
```

## Quick start (Python)

```python
import math
import random

import nextstat


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def randn(rng: random.Random) -> float:
    # Box-Muller (deterministic).
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def sample_ordered_logit_y(rng: random.Random, *, eta: float, cuts: list[float]) -> int:
    # P(y <= k) = sigmoid(c_{k+1} - eta)
    cdf = [sigmoid(cuts[0] - eta)]
    for c in cuts[1:]:
        cdf.append(sigmoid(c - eta))
    probs = [cdf[0]]
    for j in range(1, len(cdf)):
        probs.append(cdf[j] - cdf[j - 1])
    probs.append(1.0 - cdf[-1])

    u = rng.random()
    acc = 0.0
    for k, p in enumerate(probs):
        acc += float(p)
        if u <= acc:
            return int(k)
    return int(len(probs) - 1)


# Synthetic 3-level outcome with 1 feature (no intercept column).
rng = random.Random(0)
beta_true = 1.25
cuts_true = [-0.5, 0.5]  # K=3 levels => 2 cutpoints

n = 600
x = []
y = []
for _ in range(n):
    xi = randn(rng)
    eta = beta_true * xi
    yi = sample_ordered_logit_y(rng, eta=eta, cuts=cuts_true)
    x.append([xi])
    y.append(yi)

fit = nextstat.ordinal.ordered_logit.fit(x, y, n_levels=3)
print("converged:", fit.converged, "nll:", fit.nll)
print("beta_hat:", fit.coef)
print("cutpoints_hat:", fit.cutpoints)

grid = [[-2.0], [0.0], [2.0]]
probs = fit.predict_proba(grid)
print("predict_proba:", probs)
```

## Notes and limitations

- Do not add an explicit intercept column in `X`. The model already has cutpoints;
  an intercept makes the parameterization non-identifiable.
- This is a baseline MLE implementation (no priors/MAP yet).

## Optional parity (statsmodels)

This is not required for normal NextStat usage, but it is a useful reference check.

```python
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

X = np.asarray(x, dtype=float)
Y = np.asarray(y, dtype=int)

sm = OrderedModel(Y, X, distr="logit")
sm_res = sm.fit(method="bfgs", disp=False, maxiter=200)

sm_probs = sm.model.predict(sm_res.params, exog=np.asarray(grid, dtype=float))
ns_probs = np.asarray(fit.predict_proba(grid), dtype=float)

print("max|Δprob|:", float(np.max(np.abs(sm_probs - ns_probs))))
```

