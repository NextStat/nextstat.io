---
title: "Phase 12: Instrumental Variables (IV) / 2SLS — Tutorial"
status: draft
---

# IV / 2SLS baseline

NextStat includes a minimal **instrumental variables** estimator for linear models using
**two-stage least squares (2SLS)**:

- Structural: `y = d * beta + x' gamma + u` where `d` is endogenous.
- Instruments: `z` affects `d` but is excluded from the structural equation.

The baseline includes classic weak-IV diagnostics:
- first-stage partial `F`
- first-stage partial `R^2`

## Quick start

```python
import math
import random

import nextstat


def randn(rng: random.Random) -> float:
    return rng.gauss(0.0, 1.0)


def make_data(n: int = 400, seed: int = 0):
    rng = random.Random(seed)
    y, d, x, z = [], [], [], []
    for _ in range(n):
        xi = randn(rng)
        zi = randn(rng)
        vi = randn(rng)
        ui = 0.6 * vi + math.sqrt(1.0 - 0.6 * 0.6) * randn(rng)
        di = 1.0 * zi + 0.5 * xi + vi
        yi = 2.0 * di + 1.0 * xi + ui
        x.append(xi)
        z.append(zi)
        d.append(di)
        y.append(yi)
    return {"y": y, "d": d, "x": x, "z": z}


data = make_data()
fit = nextstat.econometrics.iv_2sls_from_formula(
    "y ~ 1 + x",
    data,
    endog="d",
    instruments=["z"],
    cov="hc1",
)

print(fit.column_names)
print(fit.coef)
print(fit.standard_errors)
print("excluded instruments kept:", fit.diagnostics.excluded_instruments)
print("first-stage F:", fit.diagnostics.first_stage_f)
print("partial R2:", fit.diagnostics.first_stage_partial_r2)
```

Example output (approx):

```text
['d', 'Intercept', 'x']
coef[0] (d): 1.9609006818580528
se[0] (d): 0.05105704945607526
first-stage F: [378.58893731003866]
partial R2: [0.4881309145835575]
```

## Limitations

- Diagnostics are **classic** (non-robust) first-stage metrics; they are meant as a baseline.
- Only a single-equation 2SLS baseline is implemented (no GMM, LIML, etc.).
- Only 1-way cluster-robust covariance is implemented.
