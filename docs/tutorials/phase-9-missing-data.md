---
title: "Phase 9: Missing Data Policy (Baseline)"
status: draft
---

# Phase 9: Missing Data Policy (Baseline)

This tutorial defines the baseline missing-data policies in NextStat.

The design goal is simple: missing-data handling must be explicit and reproducible.

## What is implemented

`nextstat.missing.apply_policy(...)`:

- `policy="drop_rows"`: drop any row with missing in `X` or `y`
- `policy="impute_mean"`: mean-impute missing values in `X` (per-column), drop rows with missing `y`

Missing values are `None` or `float("nan")`.

## What is not implemented (yet)

- Model-based missingness (MAR) with latent imputation inside inference
- Measurement error models (errors-in-variables)

Until those are implemented, prefer:
- `drop_rows` if missing is rare or plausibly MCAR
- `impute_mean` only as a baseline and only with clear caveats in reporting

## Quick start (Python)

```python
import math
import nextstat.missing as missing

X = [
  [1.0, 2.0],
  [None, 3.0],
  [4.0, float("nan")],
]
y = [0, 1, 0]

r0 = missing.apply_policy(X, y, policy="drop_rows")
print("kept:", r0.n_kept, "dropped:", r0.n_dropped, "X:", r0.x, "y:", r0.y)

r1 = missing.apply_policy(X, y, policy="impute_mean")
print("kept:", r1.n_kept, "dropped:", r1.n_dropped, "X:", r1.x, "y:", r1.y)
```

## Notes

- `impute_mean` will raise if a column is entirely missing.
- Do not silently impute in production reporting without documenting the policy and rationale.
