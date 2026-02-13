# Phase 6: Regression (Linear / Logistic / Poisson / NegBin)

This tutorial covers the **non-HEP** Phase 6 regression surfaces in NextStat:

- Linear regression (OLS / ridge)
- Logistic regression (Bernoulli-logit)
- Poisson regression (log link, optional offset/exposure)
- Negative binomial regression (NB2; baseline overdispersion model)

All examples are **dependency-light** (no NumPy / pandas required). Where formula support is used, NextStat accepts:

- `dict` of columns
- list of dict rows
- `pandas.DataFrame` (only if pandas is installed)

See also:
- Phase 7 (hierarchical / multilevel): `docs/tutorials/phase-7-hierarchical.md`
- Phase 8 (time series / Kalman filter): `docs/tutorials/phase-8-timeseries.md`

## Assumptions and diagnostics (quick checklist)

Linear regression (OLS):
- Linearity and (approx.) homoskedastic errors are common baseline assumptions.
- Check residuals; consider robust SE (`nextstat.robust.ols_hc_from_fit`) if variance is not constant.

Logistic regression:
- Watch for (near) perfect separation (coefficients can diverge); use `l2=1.0` if needed.
- Calibrate: probabilities can be well-ranked but miscalibrated on small datasets.

Poisson / NegBin:
- Poisson implies Var(y) ≈ E(y); if data are overdispersed, negative binomial is a common baseline.
- Use deviance-style diagnostics (`nextstat.glm.metrics.mean_poisson_deviance`) to compare fits.

## Linear regression (OLS) quickstart

```python
import nextstat

# Synthetic dataset (deterministic, no RNG).
X = [[float(i)] for i in range(1, 21)]
y = [1.0 + 2.0 * x[0] + 0.1 * ((i % 5) - 2) for i, x in enumerate(X)]

fit = nextstat.glm.linear.fit(X, y, include_intercept=True)
print(fit.coef)  # [intercept, slope]

summary = nextstat.summary.fit_summary(fit, names=["Intercept", "x"])
print(nextstat.summary.summary_to_str(summary))
```

### From a minimal formula

```python
import nextstat

data = [{"y": yi, "x": xi[0]} for xi, yi in zip(X, y)]
fit, names = nextstat.glm.linear.from_formula("y ~ 1 + x", data)

print(nextstat.summary.summary_to_str(nextstat.summary.fit_summary(fit, names=names)))
```

## Logistic regression (binary outcome)

```python
import nextstat

X = [[-2.0], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0]]
y = [0, 0, 0, 0, 1, 1, 1]

fit = nextstat.glm.logistic.fit(X, y, include_intercept=True, l2=1.0)
print("converged:", fit.converged)
print("warnings:", fit.warnings)
print("coef:", fit.coef)

summary = nextstat.summary.fit_summary(fit, names=["Intercept", "x"])
print(nextstat.summary.summary_to_str(summary))
```

Notes:
- For some datasets, the unregularized MLE can be unstable due to (near) perfect separation.
  Use `l2=1.0` (ridge/MAP) to stabilize.

### Logistic regression from a formula

```python
import nextstat

data = [{"y": yi, "x": xi[0]} for xi, yi in zip(X, y)]
fit, names = nextstat.glm.logistic.from_formula("y ~ 1 + x", data, l2=1.0)
print(nextstat.summary.summary_to_str(nextstat.summary.fit_summary(fit, names=names)))
```

## Poisson regression (counts) with exposure

This example models counts where different rows have different observation times (exposure).

```python
import nextstat

X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
exposure = [1.0, 1.0, 2.0, 2.0, 4.0]  # e.g. time-at-risk
y = [1, 2, 2, 5, 11]

fit = nextstat.glm.poisson.fit(X, y, include_intercept=True, exposure=exposure)
print(fit.coef)

summary = nextstat.summary.fit_summary(fit, names=["Intercept", "x"])
print(nextstat.summary.summary_to_str(summary))
```

### Poisson regression from a formula (exposure column)

```python
import nextstat

data = [{"y": yi, "x": xi[0], "exposure": ei} for (xi, yi, ei) in zip(X, y, exposure)]
fit, names = nextstat.glm.poisson.from_formula("y ~ 1 + x", data, exposure="exposure")
print(nextstat.summary.summary_to_str(nextstat.summary.fit_summary(fit, names=names)))
```

## Negative binomial regression (baseline overdispersion)

If your count data are more variable than the Poisson model allows, negative binomial is a common baseline.

```python
import nextstat

X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
y = [1, 2, 1, 6, 12]

fit = nextstat.glm.negbin.fit(X, y, include_intercept=True)
print("alpha (dispersion):", fit.alpha)

summary = nextstat.summary.fit_summary(fit, names=["Intercept", "x"])
print(nextstat.summary.summary_to_str(summary))
```

## Categorical variables

NextStat’s formula builder supports one-hot encoding for categoricals via `categorical=["device"]`.

```python
import nextstat

data = [
    {"y": 0.10, "x": 0.0, "device": "mobile"},
    {"y": 0.12, "x": 1.0, "device": "mobile"},
    {"y": 0.20, "x": 0.0, "device": "desktop"},
    {"y": 0.22, "x": 1.0, "device": "desktop"},
]

fit, names = nextstat.glm.linear.from_formula(
    "y ~ 1 + x + device",
    data,
    categorical=["device"],
)
print(names)
print(nextstat.summary.summary_to_str(nextstat.summary.fit_summary(fit, names=names)))
```

## Robust standard errors (HC / sandwich / cluster)

NextStat provides baseline robust covariance estimators in `nextstat.robust` (no NumPy required).

### OLS: HC0–HC3

```python
import nextstat

X = [[float(i)] for i in range(1, 21)]
y = [1.0 + 2.0 * x[0] + 0.1 * ((i % 5) - 2) for i, x in enumerate(X)]

fit = nextstat.glm.linear.fit(X, y, include_intercept=True)
cov, se = nextstat.robust.ols_hc_from_fit(fit, X, y, kind="HC3")

summary = nextstat.summary.wald_summary(fit.coef, se, names=["Intercept", "x"])
print(nextstat.summary.summary_to_str(summary))
```

### Logistic / Poisson: baseline sandwich SE

```python
import nextstat

X = [[-2.0], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0]]
y = [0, 0, 0, 0, 1, 1, 1]
fit = nextstat.glm.logistic.fit(X, y, include_intercept=True, l2=1.0)

cov, se = nextstat.robust.logistic_sandwich_from_fit(fit, X, y)
summary = nextstat.summary.wald_summary(fit.coef, se, names=["Intercept", "x"])
print(nextstat.summary.summary_to_str(summary))
```

Cluster-robust variants are available for OLS (`nextstat.robust.ols_cluster_from_fit`) and for the GLM sandwich estimators
(`cluster=cluster_ids` in the `*_sandwich_from_fit` helpers).

## Metrics and cross-validation

Phase 6 includes dependency-free metrics and minimal k-fold CV helpers:

```python
import nextstat

X = [[float(i)] for i in range(1, 31)]
y = [1.0 + 2.0 * x[0] + 0.2 * ((i % 7) - 3) for i, x in enumerate(X)]

fit = nextstat.glm.linear.fit(X, y, include_intercept=True)
rmse = nextstat.glm.rmse(y, fit.predict(X))
print("rmse:", rmse)

cv = nextstat.glm.cross_val_score("linear", X, y, k=5, seed=0, fit_kwargs={"include_intercept": True})
print("cv mean:", cv.mean, "stdev:", cv.stdev)
```

## scikit-learn style adapters

If you prefer scikit-learn style estimators, use `nextstat.sklearn`. These adapters still work without
scikit-learn installed (as plain Python classes).

```python
from nextstat.sklearn import NextStatLinearRegression

X = [[0.0], [1.0], [2.0], [3.0]]
y = [0.1, 1.0, 2.2, 3.0]

est = NextStatLinearRegression(include_intercept=True)
est.fit(X, y)
print(est.intercept_, est.coef_)
print(est.predict([[4.0]]))
```
