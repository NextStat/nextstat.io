---
title: "Python API Reference (nextstat)"
status: stable
---

# Python API Reference (nextstat)

This page documents the public Python surface exported by `nextstat`.

Notes:
- The compiled extension is `nextstat._core` (PyO3/maturin).
- Convenience wrappers and optional modules live under `nextstat.*`.

## Top-level functions

- `nextstat.from_pyhf(json_str) -> HistFactoryModel`
- `nextstat.fit(model, *, data=None) -> FitResult`
- `nextstat.map_fit(posterior) -> FitResult`
- `nextstat.fit_batch(models_or_model, datasets=None) -> list[FitResult]`
- `nextstat.sample(model_or_posterior, ...) -> dict` (NUTS)
- `nextstat.hypotest(model, *, mu, expected_set=False, data=None) -> dict`
- `nextstat.profile_scan(model, *, start=0.0, stop=5.0, points=21, data=None) -> dict`
- `nextstat.upper_limit(model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, expected=False, data=None) -> float | (float, list[float])`
- `nextstat.upper_limits(model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, data=None) -> (float, list[float])`
- `nextstat.upper_limits_root(model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, data=None) -> (float, list[float])`

The full typed surface (including overloads) is in:
- `bindings/ns-py/python/nextstat/_core.pyi`

## Core classes

### `FitResult`

Fields:
- `parameters`: best-fit parameters (NextStat order)
- `uncertainties`: 1-sigma uncertainties (diagonal or covariance-derived baseline)
- `nll`: NLL at the optimum
- `converged`, `n_iter`, `n_fev`, `n_gev`

Compatibility aliases:
- `bestfit` (same as `parameters`)
- `twice_nll = 2 * nll`
- `success` (same as `converged`)
- `n_evaluations` (back-compat)

### `MaximumLikelihoodEstimator`

The object-oriented MLE surface:

```python
import nextstat

mle = nextstat.MaximumLikelihoodEstimator(max_iter=1000, tol=1e-6, m=10)
res = mle.fit(model)
```

Also supports `fit_batch(...)` for homogeneous lists of models.

### `Posterior`

Wraps a model and exposes constrained/unconstrained log density for sampling and MAP:

```python
import nextstat

post = nextstat.Posterior(model)
post.set_prior_normal("mu", center=0.0, width=5.0)
res = nextstat.map_fit(post)
```

## Models

All models implement a shared minimal contract:
- `n_params()`
- `nll(params)`
- `grad_nll(params)`
- `parameter_names()`
- `suggested_init()`
- `suggested_bounds()`

### HEP (HistFactory / pyhf JSON)

- `HistFactoryModel`: build from pyhf JSON (via `nextstat.from_pyhf` or `HistFactoryModel.from_workspace`).

### Regression / GLM

- `LinearRegressionModel`
- `LogisticRegressionModel`
- `PoissonRegressionModel`
- `NegativeBinomialRegressionModel`
- `ComposedGlmModel`
- `ols_fit(...)` (closed-form OLS helper)

### Ordinal regression

- `OrderedLogitModel`
- `OrderedProbitModel`

### Hierarchical / mixed models

- `LmmMarginalModel` (Gaussian mixed model, marginal likelihood baseline)

### Survival analysis

- `ExponentialSurvivalModel`
- `WeibullSurvivalModel`
- `LogNormalAftModel`
- `CoxPhModel`

### Time series / state space (Phase 8)

Low-level:
- `KalmanModel`
- `nextstat.kalman_filter(...)`, `nextstat.kalman_smooth(...)`, `nextstat.kalman_em(...)`, `nextstat.kalman_forecast(...)`, `nextstat.kalman_simulate(...)`

High-level wrappers:
- `nextstat.timeseries.*` convenience helpers and plotting artifacts.

### Pharmacometrics (Phase 13)

- `OneCompartmentOralPkModel` (oral dosing, first-order absorption)
  - Adds `predict(params) -> list[float]` for predicted concentrations at the model times.
- `OneCompartmentOralPkNlmeModel` (population + per-subject log-normal random effects; diagonal Omega baseline)

## Optional modules

These are imported from `nextstat/__init__.py` as convenience wrappers. Some require optional dependencies.

- `nextstat.viz`: plot-friendly artifacts for CLs curves and profile scans.
- `nextstat.bayes`: Bayesian helpers (ArviZ integration).
- `nextstat.timeseries`: higher-level time series helpers and plotting.
- `nextstat.econometrics`: robust SE, FE baseline, DiD/event-study, IV/2SLS, and reporting helpers.
- `nextstat.causal`: propensity + AIPW baselines and sensitivity hooks.

## CLI parity

The CLI mirrors the core workflows for HEP (fit/hypotest/scan/limits) and time series.
See `docs/references/cli.md`.

