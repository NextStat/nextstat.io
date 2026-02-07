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
- `nextstat.apply_patchset(workspace_json_str, patchset_json_str, patch_name=None) -> str`
- `nextstat.fit(model, *, data=None) -> FitResult`
- `nextstat.map_fit(posterior) -> FitResult`
- `nextstat.fit_batch(models_or_model, datasets=None) -> list[FitResult]`
- `nextstat.sample(model_or_posterior, ...) -> dict` (NUTS)
- `nextstat.hypotest(poi_test, model, *, data=None, return_tail_probs=False) -> float | (float, list[float])`
- `nextstat.hypotest_toys(poi_test, model, *, n_toys=1000, seed=42, expected_set=False, data=None, return_tail_probs=False, return_meta=False) -> float | tuple | dict`
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
- `dim()` (alias of `n_params()`)
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

High-level helpers (recommended for most users):

- `nextstat.survival.exponential.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.weibull.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.lognormal_aft.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.cox_ph.fit(times, events, x, ties="efron", robust=True, groups=None) -> CoxPhFit`
  - `robust=True` returns sandwich SE (`fit.robust_se`)
  - `groups=...` switches sandwich SE to cluster-robust (`fit.robust_kind == "cluster"`)
  - `fit.predict_survival(x_new, times=grid)` returns Cox survival curves using a baseline cumulative hazard.

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
- `nextstat.survival`: high-level survival helpers (parametric right-censoring + Cox PH fit helpers).
- `nextstat.econometrics`: robust SE, FE baseline, DiD/event-study, IV/2SLS, and reporting helpers.
- `nextstat.causal`: propensity + AIPW baselines and sensitivity hooks.

## Evaluation Modes (Parity / Fast)

NextStat supports two evaluation modes for NLL computation, controllable at runtime:

```python
import nextstat

# Default: maximum speed (naive summation, SIMD/Accelerate/CUDA, multi-threaded)
nextstat.set_eval_mode("fast")

# Parity: deterministic (Kahan summation, Accelerate disabled, single-thread recommended)
nextstat.set_eval_mode("parity")

# Query current mode
print(nextstat.get_eval_mode())  # "fast" or "parity"
```

| Mode | Summation | Backend | Use Case |
|------|-----------|---------|----------|
| `"fast"` | Naive | SIMD / Accelerate / CUDA | Production inference |
| `"parity"` | Kahan compensated | SIMD only | CI, pyhf parity validation |

**When to use parity mode:**
- Validating numerical results against pyhf NumPy backend
- CI regression tests requiring bit-exact reproducibility
- Debugging numerical discrepancies

**Tolerance contract** (Parity mode vs pyhf):
- Per-bin expected data: **1e-12** (bit-exact arithmetic)
- NLL value: **1e-10** absolute
- Gradient: **1e-6** atol + **1e-4** rtol (AD vs FD noise)
- Best-fit params: **2e-4** (optimizer surface)

See `docs/pyhf-parity-contract.md` for the full 7-tier tolerance hierarchy.

**Measured overhead:** <5% (Kahan vs naive at same thread count).

## Batch Toy Fitting (CPU)

```python
import nextstat

model = nextstat.from_pyhf(json_str)
params = model.suggested_init()

# CPU batch: Rayon parallel, one AD tape per thread
results = nextstat.fit_toys_batch(model, params, n_toys=1000, seed=42)

# Each result has: .parameters, .nll, .converged, .n_iter, .n_fev, .n_gev
converged = sum(1 for r in results if r.converged)
print(f"{converged}/{len(results)} toys converged")
```

- `nextstat.fit_toys_batch(model, params, *, n_toys=1000, seed=42) -> list[FitResult]`
- `nextstat.has_accelerate() -> bool` — Check if Apple Accelerate backend is active.

## GPU acceleration

- `nextstat.has_cuda() -> bool` — Check if CUDA GPU batch backend is available at runtime.
- `nextstat.fit_toys_batch_gpu(model, params, *, n_toys=1000, seed=42, device="cpu") -> list[FitResult]` — Batch toy fitting. When `device="cuda"`, uses GPU-accelerated lockstep L-BFGS-B with fused NLL+gradient kernel. Falls back to CPU (Rayon) when `device="cpu"`.

Build with CUDA support:

```bash
cd bindings/ns-py
maturin develop --release --features cuda
```

Example:

```python
import nextstat

model = nextstat.from_pyhf(json_str)
params = model.suggested_init()

if nextstat.has_cuda():
    results = nextstat.fit_toys_batch_gpu(model, params, n_toys=10000, device="cuda")
else:
    results = nextstat.fit_toys_batch_gpu(model, params, n_toys=10000, device="cpu")
```

## CLI parity

The CLI mirrors the core workflows for HEP (fit/hypotest/scan/limits) and time series.
See `docs/references/cli.md`.
