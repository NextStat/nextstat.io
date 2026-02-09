---
title: "Python API Reference (nextstat)"
status: stable
---

# Python API Reference (nextstat)

This page documents the public Python surface exported by `nextstat`.

Notes:
- The compiled extension is `nextstat._core` (PyO3/maturin).
- Convenience wrappers and optional modules live under `nextstat.*`.
- The full typed surface (including overloads) is in `bindings/ns-py/python/nextstat/_core.pyi`.
- Installation, optional extras, and wheel build notes: `docs/references/python-packaging.md`.

## Top-level functions

### Model construction

- `nextstat.from_pyhf(json_str) -> HistFactoryModel` — create model from pyhf JSON workspace.
- `nextstat.from_histfactory_xml(xml_path) -> HistFactoryModel` — create model from HistFactory XML.
- `nextstat.workspace_audit(json_str) -> dict` — audit pyhf workspace for compatibility (counts channels, samples, modifiers; flags unsupported features).
- `nextstat.apply_patchset(workspace_json_str, patchset_json_str, patch_name=None) -> str` — apply a pyhf patchset.
- `nextstat.read_root_histogram(root_path, hist_path) -> dict` — read a TH1 histogram from a ROOT file. Returns `{name, title, bin_edges, bin_content, sumw2, underflow, overflow}`.
- `nextstat.histfactory_bin_edges_by_channel(xml_path) -> dict[str, list[float]]` — extract bin edges per channel from HistFactory XML.

Notes on HistFactory XML ingest (`from_histfactory` and `nextstat import histfactory`):
- `ShapeSys` histograms are treated as **relative** per-bin uncertainties and converted to absolute `sigma_abs = rel * nominal`.
- `StatError` histograms are treated as **relative** per-bin uncertainties and converted to absolute `sigma_abs = rel * nominal`.
- `StatError` follows channel `<StatErrorConfig ConstraintType=...>`:
  - `ConstraintType="Poisson"` => preserves `staterror` (per-channel, name `staterror_<channel>`) and attaches per-bin `Gamma` constraint
    metadata (non-standard extension) to `measurement.config.parameters` entries named `staterror_<channel>[i]`.
  - `ConstraintType="Gaussian"` => preserves `staterror` (per-channel, name `staterror_<channel>`) with Gaussian penalty (pyhf-style).
  - ROOT/HistFactory defaults when `<StatErrorConfig>` is omitted: `ConstraintType="Poisson"` and `RelErrorThreshold=0.05` (bins with
    relative stat error below threshold are pruned, i.e. the corresponding `staterror_<channel>[i]` is fixed at 1.0).
- Samples with `NormalizeByTheory="True"` receive a `lumi` modifier named `Lumi`, and `LumiRelErr` is surfaced via
  measurement parameter config (`auxdata=[1]`, `sigmas=[LumiRelErr]`).
- `NormFactor Val/Low/High` is surfaced via measurement parameter config (`inits` and `bounds`).

#### HS3 (HEP Statistics Serialization Standard)

- `HistFactoryModel.from_workspace(json_str) -> HistFactoryModel` — **auto-detects** pyhf vs HS3 format. If the JSON contains `"distributions"` + `"hs3_version"`, it is parsed as HS3; otherwise as pyhf.
- `HistFactoryModel.from_hs3(json_str, *, analysis=None, param_points=None) -> HistFactoryModel` — explicit HS3 loading with optional analysis selection and parameter point set.

```python
import json, nextstat

# Auto-detect: works with both pyhf and HS3
json_str = open("workspace-postFit_PTV.json").read()
model = nextstat.HistFactoryModel.from_workspace(json_str)

# Explicit HS3 with analysis selection
model = nextstat.HistFactoryModel.from_hs3(
    json_str,
    analysis="combPdf_obsData",        # default: first analysis
    param_points="default_values",     # default: "default_values"
)

result = nextstat.fit(model)
```

HS3 v0.2 support covers all modifier types produced by ROOT 6.37+: `normfactor`, `normsys`, `histosys`, `staterror`, `shapesys`, `shapefactor`, `lumi`. Unknown modifier/distribution types are silently skipped (forward-compatible).

### Fitting

- `nextstat.fit(model, *, data=None, init_pars=None) -> FitResult` — maximum likelihood estimation.
- `nextstat.map_fit(posterior) -> FitResult` — MAP estimation for Bayesian posteriors.
- `nextstat.fit_batch(models_or_model, datasets=None) -> list[FitResult]` — batch fitting (homogeneous model lists or single model + multiple datasets).

### Hypothesis testing

- `nextstat.hypotest(poi_test, model, *, data=None, return_tail_probs=False) -> float | (float, list[float])` — asymptotic CLs.
- `nextstat.hypotest_toys(poi_test, model, *, n_toys=1000, seed=42, expected_set=False, data=None, return_tail_probs=False, return_meta=False) -> float | tuple | dict` — toy-based CLs.

### Profile likelihood

- `nextstat.profile_scan(model, mu_values, *, data=None) -> dict` — profile likelihood scan.
- `nextstat.upper_limit(model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, expected=False, data=None) -> float | (float, list[float])` — upper limit via bisection.
- `nextstat.upper_limits(model, scan, *, alpha=0.05, data=None) -> (float, list[float])` — observed + expected limits from scan.
- `nextstat.upper_limits_root(model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, data=None) -> (float, list[float])` — ROOT-style limits.

### Sampling

- `nextstat.sample(model_or_posterior, *, n_chains=4, n_warmup=1000, n_samples=1000, seed=42, ...) -> dict` — NUTS (No-U-Turn Sampler).

### Toy data

- `nextstat.asimov_data(model, params) -> list[float]` — Asimov dataset (expected counts).
- `nextstat.poisson_toys(model, params, *, n_toys=1000, seed=42) -> list[list[float]]` — Poisson fluctuated toy datasets.
- `nextstat.fit_toys(model, params, *, n_toys=1000, seed=42) -> list[FitResult]` — generate and fit toys sequentially.
- `nextstat.fit_toys_batch(model, params, *, n_toys=1000, seed=42) -> list[FitResult]` — CPU parallel batch toy fitting (Rayon, one AD tape per thread).
- `nextstat.fit_toys_batch_gpu(model, params, *, n_toys=1000, seed=42, device="cpu") -> list[FitResult]` — GPU-accelerated batch toy fitting (see GPU section below).

### Visualization artifacts

- `nextstat.cls_curve(model, scan, *, alpha=0.05, data=None) -> dict` — asymptotic CLs exclusion curve. Returns `{alpha, nsigma_order, obs_limit, exp_limits, scan, cls_obs, cls_exp}`.
- `nextstat.profile_curve(model, mu_values, *, data=None) -> dict` — profile likelihood curve. Returns `{poi_index, mu_hat, nll_hat, mu_values, q_mu_values, twice_delta_nll, points}`.

### Parameter ranking

- `nextstat.ranking(model) -> list[dict]` — nuisance parameter ranking (impact on POI).

### Utilities

- `nextstat.ols_fit(x, y, *, include_intercept=True) -> list[float]` — closed-form OLS.
- `nextstat.rk4_linear(a, y0, t0, t1, dt, *, max_steps=100000) -> dict` — RK4 ODE solver for linear systems.
- `nextstat.set_eval_mode(mode: str) -> None` — set evaluation mode (`"fast"` or `"parity"`).
- `nextstat.set_threads(threads: int) -> bool` — best-effort: configure the global Rayon thread pool size (returns `True` if applied).
- `nextstat.get_eval_mode() -> str` — query current evaluation mode.
- `nextstat.has_accelerate() -> bool` — check Apple Accelerate backend availability.
- `nextstat.has_cuda() -> bool` — check CUDA backend availability.
- `nextstat.has_metal() -> bool` — check Metal backend availability.

---

## Core classes

### `FitResult`

Fields:
- `parameters: list[float]` — best-fit parameters (NextStat order)
- `uncertainties: list[float]` — 1-sigma uncertainties (diagonal or covariance-derived)
- `nll: float` — NLL at the optimum
- `converged: bool` — optimizer convergence flag
- `n_iter: int` — number of optimizer iterations
- `n_fev: int` — number of function evaluations
- `n_gev: int` — number of gradient evaluations
- `termination_reason: str` — optimizer termination reason (e.g. `"gradient_tolerance"`, `"max_iterations"`)
- `final_grad_norm: float` — L-infinity norm of the gradient at minimum
- `initial_nll: float` — NLL at the starting point
- `n_active_bounds: int` — number of parameters at their box constraint boundary

Compatibility aliases:
- `bestfit` (same as `parameters`)
- `twice_nll = 2 * nll`
- `success` (same as `converged`)
- `n_evaluations` (back-compat alias for `n_iter`)

### `FitMinimumResult`

Fast-path optimizer result (no covariance/Hessian). Returned by `MaximumLikelihoodEstimator.fit_minimum(...)`.

Fields:
- `parameters: list[float]`
- `nll: float`
- `converged: bool`
- `n_iter: int`
- `n_fev: int`, `n_gev: int`
- `message: str`
- `initial_nll: float`
- `final_gradient: list[float] | None`

Compatibility aliases:
- `bestfit` (same as `parameters`)
- `twice_nll = 2 * nll`
- `success` (same as `converged`)

### `MaximumLikelihoodEstimator`

The object-oriented MLE surface:

```python
import nextstat

mle = nextstat.MaximumLikelihoodEstimator(max_iter=1000, tol=1e-6, m=10)
res = mle.fit(model)
```

Also supports:
- `fit_batch(...)` for homogeneous lists of models.
- `fit_minimum(model, *, data=None, init_pars=None, bounds=None) -> FitMinimumResult` — fast-path NLL minimization intended for profile scans and conditional fits.
  - `bounds=` is currently supported for `HistFactoryModel` only; clamp a parameter to `(value, value)` to fix it.
- `fit_toys(model, params, *, n_toys=1000, seed=42) -> list[FitResult]`
- `ranking(model) -> list[dict]` — nuisance parameter ranking.
- `q0_like_loss_and_grad_nominal(model, *, channel, sample, nominal) -> (float, list[float])` — discovery q0 and gradient w.r.t. one sample's nominal yields. For ML training loops.
- `qmu_like_loss_and_grad_nominal(model, *, mu_test, channel, sample, nominal) -> (float, list[float])` — exclusion qmu and gradient.

### `Posterior`

Wraps a model and exposes constrained/unconstrained log density for sampling and MAP:

```python
import nextstat

post = nextstat.Posterior(model)
post.set_prior_normal("mu", center=0.0, width=5.0)
res = nextstat.map_fit(post)
```

Methods:
- `dim()`, `parameter_names()`, `suggested_init()`, `suggested_bounds()`
- `set_prior_flat(name)`, `set_prior_normal(name, center, width)`, `clear_priors()`, `priors() -> dict`
- `logpdf(theta)`, `grad(theta)` — constrained space
- `to_unconstrained(theta)`, `to_constrained(z)` — bijective transforms
- `logpdf_unconstrained(z)`, `grad_unconstrained(z)` — unconstrained space (for NUTS)

---

## Models

All models implement a shared minimal contract:
- `n_params()` / `dim()`
- `nll(params)`, `grad_nll(params)`
- `parameter_names()`, `suggested_init()`, `suggested_bounds()`

### HEP (HistFactory / pyhf JSON)

- `HistFactoryModel`: build from pyhf JSON (via `nextstat.from_pyhf` or `HistFactoryModel.from_workspace`).
  - `expected_data(params, *, include_auxdata=True) -> list[float]`
  - `with_observed_main(observed_main) -> HistFactoryModel` — return model with replaced observed data.
  - `set_sample_nominal(*, channel, sample, nominal)` — override one sample's nominal yields in-place (for ML/RL).
  - `poi_index() -> int | None`
  - `observed_main_by_channel() -> list[dict]`
  - `expected_main_by_channel_sample(params) -> list[dict]`

### Regression / GLM

- `LinearRegressionModel(x, y, *, include_intercept=True)`
- `LogisticRegressionModel(x, y, *, include_intercept=True)`
- `PoissonRegressionModel(x, y, *, include_intercept=True, offset=None)`
- `NegativeBinomialRegressionModel(x, y, *, include_intercept=True, offset=None)`
- `ComposedGlmModel` — hierarchical GLMs via static constructors:
  - `.linear_regression(x, y, *, group_idx, n_groups, random_intercept_non_centered, random_slope_feature_idx, correlated_feature_idx, lkj_eta, ...)`
  - `.logistic_regression(x, y, *, ...)`
  - `.poisson_regression(x, y, *, ...)`

### Ordinal regression

- `OrderedLogitModel(x, y, *, n_levels)`
- `OrderedProbitModel(x, y, *, n_levels)`

### Hierarchical / mixed models

- `LmmMarginalModel(x, y, *, include_intercept, group_idx, n_groups, random_slope_feature_idx)` — Gaussian mixed model (marginal likelihood).

### Survival analysis

- `ExponentialSurvivalModel(times, events)`
- `WeibullSurvivalModel(times, events)`
- `LogNormalAftModel(times, events)`
- `CoxPhModel(times, events, x, *, ties="efron")`

High-level helpers (recommended for most users):

- `nextstat.survival.exponential.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.weibull.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.lognormal_aft.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.cox_ph.fit(times, events, x, ties="efron", robust=True, groups=None) -> CoxPhFit`
  - `robust=True` returns sandwich SE (`fit.robust_se`)
  - `groups=...` switches sandwich SE to cluster-robust (`fit.robust_kind == "cluster"`)
  - `fit.predict_survival(x_new, times=grid)` returns Cox survival curves.

### Time series / state space

Low-level:
- `KalmanModel(f, q, h, r, m0, p0)` — linear state-space model (matrices as lists of lists).
  - `n_state()`, `n_obs()`
- `nextstat.kalman_filter(model, ys) -> dict` — forward Kalman filter (supports missing obs as `None`).
- `nextstat.kalman_smooth(model, ys) -> dict` — RTS smoother.
- `nextstat.kalman_em(model, ys, *, max_iter, tol, estimate_q, estimate_r, estimate_f, estimate_h, min_diag) -> dict` — EM parameter estimation.
- `nextstat.kalman_forecast(model, ys, *, steps, alpha) -> dict` — multi-step forecast with intervals.
- `nextstat.kalman_simulate(model, *, t_max, seed, init, x0) -> dict` — simulate from state-space model.

High-level wrappers:
- `nextstat.timeseries.*` — convenience helpers and plotting artifacts.

### Pharmacometrics

- `OneCompartmentOralPkModel(times, y, *, dose, bioavailability=1.0, sigma=1.0, lloq=None, lloq_policy="ignore")` — oral dosing, first-order absorption.
  - `predict(params) -> list[float]` — predicted concentrations.
- `OneCompartmentOralPkNlmeModel(times, y, subject_idx, n_subjects, *, dose, ...)` — population PK (NLME with per-subject random effects).

### Test / utility models

- `GaussianMeanModel(y, sigma)` — simple Gaussian mean estimation.
- `FunnelModel()` — Neal's funnel (sampler stress test).
- `StdNormalModel(dim=2)` — standard normal (sampler validation).

---

## Optional modules

These modules live under `nextstat.*` as convenience helpers. Some require optional dependencies.

- `nextstat.viz` — plot-friendly artifacts for CLs curves and profile scans.
- `nextstat.bayes` — Bayesian helpers (ArviZ integration).
- `nextstat.torch` — PyTorch differentiable wrappers (see below).
- `nextstat.timeseries` — higher-level time series helpers and plotting.
- `nextstat.survival` — high-level survival helpers (parametric right-censoring + Cox PH).
- `nextstat.econometrics` — robust SE, FE baseline, DiD/event-study, IV/2SLS, and reporting.
- `nextstat.causal` — propensity + AIPW baselines and sensitivity hooks.
- `nextstat.gym` — Gymnasium/Gym environments for RL / design-of-experiments (requires `gymnasium` + `numpy`). See below.
- `nextstat.mlops` — fit metrics extraction for experiment loggers (W&B, MLflow, Neptune).
- `nextstat.interpret` — systematic-impact ranking as ML-style Feature Importance.
- `nextstat.glm` — regression/GLM convenience wrappers.
- `nextstat.ordinal` — ordinal regression convenience wrappers.
- `nextstat.formula` — Patsy-like formula interface.
- `nextstat.ppc` — posterior predictive checks.
- `nextstat.missing` — missing data helpers.

---

## Differentiable Layer (PyTorch integration)

The `nextstat.torch` module provides `torch.autograd.Function` wrappers for end-to-end differentiable HEP inference. Two backends:

### CPU path (no CUDA required)

- `nextstat.torch.NextStatQ0` — `torch.autograd.Function` for discovery q0. CPU profile fit with envelope-theorem gradient.
- `nextstat.torch.NextStatZ0` — same but returns signed significance Z0.
- `nextstat.torch.create_session(model, signal_sample_name) -> dict` — create a CPU session.
- `nextstat.torch.nll_loss(session, signal_tensor) -> torch.Tensor` — differentiable NLL.

### CUDA path (zero-copy, requires `cuda` feature build)

- `DifferentiableSession(model, signal_sample_name)` — GPU session for differentiable NLL.
  - `nll_grad_signal(params, signal_ptr, grad_signal_ptr) -> float` — compute NLL and write gradient directly into a PyTorch CUDA tensor via raw device pointers (zero-copy).
  - `signal_n_bins() -> int`, `n_params() -> int`, `parameter_init() -> list[float]`

- `ProfiledDifferentiableSession(model, signal_sample_name)` — GPU session for profiled test statistics.
  - `profiled_q0_and_grad(signal_ptr) -> (float, list[float])` — discovery q0 with envelope-theorem gradient. Runs two GPU L-BFGS-B fits internally.
  - `profiled_qmu_and_grad(mu_test, signal_ptr) -> (float, list[float])` — exclusion qmu.
  - `signal_n_bins() -> int`, `n_params() -> int`, `parameter_init() -> list[float]`

- `nextstat.torch.NextStatNLL` — `torch.autograd.Function` wrapping `DifferentiableSession` (CUDA zero-copy).
- `nextstat.torch.ProfiledQ0Loss` / `ProfiledQmuLoss` — `torch.autograd.Function` wrapping `ProfiledDifferentiableSession`.

Convenience constructors:
- `nextstat.torch.create_profiled_session(model, signal_sample_name) -> ProfiledDifferentiableSession`
- `nextstat.torch.profiled_q0_loss(signal, session) -> torch.Tensor`
- `nextstat.torch.profiled_z0_loss(signal, session) -> torch.Tensor`
- `nextstat.torch.profiled_qmu_loss(signal, session, mu_test) -> torch.Tensor`
- `nextstat.torch.profiled_zmu_loss(signal, session, mu_test) -> torch.Tensor`

### ML-friendly API

- `nextstat.torch.SignificanceLoss(model, signal_sample_name, *, device="auto", negate=True, eps=1e-12)` — class wrapping profiled −Z₀. Init once, call per-batch. Returns −Z₀ by default (for SGD minimization).
  - `loss_fn(signal_hist) -> torch.Tensor` — differentiable loss
  - `loss_fn.q0(signal_hist)` — raw q₀
  - `loss_fn.z0(signal_hist)` — raw Z₀
  - `loss_fn.n_bins`, `loss_fn.n_params` — model dimensions
- `nextstat.torch.SoftHistogram(bin_edges, bandwidth="auto", mode="kde")` — differentiable binning (Gaussian KDE or sigmoid). Converts continuous NN scores into soft histogram.
  - `soft_hist(scores, weights=None) -> torch.Tensor [n_bins]`
- `nextstat.torch.batch_profiled_q0_loss(signal_histograms, session) -> list[torch.Tensor]` — profiled q₀ for a batch of signal histograms `[batch, n_bins]`.
- `nextstat.torch.batch_profiled_qmu_loss(signal, session, mu_values) -> list[torch.Tensor]` — profiled qμ for multiple mu values.
- `nextstat.torch.signal_jacobian(signal, session) -> torch.Tensor` — ∂q₀/∂signal without autograd.
- `nextstat.torch.signal_jacobian_numpy(signal, session) -> np.ndarray` — same as above, NumPy output.
- `nextstat.torch.as_tensor(x) -> torch.Tensor` — DLPack/array-API bridge: accepts JAX, CuPy, Arrow, NumPy arrays.

Example (CUDA):

```python
import torch
import nextstat
from nextstat.torch import create_profiled_session, profiled_q0_loss

model = nextstat.from_pyhf(json_str)
session = create_profiled_session(model, "signal")

signal = torch.randn(n_bins, device="cuda", requires_grad=True)
loss = profiled_q0_loss(session, signal)
loss.backward()  # signal.grad now contains dq0/d(signal)
```

---

## Evaluation Modes (Parity / Fast)

NextStat supports two evaluation modes for NLL computation, controllable at runtime:

```python
import nextstat

# Default: maximum speed (naive summation, SIMD/Accelerate/CUDA, multi-threaded)
nextstat.set_eval_mode("fast")

# Parity: deterministic (Kahan summation, Accelerate disabled, single-thread recommended)
nextstat.set_eval_mode("parity")

# Optional (recommended for parity in CI): force single-thread execution.
# Note: Rayon global thread pool can only be configured once per process,
# so call this early (before any parallel NextStat calls).
nextstat.set_threads(1)

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

Full 7-tier tolerance hierarchy: `tests/python/_tolerances.py`.

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

## GPU acceleration

- `nextstat.has_cuda() -> bool` — check CUDA availability.
- `nextstat.has_metal() -> bool` — check Metal availability (Apple Silicon).
- `nextstat.fit_toys_batch_gpu(model, params, *, n_toys=1000, seed=42, device="cpu") -> list[FitResult]`:
  - `device="cuda"` — NVIDIA GPU, f64 precision, lockstep L-BFGS-B with fused NLL+gradient kernel.
  - `device="metal"` — Apple Silicon GPU, f32 precision, lockstep L-BFGS-B with Metal kernel. Tolerance relaxed to 1e-3.
  - `device="cpu"` (default) — falls back to CPU (Rayon parallel, f64).

Build with GPU support:

```bash
cd bindings/ns-py

# CUDA (NVIDIA)
maturin develop --release --features cuda

# Metal (Apple Silicon)
maturin develop --release --features metal

# Both
maturin develop --release --features "cuda,metal"
```

Example:

```python
import nextstat

model = nextstat.from_pyhf(json_str)
params = model.suggested_init()

if nextstat.has_cuda():
    results = nextstat.fit_toys_batch_gpu(model, params, n_toys=10000, device="cuda")
elif nextstat.has_metal():
    results = nextstat.fit_toys_batch_gpu(model, params, n_toys=10000, device="metal")
else:
    results = nextstat.fit_toys_batch(model, params, n_toys=10000)
```

## MLOps Integration (`nextstat.mlops`)

Lightweight helpers to extract NextStat metrics as plain dicts for experiment loggers.

- `nextstat.mlops.metrics_dict(fit_result, *, prefix="", include_time=True, extra=None) -> dict[str, float]` — flat dict from `FitResult`. Keys: `mu`, `nll`, `edm`, `n_calls`, `converged`, `time_ms`, `param/<name>`, `error/<name>`.
- `nextstat.mlops.significance_metrics(z0, q0=0.0, *, prefix="", step_time_ms=0.0) -> dict[str, float]` — per-step metrics for training loop logging.
- `nextstat.mlops.StepTimer` — lightweight wall-clock timer: `.start()`, `.stop() -> float` (ms).

```python
import nextstat
from nextstat.mlops import metrics_dict

result = nextstat.fit(model)
wandb.log(metrics_dict(result))           # W&B
mlflow.log_metrics(metrics_dict(result))   # MLflow
```

## Interpretability (`nextstat.interpret`)

Systematic-impact ranking translated into ML-style Feature Importance.

- `nextstat.interpret.rank_impact(model, *, gpu=False, sort_by="total", top_n=None, ascending=False) -> list[dict]` — sorted impact table. Each dict: `name`, `delta_mu_up`, `delta_mu_down`, `total_impact`, `pull`, `constraint`, `rank`.
- `nextstat.interpret.rank_impact_df(model, **kwargs) -> pd.DataFrame` — same as above, pandas output (requires `pandas`).
- `nextstat.interpret.plot_rank_impact(model, *, top_n=20, gpu=False, figsize=(8,6), title=..., ax=None) -> matplotlib.Figure` — horizontal bar chart (requires `matplotlib`).

```python
from nextstat.interpret import rank_impact, plot_rank_impact

table = rank_impact(model, top_n=10)
for row in table:
    print(f"{row['rank']:2d}. {row['name']:30s}  impact={row['total_impact']:.4f}")

fig = plot_rank_impact(model, top_n=15)
fig.savefig("ranking.png")
```

## Agentic Analysis (`nextstat.tools`)

LLM tool definitions for AI-driven statistical analysis. Compatible with OpenAI function calling, LangChain, and MCP (Model Context Protocol).

- `nextstat.tools.get_toolkit() -> list[dict]` — OpenAI-compatible tool definitions for 9 operations: `nextstat_fit`, `nextstat_hypotest`, `nextstat_hypotest_toys`, `nextstat_upper_limit`, `nextstat_ranking`, `nextstat_discovery_asymptotic`, `nextstat_scan`, `nextstat_workspace_audit`, `nextstat_read_root_histogram`.
- `nextstat.tools.execute_tool(name, arguments) -> dict` — execute a tool call by name. Returns a stable envelope: `{schema_version, ok, result, error, meta}`.
- `nextstat.tools.execute_tool_raw(name, arguments) -> dict` — execute a tool call by name, returning only the raw payload (no envelope).
- `nextstat.tools.get_langchain_tools() -> list[StructuredTool]` — LangChain adapter (requires `langchain-core`).
- `nextstat.tools.get_mcp_tools() -> list[dict]` — MCP tool definitions (`name`, `description`, `inputSchema`).
- `nextstat.tools.get_tool_names() -> list[str]` — list available tool names.
- `nextstat.tools.get_tool_schema(name) -> dict | None` — JSON Schema for a specific tool.

```python
import json, openai
from nextstat.tools import get_toolkit, execute_tool

tools = get_toolkit()
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Fit this workspace"}],
    tools=tools,
)

for call in response.choices[0].message.tool_calls:
    result = execute_tool(call.function.name, json.loads(call.function.arguments))
    if not result.get("ok"):
        raise RuntimeError(result.get("error"))
    payload = result["result"]
```

## Surrogate Distillation (`nextstat.distill`)

Generate training datasets for neural likelihood surrogates. NextStat serves as the ground-truth oracle; the surrogate provides nanosecond inference.

- `nextstat.distill.generate_dataset(model, n_samples=100_000, *, method="sobol", bounds=None, seed=42, include_gradient=True) -> SurrogateDataset` — sample parameter space and evaluate NLL + gradient at each point. Methods: `"sobol"`, `"lhs"`, `"uniform"`, `"gaussian"`.
- `nextstat.distill.SurrogateDataset` — container with `.parameters`, `.nll`, `.gradient` (NumPy arrays), `.parameter_names`, `.parameter_bounds`, `.metadata`.
- `nextstat.distill.to_torch_dataset(ds) -> TensorDataset` — convert for PyTorch DataLoader.
- `nextstat.distill.to_numpy(ds) -> dict[str, ndarray]` — export as dict of arrays.
- `nextstat.distill.to_npz(ds, path)` / `from_npz(path)` — save/load compressed `.npz`.
- `nextstat.distill.to_parquet(ds, path)` — export to Parquet (zstd, requires `pyarrow`).
- `nextstat.distill.train_mlp_surrogate(ds, *, hidden_layers=(256,256,128), epochs=100, lr=1e-3, grad_weight=0.1, device="cpu") -> nn.Module` — convenience MLP trainer with Sobolev loss.
- `nextstat.distill.predict_nll(surrogate, params_np) -> ndarray` — evaluate trained surrogate on raw parameters.

```python
from nextstat.distill import generate_dataset, train_mlp_surrogate, predict_nll

model = nextstat.from_pyhf(workspace_json)
ds = generate_dataset(model, n_samples=100_000, method="sobol")

surrogate = train_mlp_surrogate(ds, epochs=50, device="cuda")
pred = predict_nll(surrogate, np.array(model.parameter_init()))
```

## Gymnasium / RL Environment (`nextstat.gym`)

Gymnasium-compatible environment for optimizing a single sample's nominal yields via reinforcement learning. Requires `gymnasium` (or legacy `gym`) + `numpy`.

### `HistFactoryEnv`

```python
from nextstat.gym import HistFactoryEnv

env = HistFactoryEnv(
    workspace_json=json_str,
    channel="SR",
    sample="signal",
    reward_metric="z0",       # "nll", "q0", "z0", "qmu", "zmu"
    mu_test=5.0,              # for qmu/zmu metrics
    max_steps=128,
    action_scale=0.05,
    action_mode="logmul",     # "add" or "logmul"
    init_noise=0.0,
    clip_min=1e-12,
    clip_max=1e12,
)

obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

- `observation_space` / `action_space` — standard Gym `Box` spaces (shape = n_bins).
- `reset(seed=None, options=None)` — reset episode, returns `(observation, info)`.
- `step(action)` — apply action to signal yields, compute reward, return `(obs, reward, terminated, truncated, info)`.

Compatible with both Gymnasium (5-tuple) and legacy gym (4-tuple) APIs.

### `make_histfactory_env()`

Factory function:

```python
from nextstat.gym import make_histfactory_env

env = make_histfactory_env(
    workspace_json,
    channel="SR",
    sample="signal",
    reward_metric="z0",
)
```

## Arrow / Polars Integration

Zero-copy interchange between NextStat and the Arrow columnar ecosystem (PyArrow, Polars, DuckDB, Spark). Backed by Rust `arrow` 57.3 + `parquet` 57.3 crates; the Python ↔ Rust bridge uses Arrow IPC.

See also: `docs/references/arrow-parquet-io.md` (schema contract + manifest for reproducible pipelines).

### High-level API

- `nextstat.from_arrow(table, *, poi="mu", observations=None) -> HistFactoryModel` — create a model from a **PyArrow Table** or **RecordBatch**. The table must have columns `channel` (Utf8), `sample` (Utf8), `yields` (List\<Float64\>), optionally `stat_error` (List\<Float64\>). Works with any Arrow-compatible source (Polars, DuckDB, Spark).
- `nextstat.to_arrow(model, *, params=None, what="yields") -> pyarrow.Table` — export model data. `what="yields"` returns expected yields per channel; `what="params"` returns parameter metadata (name, index, value, bounds, init).
- `nextstat.from_parquet(path, *, poi="mu", observations=None) -> HistFactoryModel` — read a Parquet file directly (Zstd/Snappy), same schema as `from_arrow`.

### Low-level IPC API

- `nextstat.from_arrow_ipc(ipc_bytes, poi="mu", observations=None) -> HistFactoryModel` — ingest raw Arrow IPC stream bytes.
- `nextstat.to_arrow_yields_ipc(model, params=None) -> bytes` — export yields as IPC bytes.
- `nextstat.to_arrow_params_ipc(model, params=None) -> bytes` — export parameters as IPC bytes.

```python
import pyarrow as pa
import nextstat

# From PyArrow
table = pa.table({
    "channel": ["SR", "SR", "CR"],
    "sample":  ["signal", "background", "background"],
    "yields":  [[5., 10., 15.], [100., 200., 150.], [500., 600.]],
})
model = nextstat.from_arrow(table, poi="mu")
result = nextstat.fit(model)

# From Polars
import polars as pl
df = pl.read_parquet("histograms.parquet")
model = nextstat.from_arrow(df.to_arrow())

# Export to Arrow
yields = nextstat.to_arrow(model, what="yields")
params = nextstat.to_arrow(model, what="params")
```

## Remote Server Client (`nextstat.remote`)

Pure-Python HTTP client for a remote `nextstat-server` instance. Zero native dependencies — only requires `httpx`.

### Core

- `nextstat.remote.connect(url, *, timeout=300) -> NextStatClient` — create a client.
- `client.fit(workspace, *, model_id=None, gpu=True) -> FitResult` — remote MLE fit. Pass `workspace` (dict or JSON string) or `model_id` from the model cache.
- `client.ranking(workspace, *, model_id=None, gpu=True) -> RankingResult` — remote nuisance-parameter ranking.
- `client.health() -> HealthResult` — server health check (status, version, uptime, device, counters, cached_models).
- `client.close()` — close the connection. Also supports context manager (`with`).

### Batch API

- `client.batch_fit(workspaces, *, gpu=True) -> BatchFitResult` — fit up to 100 workspaces in one request. Returns `.results` (list of `FitResult | None`) and `.errors`.
- `client.batch_toys(workspace, *, params=None, n_toys=1000, seed=42, gpu=True) -> BatchToysResult` — GPU-accelerated toy fitting. Returns `.results` (list of `ToyFitItem`), `.n_converged`.

### Model Cache

- `client.upload_model(workspace, *, name=None) -> str` — upload a workspace to the server's LRU cache. Returns a `model_id` (SHA-256 hash).
- `client.list_models() -> list[ModelInfo]` — list cached models (id, name, params, channels, age, hits).
- `client.delete_model(model_id) -> bool` — evict a model from the cache.

Result types are typed dataclasses: `FitResult`, `RankingResult`, `RankingEntry`, `HealthResult`, `BatchFitResult`, `BatchToysResult`, `ToyFitItem`, `ModelInfo`.

Raises `NextStatServerError(status_code, detail)` on non-2xx HTTP responses.

```python
import nextstat.remote as remote

client = remote.connect("http://gpu-server:3742")

# Single fit
result = client.fit(workspace_json)
print(result.bestfit, result.nll, result.converged)

# Model cache — skip re-parsing on repeated calls
model_id = client.upload_model(workspace_json, name="my-analysis")
result = client.fit(model_id=model_id)  # ~4x faster

# Batch fit
batch = client.batch_fit([ws1, ws2, ws3])
for r in batch.results:
    print(r.nll if r else "failed")

# Batch toys (GPU-accelerated)
toys = client.batch_toys(workspace_json, n_toys=10_000, seed=42)
print(f"{toys.n_converged}/{toys.n_toys} converged in {toys.wall_time_s:.1f}s")

# Ranking
ranking = client.ranking(workspace_json)
for e in ranking.entries:
    print(f"{e.name}: Δμ = {e.delta_mu_up:+.3f} / {e.delta_mu_down:+.3f}")
```

## CLI parity

The CLI mirrors the core workflows for HEP (fit/hypotest/scan/limits) and time series.
See `docs/references/cli.md`.
