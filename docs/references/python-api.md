---
title: "Python API Reference (nextstat)"
status: stable
---

# Python API Reference (nextstat)

This page documents the public Python surface exported by `nextstat`.

Notes:
- The compiled extension is `nextstat._core` (PyO3/maturin).
- Convenience wrappers and optional modules live under `nextstat.*` (loaded on first access).
- Type stubs for the native extension (including overloads) are in `bindings/ns-py/python/nextstat/_core.pyi`.
- Installation, optional extras, and wheel build notes: `docs/references/python-packaging.md`.

## Top-level functions

### Model construction

- `nextstat.from_pyhf(json_str) -> HistFactoryModel` — create model from pyhf JSON workspace.
- `nextstat.from_histfactory_xml(xml_path) -> HistFactoryModel` — create model from HistFactory XML.
- `nextstat.UnbinnedModel.from_config(path) -> UnbinnedModel` — compile an event-level (unbinned) model from an `unbinned_spec_v0` JSON/YAML file. Supported PDF types: `gaussian`, `crystal_ball`, `double_crystal_ball`, `exponential`, `chebyshev`, `argus`, `voigtian`, `spline`, `histogram`, `histogram_from_tree`, `kde`, `kde_from_tree`, `product`, `flow`, `conditional_flow`, `dcr_surrogate`. The `flow`, `conditional_flow`, and `dcr_surrogate` types require building with `--features neural` (ONNX Runtime). See `docs/neural-density-estimation.md` for the full workflow.
- `nextstat.unbinned.from_config(path) -> nextstat.unbinned.UnbinnedAnalysis` — high-level unbinned workflow wrapper (compile + fit/fit_toys/scan/hypotest/toys/ranking helpers).
- `nextstat.workspace_audit(json_str) -> dict` — audit pyhf workspace for compatibility (counts channels, samples, modifiers; flags unsupported features).
- `nextstat.apply_patchset(workspace_json, patchset_json, *, patch_name=None) -> str` — apply a pyhf patchset.
- `nextstat.workspace_combine(ws1_json, ws2_json, *, join="none") -> str` — combine two pyhf workspace JSON strings. Join modes: `"none"` (error on conflict), `"outer"` (union), `"left_outer"`, `"right_outer"`.
- `nextstat.workspace_prune(ws_json, *, channels=[], samples=[], modifiers=[], measurements=[]) -> str` — remove channels, samples, modifiers, and/or measurements from a workspace.
- `nextstat.workspace_rename(ws_json, *, channels=None, samples=None, modifiers=None, measurements=None) -> str` — rename workspace elements. Each argument is an `{old: new}` dict.
- `nextstat.workspace_sorted(ws_json) -> str` — return workspace with all components in canonical (sorted) order.
- `nextstat.workspace_digest(ws_json) -> str` — compute SHA-256 digest of the canonical workspace.
- `nextstat.workspace_to_xml(ws_json, output_prefix="output") -> list[tuple[str, str]]` — export workspace to HistFactory XML. Returns `[(filename, xml_content), ...]`.
- `nextstat.simplemodel_uncorrelated(signal, bkg, bkg_uncertainty) -> str` — build workspace with uncorrelated background (shapesys). pyhf-compatible.
- `nextstat.simplemodel_correlated(signal, bkg, bkg_up, bkg_down) -> str` — build workspace with correlated background (histosys). pyhf-compatible.
- `nextstat.read_root_histogram(root_path, hist_path) -> dict` — read a TH1 histogram from a ROOT file. Returns `{name, title, bin_edges, bin_content, sumw2, underflow, overflow, underflow_sumw2, overflow_sumw2}`.
- `nextstat.histfactory_bin_edges_by_channel(xml_path) -> dict[str, list[float]]` — extract bin edges per channel from HistFactory XML.

Notes on HistFactory XML ingest (`nextstat.from_histfactory_xml`):
- `ShapeSys` histograms are treated as **relative** per-bin uncertainties and converted to absolute `sigma_abs = rel * nominal`.
- `StatError` histograms are treated as **relative** per-bin uncertainties and converted to absolute `sigma_abs = rel * nominal`.
- `StatError` follows channel `<StatErrorConfig ConstraintType="Poisson">` or `<StatErrorConfig ConstraintType="Gaussian">`:
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
- `HistFactoryModel.from_xml(xml_path) -> HistFactoryModel` — create model from HistFactory XML (`combination.xml` + referenced ROOT histograms).
- `HistFactoryModel.from_hs3(json_str, analysis=None, param_points=None) -> HistFactoryModel` — explicit HS3 loading with optional analysis selection and parameter point set.

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

result = nextstat.fit(model, device="cpu")
```

HS3 v0.2 support covers all modifier types produced by ROOT 6.37+: `normfactor`, `normsys`, `histosys`, `staterror`, `shapesys`, `shapefactor`, `lumi`. Unknown modifier/distribution types are silently skipped (forward-compatible).

### Fitting

- `nextstat.fit(model, *, data=None, init_pars=None, device="cpu") -> FitResult` — maximum likelihood estimation (`device="cuda"` for `HistFactoryModel`, CUDA build only).
- `nextstat.map_fit(posterior) -> FitResult` — MAP estimation for Bayesian posteriors.
- `nextstat.fit_batch(models_or_model, datasets=None) -> list[FitResult]` — batch fitting (homogeneous model lists; `datasets=` is supported for `HistFactoryModel` only).

### Hypothesis testing

- `nextstat.hypotest(poi_test, model, *, data=None, return_tail_probs=False) -> float | (float, list[float])` — asymptotic CLs.
- `nextstat.hypotest_toys(poi_test, model, *, n_toys=1000, seed=42, expected_set=False, data=None, return_tail_probs=False, return_meta=False) -> float | tuple | dict` — toy-based CLs.
- `nextstat.unbinned_hypotest(mu_test, model) -> dict` — compute unbinned `q_mu` (and `q0` if `mu=0` is within bounds).
- `nextstat.unbinned_hypotest_toys(poi_test, model, *, n_toys=1000, seed=42, expected_set=False, return_tail_probs=False, return_meta=False) -> float | tuple | dict` — toy-based CLs (qtilde) for unbinned models.

### Monte Carlo / Safety

- `nextstat.fault_tree_mc_ce_is(spec, *, n_per_level=10000, elite_fraction=0.01, max_levels=20, q_max=0.99, seed=42) -> dict` — Cross-Entropy Importance Sampling for rare-event fault tree probability estimation with multi-level adaptive biasing. Handles probabilities down to ~1e-16 via soft importance function when no TOP failures are observed. Returns `{p_failure, se, ci_lower, ci_upper, n_levels, n_total_scenarios, final_proposal, coefficient_of_variation, wall_time_s}`. Supports all failure modes: Bernoulli, WeibullMission, BernoulliUncertain.
- `nextstat.fault_tree_mc(spec, n_scenarios, seed=42, device='cpu', chunk_size=0) -> dict` — Monte Carlo fault tree simulation. `device`: `'cpu'`, `'cuda'`, `'metal'`.

### Profile likelihood

- `nextstat.profile_ci(model, fit_result, *, param_idx=None, chi2_level=3.841, tol=1e-4) -> dict | list[dict]` — profile likelihood confidence intervals for any `LogDensityModel`. If `param_idx` is given, returns a single dict; otherwise returns CI for all parameters. Each dict: `{param_idx, mle, ci_lower, ci_upper, n_evals}`.
- `nextstat.profile_scan(model, mu_values, *, data=None, device="cpu", return_params=False) -> dict` — profile likelihood scan.
- `nextstat.unbinned_profile_scan(model, mu_values) -> dict` — unbinned profile likelihood scan (q_mu) over POI values.
- `nextstat.upper_limit(model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, data=None) -> float` — upper limit via bisection.
- `nextstat.upper_limits(model, scan, *, alpha=0.05, data=None) -> (float, list[float])` — observed + expected limits from scan.
- `nextstat.upper_limits_root(model, *, alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, data=None) -> (float, list[float])` — ROOT-style limits.

### Sampling

- `nextstat.sample(model, *, method="nuts", return_idata=False, out=None, out_format="json", **kwargs) -> dict | InferenceData` — **Unified sampling interface**. Dispatches to NUTS, MAMS, or LAPS based on `method`. Set `return_idata=True` to get an ArviZ `InferenceData` object (requires `arviz`). Set `out="trace.json"` to save results to disk. All method-specific kwargs are forwarded to the underlying sampler.
- `nextstat.sample_nuts(model, *, n_chains=4, n_warmup=500, n_samples=1000, seed=42, max_treedepth=10, target_accept=0.8, init_strategy="random", init_jitter=0.0, init_jitter_rel=None, init_overdispersed_rel=None, data=None) -> dict` — NUTS (No-U-Turn Sampler). Also available via `nextstat.sample(model, method="nuts", ...)`. Accepts `Posterior` as well; `data=` is not supported when sampling a `Posterior`. `init_strategy`: `"random"` (default, Stan-style Uniform(-2,2)), `"mle"` (L-BFGS mode), or `"pathfinder"` (L-BFGS mode + diagonal inverse Hessian as initial mass matrix for faster warmup).
- `nextstat.sample_mams(model, *, n_chains=4, n_warmup=1000, n_samples=1000, seed=42, target_accept=0.9, init_strategy="random", metric="diagonal", init_step_size=0.0, init_l=0.0, max_leapfrog=1024, diagonal_precond=True, data=None) -> dict` — MAMS (Metropolis-Adjusted Microcanonical Sampler, arXiv:2503.01707). Also available via `nextstat.sample(model, method="mams", ...)`. Exact sampler using isokinetic dynamics on the unit velocity sphere. 4-phase Stan-style DualAveraging warmup with adaptive phase durations: when Pathfinder provides a Hessian-derived mass matrix, warmup phases are rebalanced (10%/15%/10%/65% vs default 15%/40%/15%/30%) to spend less time on mass matrix collection and more on equilibration. Returns ArviZ-compatible dict with `posterior`, `sample_stats`, `diagnostics`. Typically 1.3–1.7x better ESS/gradient than NUTS on hierarchical models. `init_strategy`: `"random"` (default), `"mle"`, or `"pathfinder"` (recommended for well-conditioned posteriors; avoid on funnel-like geometries).
- `nextstat.sample_laps(model, *, model_data=None, n_chains=4096, n_warmup=500, n_samples=2000, seed=42, target_accept=0.9, init_step_size=0.0, init_l=0.0, max_leapfrog=1024, device_ids=None) -> dict` — **LAPS** (Late-Adjusted Parallel Sampler): GPU-accelerated MAMS on CUDA. Also available via `nextstat.sample(model, method="laps", ...)`. Runs `n_chains` chains simultaneously on GPU with zero warp divergence (fixed trajectory length). Two-phase warmup: Phase 1 (unadjusted MCLMC) + Phase 2 (exact MH). `model`: `"std_normal"`, `"eight_schools"`, `"neal_funnel"`, `"glm_logistic"`, or a `RawCudaModel` instance. `model_data`: dict with model-specific data (e.g. `{"y": [...], "sigma": [...]}` for eight_schools, `{"dim": 10}` for std_normal). `device_ids`: list of GPU device indices (default `None` = auto-detect all GPUs). Multi-GPU: chains are split across devices with synchronized warmup adaptation and independent sampling. Returns same format as `sample_mams()` plus `wall_time_s`, `n_kernel_launches`, `n_gpu_chains`, `n_devices`, `device_ids`. Requires `cuda` feature and NVIDIA GPU at runtime.
- `nextstat.RawCudaModel(dim, cuda_src, *, data=None, param_names=None)` — User-defined CUDA model for LAPS JIT compilation via NVRTC. The `cuda_src` must define `__device__ double user_nll(const double* x, int dim, const double* model_data)` and `__device__ void user_grad(const double* x, double* grad, int dim, const double* model_data)`. The `data` array is uploaded to GPU as `model_data`. PTX is cached to disk (`~/.cache/nextstat/ptx/`) keyed by SHA-256(source + GPU arch). Requires `cuda` feature.
- `nextstat.bayes.sample(model, *, method="nuts", return_idata=True, **kwargs)` — convenience wrapper that returns ArviZ `InferenceData` by default. Supports all three methods (nuts/mams/laps).
- `nextstat.bayes.to_inferencedata(raw) -> InferenceData` — convert a raw sampling dict into ArviZ `InferenceData`.

#### Sampling quick start

```python
import nextstat as ns

# 1. Eight Schools — NUTS (3 lines)
model = ns.EightSchoolsModel([28,8,-3,7,-1,1,18,12], [15,10,16,11,9,11,10,18])
idata = ns.sample(model, method="nuts", n_samples=2000, return_idata=True)

# 2. Same model — MAMS (typically better ESS/grad on hierarchical models)
idata = ns.sample(model, method="mams", n_samples=2000, return_idata=True)

# 3. ArviZ diagnostics and plots
import arviz as az
az.summary(idata)          # R-hat, ESS, posterior summary
az.plot_trace(idata)       # trace + density plots
az.plot_pair(idata)        # pairwise scatter

# 4. Save to disk
idata = ns.sample(model, n_samples=2000, return_idata=True, out="trace.json")

# 5. GPU sampling with LAPS (requires CUDA build)
result = ns.sample("eight_schools", method="laps",
                   model_data={"y": [28,8,-3,7,-1,1,18,12],
                               "sigma": [15,10,16,11,9,11,10,18]},
                   n_chains=4096, n_samples=2000)

# 6. User-defined GPU model (NVRTC JIT, requires CUDA)
model = ns.RawCudaModel(dim=10, cuda_src=r'''
    __device__ double user_nll(const double* x, int dim, const double* data) {
        double v = data[0];
        double nll = 0.0;
        for (int i = 0; i < dim; i++) nll += 0.5 * x[i] * x[i] / v;
        return nll;
    }
    __device__ void user_grad(const double* x, double* grad, int dim, const double* data) {
        double v = data[0];
        for (int i = 0; i < dim; i++) grad[i] = x[i] / v;
    }
''', data=[1.0])
result = ns.sample_laps(model, n_chains=4096, n_samples=2000)

# 7. Raw dict (no ArviZ dependency needed)
raw = ns.sample(model, method="nuts", n_samples=1000)
print(raw["diagnostics"]["quality"]["status"])  # "ok" / "warn" / "fail"
```

### Toy data

- `nextstat.asimov_data(model, params) -> list[float]` — Asimov dataset (expected counts).
- `nextstat.poisson_toys(model, params, *, n_toys=1000, seed=42) -> list[list[float]]` — Poisson fluctuated toy datasets.
- `nextstat.fit_toys(model, params, *, n_toys=1000, seed=42) -> list[FitResult]` — CPU parallel toy fitting (Rayon; tape reuse).
- `nextstat.unbinned_fit_toys(model, params, *, n_toys=1000, seed=42, init_params=None, max_retries=3, max_iter=5000, compute_hessian=False) -> list[FitResult]` — generate and fit Poisson-fluctuated toys for unbinned models (CPU parallel path). Uses warm-start from MLE θ̂, retry with jitter, and smooth bounds escalation. Hessian is skipped by default (uncertainties=0) for throughput; set `compute_hessian=True` when parameter pulls are needed.
- `nextstat.fit_toys_batch(model, params, *, n_toys=1000, seed=42) -> list[FitResult]` — CPU parallel batch toy fitting (fast-path: no Hessian/covariance; `uncertainties = 0`).
- `nextstat.fit_toys_batch_gpu(model, params, *, n_toys=1000, seed=42, device="cpu") -> list[FitResult]` — GPU-accelerated batch toy fitting (see GPU section below).

### Visualization artifacts

- `nextstat.cls_curve(model, scan, *, alpha=0.05, data=None) -> dict` — asymptotic CLs exclusion curve. Returns `{alpha, nsigma_order, obs_limit, exp_limits, mu_values, cls_obs, cls_exp, points}`.
- `nextstat.profile_curve(model, mu_values, *, data=None) -> dict` — profile likelihood curve. Returns `{poi_index, mu_hat, nll_hat, mu_values, q_mu_values, twice_delta_nll, points}`.

### Parameter ranking

- `nextstat.ranking(model) -> list[dict]` — nuisance parameter ranking (impact on POI).
- `nextstat.unbinned_ranking(model) -> list[dict]` — nuisance parameter ranking (impact on POI) for unbinned models.
- `nextstat._core.ranking_gpu(model) -> list[dict]` — CUDA-only nuisance parameter ranking for `HistFactoryModel`. Requires building with `--features cuda` (`maturin develop --release --features cuda`). Not available in non-CUDA builds (function will not exist in `_core`). Returns same format as `ranking()`.
- `nextstat._core.ranking_metal(model) -> list[dict]` — Metal-only nuisance parameter ranking for `HistFactoryModel`. Requires building with `--features metal`. Same API as `ranking_gpu` but f32 precision.

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
- `termination_reason: str` — optimizer termination message (e.g. `"SolverConverged"`, `"TargetCostReached"`, `"MaxIterReached"`, `"1D golden-section search"`)
- `final_grad_norm: float` — L2 norm (Euclidean) of the gradient at minimum
- `initial_nll: float` — NLL at the starting point
- `n_active_bounds: int` — number of parameters at their box constraint boundary
- `edm: float` — Estimated Distance to Minimum (EDM = g^T H^{-1} g). Uses the L-BFGS inverse Hessian approximation. `NaN` if unavailable (gradient-free paths). Minuit-compatible convergence metric.
- `warnings: list[str]` — identifiability warnings (near-singular Hessian, non-finite uncertainties, near-zero Hessian diagonal). Empty list when model is well-identified.

Compatibility aliases:
- `bestfit` (same as `parameters`)
- `twice_nll = 2 * nll`
- `success` (same as `converged`)
- `n_evaluations` (back-compat alias for `n_iter`)

### `FitMinimumResult`

Fast-path optimizer result (no covariance/Hessian). Returned by `MaximumLikelihoodEstimator.fit_minimum(model, *, data=None, init_pars=None, bounds=None)`.

Fields:
- `parameters: list[float]`
- `nll: float`
- `converged: bool`
- `n_iter: int`
- `n_fev: int`, `n_gev: int`
- `message: str`
- `initial_nll: float`
- `final_gradient: list[float] | None`
- `edm: float` — Estimated Distance to Minimum (EDM = g^T H^{-1} g). `NaN` if unavailable.

Compatibility aliases:
- `bestfit` (same as `parameters`)
- `twice_nll = 2 * nll`
- `success` (same as `converged`)

### `MaximumLikelihoodEstimator`

The object-oriented MLE surface:

```python
import nextstat

mle = nextstat.MaximumLikelihoodEstimator(max_iter=1000, tol=1e-6, m=0, smooth_bounds=False)
res = mle.fit(model)
```

 Constructor args (keyword-only):
 - `max_iter=1000`: max optimizer iterations
 - `tol=1e-6`: convergence tolerance (gradient norm)
 - `m=0`: L-BFGS memory size (0 = auto-select based on model dimension: `max(10, min(50, n_params/5))`)
 - `smooth_bounds=False`: enable smooth bounds transform instead of hard clamping
 
 Also supports:
 - `fit_batch(models_or_model, datasets=None)` for homogeneous lists of models, or `HistFactoryModel` + multiple datasets.
 - `fit_minimum(model, *, data=None, init_pars=None, bounds=None) -> FitMinimumResult` — fast-path NLL minimization intended for profile scans and conditional fits.
   - `bounds=` is currently supported for `HistFactoryModel` only; clamp a parameter to `(value, value)` to fix it.
 - `fit_toys(model, params, *, n_toys=1000, seed=42) -> list[FitResult]` — CPU parallel toy fitting (Rayon).
 - `ranking(model) -> list[dict]` — nuisance parameter ranking.
 - `q0_like_loss_and_grad_nominal(model, *, channel, sample, nominal) -> (float, list[float])` — discovery q₀ and gradient w.r.t. one sample's nominal yields. Runs profiled fit internally. For ML training loops where the signal histogram is a differentiable function of NN weights.
 - `qmu_like_loss_and_grad_nominal(model, *, mu_test, channel, sample, nominal) -> (float, list[float])` — exclusion qμ and gradient. Same contract as q₀ but tests `mu = mu_test` instead of `mu = 0`.

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

### HEP / Unbinned (event-level)

- `UnbinnedModel.from_config(path) -> UnbinnedModel` — compile from `unbinned_spec_v0` JSON/YAML.
  - `schema_version() -> str`
  - `poi_index() -> int | None`
  - `with_fixed_param(param_idx, value) -> UnbinnedModel`
- `nextstat.unbinned.UnbinnedAnalysis(model)` — high-level workflow helper over `UnbinnedModel`.
  - `UnbinnedAnalysis.from_config(path) -> UnbinnedAnalysis`
  - `fit(*, init_pars=None) -> FitResult`
  - `fit_toys(params=None, *, n_toys=1000, seed=42) -> list[FitResult]`
  - `scan(mu_values) -> dict` (delegates to `unbinned_profile_scan`)
  - `hypotest(mu_test) -> dict`
  - `hypotest_toys(poi_test, *, n_toys=1000, seed=42, expected_set=False, return_tail_probs=False, return_meta=False) -> float | tuple | dict`
  - `ranking() -> list[dict]`
  - `parameter_index(param: int | str) -> int`
  - `with_fixed_param(param: int | str, value: float) -> UnbinnedAnalysis`
  - `summary() -> dict`

Toy-fit parity (Python vs CLI, same spec/seed):

```python
import nextstat

model = nextstat.UnbinnedModel.from_config("spec.json")
params = model.suggested_init()

# CPU batch: Rayon-parallel unbinned toy fits
results = nextstat.unbinned_fit_toys(model, params, n_toys=100, seed=42)

# Each result has: .parameters, .nll, .converged, .n_iter, .n_fev, .n_gev
converged = sum(1 for r in results if r.converged)
print(f"{converged}/{len(results)} toys converged")
```

```bash
nextstat unbinned-fit-toys --config spec.json --n-toys 100 --seed 42 --threads 1
# GPU parity variant:
nextstat unbinned-fit-toys --config spec.json --n-toys 100 --seed 42 --threads 1 --gpu cuda
```

### HEP / Hybrid (binned + unbinned)

- `HybridModel.from_models(binned, unbinned, poi_from="binned") -> HybridModel` — combine a `HistFactoryModel` and an `UnbinnedModel` into a single likelihood with shared parameters matched by name.
  - `poi_from`: `"binned"` (default) or `"unbinned"` — which model provides the POI.
  - `n_shared() -> int` — number of shared parameters.
  - `poi_index() -> int | None`
  - `with_fixed_param(param_idx, value) -> HybridModel`

```python
import nextstat

binned = nextstat.HistFactoryModel.from_workspace(open("workspace.json").read())
unbinned = nextstat.UnbinnedModel.from_config("unbinned.yaml")
hybrid = nextstat.HybridModel.from_models(binned, unbinned, poi_from="binned")

result = nextstat.fit(hybrid)
print(f"Shared params: {hybrid.n_shared()}, Total: {hybrid.n_params()}")
```

### Regression / GLM

- `LinearRegressionModel(x, y, *, include_intercept=True)`
- `LogisticRegressionModel(x, y, *, include_intercept=True)`
- `PoissonRegressionModel(x, y, *, include_intercept=True, offset=None)`
- `NegativeBinomialRegressionModel(x, y, *, include_intercept=True, offset=None)`
- `GammaRegressionModel(x, y, *, include_intercept=True)` — Gamma GLM with log link. Parameters: regression coefficients β + `log_alpha` (shape). For strictly positive continuous data (insurance claims, hospital costs).
- `TweedieRegressionModel(x, y, *, p=1.5, include_intercept=True)` — Tweedie compound Poisson-Gamma GLM with log link. Power `p ∈ (1, 2)`. Handles exact zeros. Parameters: β + `log_phi` (dispersion). For insurance aggregate claims, rainfall.
  - `.power()` — returns the Tweedie power parameter.
- `ComposedGlmModel` — hierarchical GLMs via static constructors:
  - `.linear_regression(x, y, *, include_intercept, group_idx, n_groups, coef_prior_mu, coef_prior_sigma, penalize_intercept, obs_sigma_prior_m, obs_sigma_prior_s, random_intercept_non_centered, random_slope_feature_idx, random_slope_non_centered, correlated_feature_idx, lkj_eta)`
  - `.logistic_regression(x, y, *, include_intercept, group_idx, n_groups, coef_prior_mu, coef_prior_sigma, penalize_intercept, random_intercept_non_centered, random_slope_feature_idx, random_slope_non_centered, correlated_feature_idx, lkj_eta)`
  - `.poisson_regression(x, y, *, include_intercept, offset, group_idx, n_groups, coef_prior_mu, coef_prior_sigma, penalize_intercept, random_intercept_non_centered, random_slope_feature_idx, random_slope_non_centered, correlated_feature_idx, lkj_eta)`

### Ordinal regression

- `OrderedLogitModel(x, y, *, n_levels)`
- `OrderedProbitModel(x, y, *, n_levels)`

### Hierarchical / mixed models

- `LmmMarginalModel(x, y, *, include_intercept, group_idx, n_groups, random_slope_feature_idx)` — Gaussian mixed model (marginal likelihood).

### Extreme Value Theory (EVT)

- `GevModel(data)` — Generalized Extreme Value distribution for block maxima. Parameters: `[mu, log_sigma, xi]` (location, log-scale, shape). Fréchet (ξ>0), Gumbel (ξ≈0), Weibull (ξ<0).
  - `GevModel.return_level(params, return_period)` — static method, computes the T-block return level (e.g. 100-year flood).
- `GpdModel(exceedances)` — Generalized Pareto Distribution for peaks-over-threshold. Parameters: `[log_sigma, xi]` (log-scale, shape).
  - `GpdModel.quantile(params, p)` — static method, computes excess quantile (VaR/ES).

### Meta-analysis

- `nextstat.meta_fixed(estimates, standard_errors, *, labels=None, conf_level=0.95) -> dict` — fixed-effects meta-analysis (inverse-variance weighting).
- `nextstat.meta_random(estimates, standard_errors, *, labels=None, conf_level=0.95) -> dict` — random-effects meta-analysis (DerSimonian–Laird).

**Returns** a dict with keys:
- `estimate`, `se`, `ci_lower`, `ci_upper`, `z`, `p_value` — pooled effect
- `method` — `"fixed"` or `"random"`
- `conf_level`, `k` — confidence level and number of studies
- `heterogeneity` — dict with `q`, `df`, `p_value`, `i_squared`, `h_squared`, `tau_squared`
- `forest` — list of per-study dicts with `label`, `estimate`, `se`, `ci_lower`, `ci_upper`, `weight`

### Survival analysis
- `ExponentialSurvivalModel(times, events)`
- `WeibullSurvivalModel(times, events)`
- `LogNormalAftModel(times, events)`
- `CoxPhModel(times, events, x, *, ties="efron")` — Cox proportional hazards model (partial likelihood).
- `IntervalCensoredWeibullModel(time_lower, time_upper, censor_type)` — Weibull model with interval censoring (exact, right, left, interval).
- `IntervalCensoredWeibullAftModel(time_lower, time_upper, censor_type, covariates)` — Weibull AFT with covariates and interval censoring. Parameters: `[log_k, beta_0, ..., beta_{p-1}]`. `log(λ_i) = x_i^T β`.
- `IntervalCensoredExponentialModel(time_lower, time_upper, censor_type)` — Exponential model (Weibull k=1) with interval censoring.
- `IntervalCensoredLogNormalModel(time_lower, time_upper, censor_type)` — LogNormal model with interval censoring.
 
High-level helpers (recommended for most users):
 
- `nextstat.survival.exponential.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.weibull.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.lognormal_aft.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.cox_ph.fit(times, events, x, *, ties="efron", robust=True, compute_cov=True, groups=None, cluster_correction=True, compute_baseline=True) -> CoxPhFit`
  - `robust=True` returns sandwich SE (`fit.robust_se`) (requires `compute_cov=True`)
  - `compute_cov=True` computes covariance/SE (`fit.cov`, `fit.se`)
  - `groups=cluster_ids` switches sandwich SE to cluster-robust (`fit.robust_kind == "cluster"`)
  - `cluster_correction=True` applies small-sample correction for clustered sandwich covariance (requires `groups`)
  - `compute_baseline=True` enables baseline hazard estimation (required for `fit.predict_survival(x_new, times=grid)`)
 
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
- `nextstat.timeseries.*` — convenience helpers and plotting.

### GARCH / Stochastic Volatility

Volatility models for financial time series. Available via the `nextstat.volatility` convenience module or directly from `nextstat._core`.

- `nextstat.volatility.garch(ys, *, max_iter=1000, tol=1e-6, alpha_beta_max=0.999, min_var=1e-18) -> dict` — fit Gaussian GARCH(1,1) by MLE.
  - Returns: `params` (`{mu, omega, alpha, beta}`), `conditional_variance`, `conditional_sigma`, `log_likelihood`, `converged`, `n_iter`, `message`.
- `nextstat.volatility.sv(ys, *, max_iter=1000, tol=1e-6, log_eps=1e-12) -> dict` — fit approximate stochastic volatility (SV) via log(χ²₁) Gaussian approximation + Kalman MLE.
  - Returns: `params` (`{mu, phi, sigma}`), `smoothed_h` (log-variance), `smoothed_sigma` (exp(h/2)), `log_likelihood`, `converged`, `n_iter`, `message`.

Low-level (same functions, no wrapper):
- `nextstat._core.garch11_fit(ys, *, max_iter, tol, alpha_beta_max, min_var) -> dict`
- `nextstat._core.sv_logchi2_fit(ys, *, max_iter, tol, log_eps) -> dict`

```python
from nextstat.volatility import garch, sv

returns = [0.01, -0.02, 0.005, 0.03, -0.015, 0.002, -0.04, 0.035]

# GARCH(1,1)
g = garch(returns)
print(f"omega={g['params']['omega']:.4f}, alpha={g['params']['alpha']:.4f}, beta={g['params']['beta']:.4f}")
print(f"Conditional sigma: {g['conditional_sigma'][:5]}")

# Stochastic Volatility
s = sv(returns)
print(f"phi={s['params']['phi']:.4f}, sigma_eta={s['params']['sigma']:.4f}")
print(f"Smoothed sigma: {s['smoothed_sigma'][:5]}")
```

### Pharmacometrics

#### Individual PK Models

- `OneCompartmentOralPkModel(times, y, *, dose, bioavailability=1.0, sigma=0.05, lloq=None, lloq_policy="censored")` — 1-compartment oral PK (3 params: CL, V, Ka).
  - `predict(params) -> list[float]` — predicted concentrations.
- `TwoCompartmentIvPkModel(times, y, *, dose, error_model="additive", sigma=0.05, sigma_add=None, lloq=None, lloq_policy="censored")` — 2-compartment IV bolus PK (4 params: CL, V1, V2, Q). Supports additive/proportional/combined error models.
  - `predict(params) -> list[float]` — predicted concentrations.
- `TwoCompartmentOralPkModel(times, y, *, dose, bioavailability=1.0, error_model="additive", sigma=0.05, sigma_add=None, lloq=None, lloq_policy="censored")` — 2-compartment oral PK (5 params: CL, V1, V2, Q, Ka). Supports additive/proportional/combined error models.
  - `predict(params) -> list[float]` — predicted concentrations.

All PK models implement the `LogDensityModel` interface: `nll(params)`, `grad_nll(params)`, `parameter_names()`, `suggested_init()`, `suggested_bounds()`, `n_params()`, `dim()`.

#### Population PK (NLME)

- `OneCompartmentOralPkNlmeModel(times, y, subject_idx, n_subjects, *, dose, bioavailability=1.0, sigma=0.05, lloq=None, lloq_policy="censored")` — population PK (NLME with per-subject random effects). `LogDensityModel` interface.

- `nextstat.nlme_foce(times, y, subject_idx, n_subjects, *, dose, bioavailability=1.0, error_model="proportional", sigma=0.1, sigma_add=None, theta_init, omega_init, max_outer_iter=100, max_inner_iter=20, tol=1e-4, interaction=True) -> dict` — FOCE/FOCEI population estimation (1-cpt oral). Returns `theta`, `omega`, `omega_matrix`, `correlation`, `eta`, `ofv`, `converged`, `n_iter`.

- `nextstat.nlme_saem(times, y, subject_idx, n_subjects, *, dose, bioavailability=1.0, error_model="proportional", sigma=0.1, sigma_add=None, theta_init, omega_init, n_burn=200, n_iter=100, n_chains=1, seed=12345, tol=1e-4) -> dict` — SAEM population estimation (1-cpt oral). Returns FOCE-result dict plus `saem` sub-dict with `acceptance_rates`, `ofv_trace`, `burn_in_only`.

#### Model Diagnostics

- `nextstat.pk_vpc(times, y, subject_idx, n_subjects, *, dose, bioavailability=1.0, theta, omega_matrix, error_model="proportional", sigma=0.1, sigma_add=None, n_sim=200, quantiles=None, n_bins=10, seed=42, pi_level=0.90) -> dict` — Visual Predictive Check (1-cpt oral). Returns `bins` (list of per-bin quantile comparisons), `quantiles`, `n_sim`.

- `nextstat.pk_gof(times, y, subject_idx, *, dose, bioavailability=1.0, theta, eta, error_model="proportional", sigma=0.1, sigma_add=None) -> list[dict]` — Goodness of Fit (1-cpt oral). Returns per-observation records with `subject`, `time`, `dv`, `pred`, `ipred`, `iwres`, `cwres`.

#### Data I/O

- `nextstat.read_nonmem(csv_text) -> dict` — parse NONMEM-format CSV. Returns `n_subjects`, `subject_ids`, `times`, `dv`, `subject_idx`.

### Test / utility models

- `GaussianMeanModel(y, sigma)` — simple Gaussian mean estimation.
- `FunnelModel(dim=2)` — Neal's funnel (sampler stress test). `dim` controls dimensionality: `y ~ N(0,3)`, `x_i|y ~ N(0, exp(y/2))` for `i = 1..dim-1`.
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
- `nextstat.volatility` — GARCH(1,1) and stochastic volatility convenience wrappers.
- `nextstat.gym` — Gymnasium/Gym environments for RL / design-of-experiments (requires `gymnasium` + `numpy`). See below.
- `nextstat.mlops` — fit metrics extraction for experiment loggers (W&B, MLflow, Neptune).
- `nextstat.interpret` — systematic-impact ranking as ML-style Feature Importance.
- `nextstat.glm` — regression/GLM convenience wrappers.
- `nextstat.ordinal` — ordinal regression convenience wrappers.
- `nextstat.formula` — Patsy-like formula interface (see below).
- `nextstat.ppc` — posterior predictive checks (see below).
- `nextstat.missing` — missing data helpers (see below).
- `nextstat.arrow_io` — Arrow/Parquet authoring helpers (schema validation + manifest writing; requires `pyarrow`).
- `nextstat.analysis` — TREx replacement workflow helpers (ROOT HIST ingest + expression eval; requires `numpy`, and `_core` for ROOT IO).
- `nextstat.trex_config` — TRExFitter `.config` parser + conversion to analysis spec (no native deps).
- `nextstat.audit` — reproducible local run bundles (no optional deps).
- `nextstat.report` — render report artifacts to PDF/SVG (requires `matplotlib`).
- `nextstat.validation_report` — render `validation_report.json` to PDF (requires `matplotlib`).

### `nextstat.econometrics` (core functions)

Low-level econometrics functions exposed via `nextstat._core`:

- `nextstat._core.panel_fe(entity_ids, x, y, p, *, cluster_ids=None) -> dict` — panel fixed-effects regression. Returns `coefficients`, `se_ols`, `se_cluster`, `r_squared_within`, `n_obs`, `n_entities`, `n_regressors`, `rss`.
- `nextstat._core.did(y, treat, post, cluster_ids) -> dict` — difference-in-differences estimator. Returns `att`, `se`, `se_cluster`, `t_stat`, `mean_treated_post`, `mean_treated_pre`, `mean_control_post`, `mean_control_pre`, `n_obs`.
- `nextstat._core.event_study(y, entity_ids, time_ids, relative_time, min_lag, max_lag, reference_period, cluster_ids) -> dict` — event study with dynamic treatment effects. Returns `relative_times`, `coefficients`, `se_cluster`, `ci_lower`, `ci_upper`, `n_obs`, `reference_period`.
- `nextstat._core.iv_2sls(y, x_exog, k_exog, x_endog, k_endog, z, m, *, exog_names=None, endog_names=None, cluster_ids=None) -> dict` — instrumental variables / 2SLS. Returns `coefficients`, `names`, `se`, `se_cluster`, `n_obs`, `n_instruments`, `first_stage`.
- `nextstat._core.aipw_ate(y, treat, propensity, mu1, mu0, *, trim=0.01) -> dict` — augmented IPW average treatment effect. Returns `ate`, `se`, `ci_lower`, `ci_upper`, `n_treated`, `n_control`, `mean_propensity`.
- `nextstat.rosenbaum_bounds(y_treated, y_control, gammas) -> dict` — Rosenbaum bounds in the native extension (matched pairs). Returns `gammas`, `p_upper`, `p_lower`, `gamma_critical`.
- `nextstat.causal.aipw.rosenbaum_bounds(y_treated, y_control, *, gammas=None) -> RosenbaumBoundsResult` — Python convenience wrapper for the same bounds (matched pairs).

The `nextstat.econometrics` convenience module provides higher-level, dependency-light Python estimators and reporting helpers.

### `nextstat.glm` (convenience)

Convenience wrappers around `nextstat._core` regression models:

- `nextstat.glm.linear(x, y, *, intercept=True) -> FitResult`
- `nextstat.glm.logistic(x, y, *, intercept=True) -> FitResult`
- `nextstat.glm.poisson(x, y, *, intercept=True, offset=None) -> FitResult`
- `nextstat.glm.negbin(x, y, *, intercept=True, offset=None) -> FitResult`
- `nextstat.glm.gamma(x, y, *, intercept=True) -> FitResult`
- `nextstat.glm.tweedie(x, y, *, p=1.5, intercept=True) -> FitResult`

### `nextstat.ordinal` (convenience)

- `nextstat.ordinal.logit(x, y, *, n_levels) -> FitResult`
- `nextstat.ordinal.probit(x, y, *, n_levels) -> FitResult`

### `nextstat.formula` (Patsy-like interface)

- `nextstat.formula.parse_formula(formula_str) -> (str, list[str], bool)` — parse a minimal Wilkinson-style formula and return `(y_name, terms, include_intercept)`.
- `nextstat.formula.to_columnar(data, columns) -> Mapping[str, Sequence[Any]]` — normalize tabular inputs (dict-of-columns, list-of-dict rows, or pandas DataFrame) into a dict-of-columns view.
- `nextstat.formula.design_matrices(formula_str, data, *, categorical=None) -> (list[float], list[list[float]], list[str])` — build deterministic `(y, X, column_names)`.

### `nextstat.ppc` (posterior predictive checks)

- `nextstat.ppc.ppc_glm_from_sample(spec, sample_raw, *, param_names=None, n_draws=50, seed=0, stats_fn=None) -> PpcStats` — PPC from a raw `nextstat.sample(...)` dict for GLM specs.
- `nextstat.ppc.ppc_negbin_from_sample(spec, sample_raw, *, param_names=None, n_draws=50, seed=0, stats_fn=None) -> PpcStats` — PPC for NB2 (mean/dispersion) regression.
- `nextstat.ppc.ppc_ordered_from_sample(spec, sample_raw, *, param_names=None, n_draws=50, seed=0, stats_fn=None) -> PpcStats` — PPC for ordered outcomes (logit/probit).

### `nextstat.missing` (missing data)

- `nextstat.missing.apply_policy(x, y=None, *, policy="drop_rows") -> MissingResult` — apply an explicit missing-data policy (`"drop_rows"` or `"impute_mean"`).

### `nextstat.arrow_io` (Arrow/Parquet authoring, `pyarrow`)

Requires `pyarrow` (install with: `pip install "nextstat[io]"`).

- `nextstat.arrow_io.validate_histogram_table(table) -> HistogramTableStats` — validate the histogram table contract.
- `nextstat.arrow_io.validate_modifiers_table(table) -> ModifiersTableStats` — validate the modifiers table contract (binned Parquet v2).
- `nextstat.arrow_io.write_histograms_parquet(table, path, *, compression="zstd", write_manifest=True, manifest_path=None, poi="mu", observations=None, observations_path=None) -> dict` — write Parquet + optional manifest JSON; returns manifest dict.
- `nextstat.arrow_io.validate_histograms_parquet_manifest(manifest, *, check_sha256=True) -> None` — validate a manifest against the referenced Parquet file.
- `nextstat.arrow_io.load_parquet_as_histfactory_model(path, *, poi="mu", observations=None) -> HistFactoryModel` — validate Parquet schema with PyArrow, then call `nextstat.from_parquet()`.
- `nextstat.arrow_io.load_parquet_v2_as_histfactory_model(yields_path, modifiers_path, *, poi="mu", observations=None) -> HistFactoryModel` — validate Parquet v2 schemas, then call `nextstat.from_parquet_with_modifiers()`.
- `nextstat.arrow_io.validate_event_table(table) -> EventTableStats` — validate an unbinned event table contract.
- `nextstat.arrow_io.write_events_parquet(table, path, *, observables=None, compression="zstd") -> dict` — write an unbinned event table to Parquet with NextStat metadata.

Types: `HistogramTableStats`, `ModifiersTableStats`, `EventTableStats`. Constants: `HISTOGRAM_TABLE_MANIFEST_V1`, `UNBINNED_EVENTS_SCHEMA_V1`.

### `nextstat.analysis` (TREx replacement helpers)

- `nextstat.analysis.read_root_histogram(root_path, hist_path, *, flow_policy="drop") -> dict` — read one ROOT TH1 histogram and optionally fold under/overflow into the edge bins. Guarantees `sumw2` (and adds `sumw2_policy`) and adds `flow_policy` to the returned dict.
- `nextstat.analysis.read_root_histograms(root_path, hist_paths, *, flow_policy="drop") -> dict[str, dict]` — read many histograms from the same ROOT file.
- `nextstat.analysis.eval_expr(expr, env, *, n=None) -> np.ndarray` — vectorized TREx-style expression evaluation (requires `numpy`). `env` maps variable names to 1D NumPy arrays/scalars (and list-of-arrays for indexed vars like `jets_pt[0]`).

### `nextstat.trex_config` (TRExFitter config migration)

- `nextstat.trex_config.parse_trex_config(text, *, path=None) -> TrexConfigDoc` — parse a TREx `.config` file from a string.
- `nextstat.trex_config.parse_trex_config_file(path) -> TrexConfigDoc` — parse a TREx `.config` file from disk.
- `nextstat.trex_config.trex_doc_to_analysis_spec_v0(doc, *, source_path=None, out_path=None, threads=1, workspace_out="tmp/trex_workspace.json") -> (dict, dict)` — convert TREx config doc to analysis spec v0. Returns `(spec, report)`; unsupported keys are recorded in the report.
- `nextstat.trex_config.trex_config_file_to_analysis_spec_v0(config_path, *, out_path=None, threads=1, workspace_out="tmp/trex_workspace.json") -> (dict, dict)` — convenience wrapper: parse file -> convert -> `(spec, report)`.
- `nextstat.trex_config.dump_yaml(obj) -> str` — deterministic minimal YAML emitter (no external deps).

Types: `TrexConfigParseError`, `TrexConfigImportError`, `TrexConfigDoc`, `TrexConfigBlock`, `TrexConfigEntry`, `TrexValue`.

### `nextstat.audit` (run bundles)

- `nextstat.audit.environment_fingerprint() -> dict[str, Any]` — small, privacy-preserving environment fingerprint for reproducibility.
- `nextstat.audit.write_bundle(bundle_dir, *, command, args, input_path, output_value, tool_version=None) -> None` — write a reproducible run bundle to a directory (Python mirror of CLI `--bundle`).

Dataclasses: `BundleMeta`, `BundleInputMeta`.

### `nextstat.report` (rendering, matplotlib)

- `nextstat.report.render_report(input_dir, *, pdf, svg_dir, corr_include=None, corr_exclude=None, corr_top_n=None) -> None` — render a report PDF (+ optional per-plot SVG) from an artifacts directory (requires `matplotlib`).

CLI entry: `python -m nextstat.report render --input-dir ... --pdf ... [--svg-dir ...]`.

### `nextstat.validation_report` (rendering, matplotlib)

CLI entry: `python -m nextstat.validation_report render --input validation_report.json --pdf out.pdf` (requires `matplotlib`).

### `GpuFlowSession` (CUDA, feature-gated)

GPU-accelerated NLL reduction for unbinned models with neural PDFs:

- `nextstat._core.GpuFlowSession(n_events, n_params, processes, *, gauss_constraints=None, constraint_const=0.0)` — create session.
  - `nll(logp_flat, params) -> float` — compute NLL from pre-computed log-probabilities.
  - `nll_device_ptr_f32(d_logp_flat_ptr, params) -> float` — device-resident path (zero-copy from ONNX CUDA EP).
  - `compute_yields(params) -> list[float]` — yield computation.
  - `n_events() -> int`, `n_procs() -> int`, `n_params() -> int`

---
## Differentiable Layer (PyTorch integration)

The `nextstat.torch` module provides `torch.autograd.Function` wrappers for end-to-end differentiable HEP inference. Two backends:

### CPU path (no CUDA required)

- `nextstat.torch.NextStatQ0` — `torch.autograd.Function` for discovery q0. CPU profile fit with envelope-theorem gradient.
- `nextstat.torch.NextStatZ0` — same but returns discovery significance Z0.
- `nextstat.torch.discovery_z0(nominal, *, mle, model, channel, sample, eps=1e-12) -> torch.Tensor` — convenience wrapper returning differentiable Z0.

### CUDA path (zero-copy, requires `cuda` feature build)

- `nextstat.torch.create_session(model, signal_sample_name="signal") -> nextstat._core.DifferentiableSession` — create CUDA session for differentiable NLL (requires building with `--features cuda`).
- `nextstat.torch.nll_loss(signal_histogram, session, params=None) -> torch.Tensor` — differentiable NLL (zero-copy). `signal_histogram` must be contiguous CUDA `float64`.
- `nextstat.torch.create_profiled_session(model, signal_sample_name="signal", device="auto") -> nextstat._core.ProfiledDifferentiableSession | nextstat._core.MetalProfiledDifferentiableSession` — create GPU session for profiled test statistics.
- `nextstat.torch.profiled_q0_loss(signal_histogram, session) -> torch.Tensor` — differentiable profiled q₀.
- `nextstat.torch.profiled_z0_loss(signal_histogram, session, eps=1e-12) -> torch.Tensor` — differentiable Z₀ = sqrt(q₀).
- `nextstat.torch.profiled_qmu_loss(signal_histogram, session, mu_test) -> torch.Tensor` — differentiable profiled qμ.
- `nextstat.torch.batch_profiled_qmu_loss(signal_histogram, session, mu_values) -> list[torch.Tensor]` — qμ for multiple `mu_test` values.
- `nextstat.torch.profiled_zmu_loss(signal_histogram, session, mu_test, eps=1e-12) -> torch.Tensor` — differentiable Zμ = sqrt(qμ).
- `nextstat.torch.batch_profiled_q0_loss(signal_histograms, session) -> list[torch.Tensor]` — q₀ over a batch of histograms.
- `nextstat.torch.signal_jacobian(signal_histogram, session) -> torch.Tensor` — extract ∂q₀/∂signal without going through autograd.
- `nextstat.torch.signal_jacobian_numpy(signal_histogram, session) -> np.ndarray` — NumPy variant of `signal_jacobian`.

### Metal path (f32, requires `metal` feature build)

- `MetalProfiledDifferentiableSession(model, signal_sample_name)` — Metal GPU session for profiled test statistics (f32 compute). Signal is uploaded via CPU (no raw-pointer interop with MPS tensors).

High-level helpers:
- `nextstat.torch.SignificanceLoss(model, signal_sample_name="signal", *, device="auto", negate=True, eps=1e-12)` — ML-friendly callable loss wrapper around profiled Z₀.
- `nextstat.torch.SoftHistogram(bin_edges, bandwidth="auto", mode="kde")` — differentiable binning (KDE/sigmoid) to produce a signal histogram for `SignificanceLoss`.

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

model = nextstat.UnbinnedModel.from_config("spec.json")
params = model.suggested_init()

# CPU parallel: Rayon parallel
results = nextstat.unbinned_fit_toys(model, params, n_toys=100, seed=42)

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

## MLOps Integration (`nextstat.mlops`)

Lightweight helpers to extract NextStat metrics as plain dicts for experiment loggers.

- `nextstat.mlops.metrics_dict(fit_result, *, prefix="", include_time=True, extra=None) -> dict[str, float]` — flat dict from `FitResult`.
- `nextstat.mlops.significance_metrics(z0, q0=0.0, *, prefix="", step_time_ms=0.0) -> dict[str, float]`
- `nextstat.mlops.StepTimer` — lightweight wall-clock timer: `.start()`, `.stop() -> float` (ms).

## Interpretability (`nextstat.interpret`)

Systematic-impact ranking translated into ML-style Feature Importance.

- `nextstat.interpret.rank_impact(model, *, gpu=False, sort_by="total", top_n=None, ascending=False) -> list[dict]`
- `nextstat.interpret.rank_impact_df(model, **kwargs) -> pd.DataFrame` (requires `pandas`)
- `nextstat.interpret.plot_rank_impact(model, *, top_n=20, gpu=False, figsize=(8,6), title="Systematic Impact on Signal Strength (μ)", ax=None, **kwargs) -> matplotlib.Figure` (requires `matplotlib`)

## Agentic Analysis (`nextstat.tools`)

- `nextstat.tools.get_toolkit(*, transport="local", server_url=None, timeout_s=10.0) -> list[dict]`
- `nextstat.tools.execute_tool(name, arguments, *, transport="local", server_url=None, timeout_s=30.0, fallback_to_local=True) -> dict`
- `nextstat.tools.execute_tool_raw(name, arguments) -> dict`
- `nextstat.tools.get_langchain_tools() -> list[StructuredTool]` (requires `langchain-core`)
- `nextstat.tools.get_mcp_tools() -> list[dict]`
- `nextstat.tools.handle_mcp_call(name, arguments) -> dict`
- `nextstat.tools.get_tool_names() -> list[str]`
- `nextstat.tools.get_tool_schema(name) -> dict | None`

## Neural PDFs (`FlowPdf`, `DcrSurrogate`)

Standalone ONNX-backed normalizing flow and DCR surrogate classes. Requires building with `--features neural` (`maturin develop --release --features neural`).

## Surrogate Distillation (`nextstat.distill`)

Generate training datasets for neural likelihood surrogates. NextStat serves as the ground-truth oracle; the surrogate provides nanosecond inference.

- `nextstat.distill.generate_dataset(model, n_samples=100_000, *, method="sobol", bounds=None, seed=42, include_gradient=True, batch_size=10_000, gpu=False) -> SurrogateDataset` — sample parameter space and evaluate NLL + gradient at each point. Methods: `"sobol"`, `"lhs"`, `"uniform"`, `"gaussian"`.
  - `gpu` is currently reserved/ignored by the Python implementation.
- `nextstat.distill.SurrogateDataset` — container with `.parameters`, `.nll`, `.gradient` (NumPy arrays), `.parameter_names`, `.parameter_bounds`, `.metadata`.
- `nextstat.distill.to_torch_dataset(ds) -> TensorDataset` — convert for PyTorch DataLoader.
- `nextstat.distill.to_numpy(ds) -> dict[str, ndarray]` — export as dict of arrays.
- `nextstat.distill.to_npz(ds, path)` / `nextstat.distill.from_npz(path)` — save/load compressed `.npz`.
- `nextstat.distill.to_parquet(ds, path)` — export to Parquet (zstd, requires `pyarrow`).
- `nextstat.distill.train_mlp_surrogate(ds, *, hidden_layers=(256,256,128), epochs=100, lr=1e-3, batch_size=4096, val_fraction=0.1, grad_weight=0.1, device="cpu", verbose=True) -> nn.Module` — convenience MLP trainer with Sobolev loss.
- `nextstat.distill.predict_nll(surrogate, params_np) -> ndarray` — evaluate trained surrogate on raw parameters.

## Gymnasium / RL Environment (`nextstat.gym`)

Gymnasium-compatible environment for optimizing a single sample's nominal yields via reinforcement learning. Requires `gymnasium` (or legacy `gym`) + `numpy`.

- `nextstat.gym.HistFactoryEnv(*, workspace_json, channel, sample, reward_metric="nll", mu_test=5.0, max_steps=128, action_scale=0.05, action_mode="logmul", init_noise=0.0, clip_min=1e-12, clip_max=1e12, fixed_params=None, mle_max_iter=200, mle_tol=1e-6, mle_m=10)`
- `nextstat.gym.make_histfactory_env(workspace_json, *, channel, sample, reward_metric="nll", **kwargs) -> HistFactoryEnv`

Reward metrics: `"nll"`, `"q0"`, `"z0"`, `"qmu"`, `"zmu"`. Action modes: `"add"`, `"logmul"`.

## Arrow / Polars Integration

IPC-based interchange between NextStat and the Arrow columnar ecosystem (PyArrow, Polars, DuckDB, Spark). Backed by Rust `arrow` 57.3 + `parquet` 57.3 crates; the Python ↔ Rust bridge uses Arrow IPC.

See also: `docs/references/arrow-parquet-io.md` (schema contract + manifest for reproducible pipelines).

For Parquet/Arrow *authoring* from Python (schema validation + manifest writing): see `nextstat.arrow_io` (requires `pyarrow`).

### High-level API

- `nextstat.from_arrow(table, *, poi="mu", observations=None) -> HistFactoryModel` — create a model from a **PyArrow Table** or **RecordBatch** (requires `pyarrow`). The table must have columns `channel` (Utf8), `sample` (Utf8), `yields` (List\<Float64\>), optionally `stat_error` (List\<Float64\>). Works with any Arrow-compatible source (Polars, DuckDB, Spark).
- `nextstat.to_arrow(model, *, params=None, what="yields") -> pyarrow.Table` — export model data as a PyArrow table (requires `pyarrow`). `what="yields"` returns expected yields per channel; `what="params"` returns parameter metadata (name, index, value, bounds, init).
- `nextstat.from_parquet(path, poi="mu", observations=None) -> HistFactoryModel` — read a Parquet file directly (Snappy; Zstd support depends on build features), same schema as `from_arrow`.
- `nextstat.from_parquet_with_modifiers(yields_path, modifiers_path, poi="mu", observations=None) -> HistFactoryModel` — read **two** Parquet files: yields table + modifiers table (binned Parquet v2).

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

- `nextstat.remote.connect(url, *, timeout=300.0) -> NextStatClient` — create a client.
- `client.fit(workspace=None, *, model_id=None, gpu=True) -> FitResult` — remote MLE fit. Pass `workspace` (dict or JSON string) or `model_id` from the model cache.
- `client.ranking(workspace=None, *, model_id=None, gpu=True) -> RankingResult` — remote nuisance-parameter ranking.
- `client.health() -> HealthResult` — server health check (status, version, uptime, device, counters, cached_models).
- `client.close()` — close the connection. Also supports context manager (`with`).

### Batch API

- `client.batch_fit(workspaces, *, gpu=True) -> BatchFitResult` — fit multiple workspaces in one request. Returns `.results` (list of `FitResult | None`) and `.errors`.
- `client.batch_toys(workspace, *, params=None, n_toys=1000, seed=42, gpu=True) -> BatchToysResult` — GPU-accelerated toy fitting. Returns `.results` (list of `ToyFitItem`), `.n_converged`, `.n_failed`.

### Model Cache

- `client.upload_model(workspace, *, name=None) -> str` — upload a workspace to the server's model cache. Returns a `model_id` (SHA-256 hash).
- `client.list_models() -> list[ModelInfo]` — list cached models (model_id, name, n_params, n_channels, age_s, last_used_s, hit_count).
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

## Survival analysis (non-parametric)

### `nextstat.kaplan_meier(times, events, *, conf_level=0.95) -> dict`

Kaplan-Meier survival estimator with Greenwood variance and log-log confidence intervals.

**Arguments:**
- `times` — list of observation times (≥ 0).
- `events` — list of booleans (`True` = event, `False` = right-censored).
- `conf_level` — confidence level for pointwise CIs (default 0.95).

**Returns** a dict with keys:
- `n`, `n_events`, `conf_level`, `median` (or `None`)
- `time`, `n_risk`, `n_event`, `n_censored` — per-step lists
- `survival`, `variance`, `ci_lower`, `ci_upper` — per-step lists

### `nextstat.log_rank_test(times, events, groups) -> dict`

Log-rank (Mantel-Cox) test comparing survival distributions of 2+ groups.

**Arguments:**
- `times` — list of observation times.
- `events` — list of booleans.
- `groups` — list of integer group labels (same length as `times`).

**Returns** a dict with keys:
- `n`, `chi_squared`, `df`, `p_value`
- `group_ids`, `observed`, `expected` — per-group lists

## Churn / Subscription vertical

### `nextstat.churn_generate_data(*, n_customers=2000, n_cohorts=6, max_time=24.0, treatment_fraction=0.3, seed=42) -> dict`

Generate a synthetic SaaS churn dataset (deterministic, seeded).

**Returns** a dict with keys: `n`, `n_events`, `times`, `events`, `groups`, `treated`, `covariates`, `covariate_names`, `plan`, `region`, `cohort`, `usage_score`.

### `nextstat.churn_retention(times, events, groups, *, conf_level=0.95) -> dict`

Cohort retention analysis: stratified KM + log-rank comparison.

**Returns** a dict with keys: `overall` (KM summary), `by_group` (list of per-group KM), `log_rank` (chi_squared, df, p_value).

### `nextstat.churn_risk_model(times, events, covariates, names, *, conf_level=0.95) -> dict`

Cox PH churn risk model with hazard ratios and CIs.

**Returns** a dict with keys: `n`, `n_events`, `nll`, `names`, `coefficients`, `se`, `hazard_ratios`, `hr_ci_lower`, `hr_ci_upper`.

### `nextstat.churn_uplift(times, events, treated, covariates, *, horizon=12.0) -> dict`

AIPW causal uplift estimate of intervention impact on churn.

**Returns** a dict with keys: `ate`, `se`, `ci_lower`, `ci_upper`, `n_treated`, `n_control`, `gamma_critical`, `horizon`.

### `nextstat.churn_diagnostics(times, events, groups, *, treated=[], covariates=[], covariate_names=[], trim=0.01) -> dict`

Data quality diagnostics and trust gate for churn analysis. Checks censoring rates per segment, covariate balance (SMD), propensity overlap, and sample-size adequacy.

**Returns** a dict with keys:
- `n`, `n_events`, `overall_censoring_frac`, `trust_gate_passed`
- `censoring_by_segment` — list of `{group, n, n_events, n_censored, frac_censored}`
- `covariate_balance` — list of `{name, smd_raw, mean_treated, mean_control}`
- `propensity_overlap` — `{quantiles, mean, n_trimmed_low, n_trimmed_high, trim}` or `None`
- `warnings` — list of `{category, severity, message}`

### `nextstat.churn_cohort_matrix(times, events, groups, period_boundaries) -> dict`

Life-table cohort retention matrix. For each cohort and period, computes at-risk count, events, censored, period retention rate, and cumulative retention.

**Arguments:**
- `period_boundaries` — sorted time points defining period ends (e.g. `[1, 3, 6, 12, 24]`).

**Returns** a dict with keys:
- `period_boundaries` — echo of input boundaries
- `cohorts` — list of `{cohort, n_total, n_events, periods}` where each period is `{n_at_risk, n_events, n_censored, retention_rate, cumulative_retention}`
- `overall` — same structure, all cohorts combined

### `nextstat.churn_compare(times, events, groups, *, conf_level=0.95, correction="benjamini_hochberg", alpha=0.05) -> dict`

Pairwise segment comparison with log-rank tests, hazard ratio proxies, and multiple comparisons correction.

**Arguments:**
- `correction` — `"benjamini_hochberg"` (alias `"bh"`) or `"bonferroni"`.

**Returns** a dict with keys:
- `overall_chi_squared`, `overall_p_value`, `overall_df`, `alpha`, `n`, `n_events`, `correction_method`
- `segments` — list of `{group, n, n_events, median, observed, expected}`
- `pairwise` — list of `{group_a, group_b, chi_squared, p_value, p_adjusted, hazard_ratio_proxy, median_diff, significant}`

### `nextstat.churn_uplift_survival(times, events, treated, *, covariates=[], horizon=12.0, eval_horizons=[3,6,12,24], trim=0.01) -> dict`

Survival-native causal uplift: RMST, IPW-weighted Kaplan-Meier, and ΔS(t) at specified horizons.

**Returns** a dict with keys:
- `rmst_treated`, `rmst_control`, `delta_rmst`, `horizon`, `ipw_applied`
- `arms` — list of `{arm, n, n_events, rmst, median}`
- `survival_diffs` — list of `{horizon, survival_treated, survival_control, delta_survival}`
- `overlap` — `{n_total, n_after_trim, n_trimmed, mean_propensity, min_propensity, max_propensity, ess_treated, ess_control}`

### `nextstat.churn_bootstrap_hr(times, events, covariates, names, *, n_bootstrap=1000, seed=42, conf_level=0.95) -> dict`

Bootstrap hazard ratios via parallel Cox PH refitting. Returns percentile-based CIs.

**Returns** a dict with keys: `names`, `hr_point`, `hr_ci_lower`, `hr_ci_upper`, `n_bootstrap`, `n_converged`, `elapsed_s`.

### `nextstat.churn_ingest(times, events, *, groups=None, treated=None, covariates=[], covariate_names=[], observation_end=None) -> dict`

Validate and ingest raw customer arrays into a clean churn dataset. Drops rows with invalid/missing times, applies observation-end censoring cap.

**Returns** a dict with keys: `n`, `n_events`, `times`, `events`, `groups`, `treated`, `covariates`, `covariate_names`, `n_dropped`, `warnings`.

## CLI parity

The CLI mirrors the core workflows for HEP (fit/hypotest/scan/limits), time series, survival, and churn analysis.
See `docs/references/cli.md`.
