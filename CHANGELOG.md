# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### GPU Acceleration

- **CUDA (NVIDIA, f64)** — fused NLL+gradient kernel covering all 7 HistFactory modifier types in a single launch. Lockstep batch optimizer fits thousands of toys in parallel. Dynamic loading via cudarc — binary works without CUDA installed.
  - `nextstat.fit(model, device="cuda")`, `--gpu cuda` CLI flag
- **Metal (Apple Silicon, f32)** — same fused kernel in MSL. Zero-copy unified memory. NLL parity vs CPU f64: 1.27e-6 relative diff.
  - `nextstat.fit_toys_batch_gpu(model, ..., device="metal")`, `--gpu metal` CLI flag
- **Apple Accelerate** — vDSP/vForce vectorized NLL on macOS. <5% overhead vs naive summation.
- **CPU batch toys** — Rayon-parallel toy fitting with per-thread tape reuse, seed-based reproducibility.
- Reverse-mode tape optimization: faster gradient computation with reduced memory allocation.

#### Differentiable Analysis (PyTorch)

- Zero-copy CUDA kernel reads signal histogram from a PyTorch tensor and writes dNLL/dsignal directly into the grad buffer — no device-host roundtrip.
- `DifferentiableSession`: NLL + signal gradient at fixed nuisance parameters.
- `ProfiledDifferentiableSession`: profiled test statistics with envelope-theorem gradients — enables NN → signal histogram → profiled CLs → loss.
- `nextstat.torch` Python module: `NextStatNLLFunction`, `NextStatProfiledQ0Function` (autograd), `NextStatLayer(nn.Module)`.
- `profiled_zmu_loss()` — Zμ loss wrapper (sqrt(qμ) with numerical stability) for signal-strength optimization.
- `SignificanceLoss(model)` — ML-friendly class wrapping profiled −Z₀. Init once, call per-batch: `loss_fn(signal_hist).backward()`.
- `SoftHistogram` — differentiable binning (Gaussian KDE / sigmoid): NN classifier scores → soft histogram → `SignificanceLoss`.
- `batch_profiled_q0_loss()` — profiled q₀ for a batch of signal histograms (ensemble training).
- `signal_jacobian()`, `signal_jacobian_numpy()` — direct ∂q₀/∂signal without autograd for SciPy bridge and fast pruning.
- `as_tensor()` — DLPack array-API bridge: JAX, CuPy, Arrow, NumPy → `torch.Tensor`.
- `nextstat.mlops` — fit metrics extraction for W&B / MLflow / Neptune: `metrics_dict(result)`, `significance_metrics(z0)`, `StepTimer`.
- `nextstat.interpret` — systematic-impact ranking as Feature Importance: `rank_impact(model)`, `rank_impact_df()`, `plot_rank_impact()`.
- **`nextstat.tools`** — LLM tool definitions (OpenAI function calling, LangChain, MCP) for 7 operations: fit, hypotest, upper_limit, ranking, significance, scan, workspace_audit. `get_toolkit()` returns JSON Schema; `execute_tool(name, args)` bridges agent calls to NextStat.
- **`nextstat.distill`** — surrogate training dataset generator. `generate_dataset(model, n_samples=100k, method="sobol")` produces `(params, NLL, gradient)` tuples. Export to PyTorch `TensorDataset`, `.npz`, or Parquet. Built-in `train_mlp_surrogate()` with Sobolev loss.
- Fit convergence check: returns error if GPU profile fit fails to converge.

#### Gymnasium RL Environment

- `nextstat.gym` — optional Gymnasium/Gym wrapper treating a HistFactory workspace as an RL/DOE environment.
- Propose updates to a sample's nominal yields, receive a NextStat metric as reward (NLL, q₀, Z₀, qμ, Zμ).
- `make_histfactory_env()` factory with configurable `reward_metric`, `action_mode` (additive/logmul), `action_scale`.
- Compatible with `gymnasium` (preferred) and legacy `gym`.

#### Deterministic Validation

- **EvalMode** — process-wide flag: **Parity** (Kahan summation, single-threaded, bit-exact) vs **Fast** (default, SIMD/GPU, multi-threaded).
- CLI: `--parity` · Python: `nextstat.set_eval_mode("parity")`.
- 7-tier tolerance contract vs pyhf (per-bin ~1e-14 worst case).

#### Native ROOT I/O

- **TTree reader** — mmap file access, native binary deserialization, basket decompression (zlib/LZ4/ZSTD) with rayon-parallel extraction. 9 leaf types + jagged branches.
- **Expression engine** — bytecode-compiled, vectorized. Full grammar: arithmetic, comparisons, boolean logic, ternary, builtins. Dynamic jagged indexing (`jet_pt[idx]`) follows ROOT/TTreeFormula convention. Python wrapper: `nextstat.analysis.expr_eval`.
- **Histogram filler** — single-pass with selection cuts, weights, variable binning.
- **Unsplit vector branch decoding** — best-effort decoding for `std::vector<T>` branches without offset tables.
- **~8.5× faster** than uproot+numpy on the full pipeline.

#### Ntuple-to-Workspace Pipeline

- `NtupleWorkspaceBuilder`: ROOT ntuples → HistFactory `Workspace` via fluent Rust API.
- Per-sample modifiers: NormFactor, NormSys, WeightSys, TreeSys, HistoSys, StatError.
- Produces the same `Workspace` struct as the pyhf JSON path — no ROOT C++ dependency.

#### TRExFitter Interop

- `nextstat import trex-config` — import TRExFitter `.config` into pyhf JSON workspace.
- `nextstat build-hists` — run NTUP pipeline, write `workspace.json`.
- **HIST mode** — read pre-built ROOT histograms (`ReadFrom: HIST`) alongside NTUP.
- **Analysis Spec v0** (YAML + JSON Schema) — `nextstat run <spec.yaml>` orchestrates import/fit/scan/report.
- Jagged column support and TRExFitter-style expression compatibility.

#### Systematics Preprocessing

- **Smoothing**: 353QH,twice algorithm (ROOT `TH1::Smooth` equivalent) + Gaussian kernel.
- **Pruning**: shape, norm, and overall pruning with audit trail.
- **`nextstat preprocess`** CLI with declarative YAML config and content-hash caching.
- Recommended order: hygiene → symmetrize → smooth → prune.

#### HistFactory Enhancements

- **HS3 v0.2 ingestion** — load HS3 JSON workspaces (ROOT 6.37+) natively. Auto-detects format (pyhf vs HS3) at load time.
- **HS3 roundtrip export** — export `HistFactoryModel` back to HS3 JSON with bestfit parameter points.
- Python: `HistFactoryModel.from_workspace()` (auto-detect), `HistFactoryModel.from_hs3(json_str)`. CLI: auto-detection in `nextstat fit`, `nextstat scan`.
- HS3 inputs use ROOT HistFactory defaults (NormSys Code1, HistoSys Code0). For pyhf JSON inputs, NextStat defaults to smooth interpolation (NormSys Code4, HistoSys Code4p); use `--interp-defaults pyhf` (CLI) or `from_workspace_with_settings(Code1, Code0)` (Rust) for strict pyhf defaults.
- HEPData patchset support: `nextstat import patchset`, Python `nextstat.apply_patchset()`.
- **Arrow / Polars ingestion** — `nextstat.from_arrow(table)` creates a HistFactoryModel from PyArrow Table, RecordBatch, or any Arrow-compatible source (Polars, DuckDB, Spark). `nextstat.from_parquet(path)` reads Parquet directly.
- **Arrow export** — `nextstat.to_arrow(model, what="yields"|"params")` exports expected yields or parameter metadata as a PyArrow Table. Uses Arrow IPC bridge (zero pyo3 version conflicts).
- **ConstraintTerm semantics** — LogNormal alpha-transform (`normsys_alpha_effective`), Gamma constraint for ShapeSys, Uniform and NoConstraint handling. Parsed from `<ConstraintTerm>` metadata in HistFactory XML.

#### Report System

- `nextstat report` — generates distributions, pulls, correlations, yields (.json/.csv/.tex), and uncertainty ranking from a workspace.
- Python rendering: multi-page PDF + per-plot SVGs via matplotlib.
- `--blind` flag masks observed data for unblinded regions.
- `--deterministic` for stable JSON key ordering.
- **`nextstat validation-report`** — unified validation artifact combining Apex2 results with workspace fingerprints. Outputs `validation_report.json` (schema `validation_report_v1`) with dataset SHA-256, model spec, environment, and per-suite pass/fail summary. Optional `--pdf` renders a 4-page audit-ready PDF via matplotlib.

#### Survival Analysis

- Parametric models: Exponential, Weibull, LogNormal AFT (with right-censoring).
- Cox Proportional Hazards: Efron/Breslow ties, robust sandwich SE, Schoenfeld residuals.
- Python: `nextstat.survival.{exponential,weibull,lognormal_aft,cox_ph}.fit(...)`.
- CLI: `nextstat survival fit`, `nextstat survival predict`.

#### Linear Mixed Models

- Analytic marginal likelihood (random intercept, random intercept + slope).
- Laplace approximation for approximate posteriors.
- Python: `nextstat.LmmMarginalModel(...)`.

#### Ordinal Models

- Ordered logit/probit with stable cutpoint parameterization.
- Python: `nextstat.ordinal.ordered_logit.fit(...)`, `nextstat.ordinal.ordered_probit.fit(...)`.

#### Econometrics & Causal Inference

- **Panel FE** with 1-way cluster SE.
- **DiD TWFE** + event-study helpers.
- **IV / 2SLS** with weak-IV diagnostics (first-stage F, partial R²).
- **AIPW** for ATE/ATT + E-value helper. Propensity scores, IPW weights, overlap diagnostics.

#### Pharmacometrics

- RK4 integrator for linear ODE systems.
- One-compartment oral PK model with LLOQ censoring.
- NLME extension with per-subject random effects.

#### Applied Statistics API

- Formula parsing + deterministic design matrices (`nextstat.formula`).
- `from_formula` wrappers for all GLM and hierarchical builders.
- Wald summaries + robust covariance (HC0-HC3, 1-way cluster).
- scikit-learn adapters: `NextStatLinearRegression`, `NextStatLogisticRegression`, `NextStatPoissonRegressor`.
- Missing-data policies: `drop_rows`, `impute_mean`.

#### WASM Playground

- Browser-based inference via `wasm-bindgen`: `fit_json()`, `hypotest_json()`, `upper_limit_json()`.
- Drag-and-drop `workspace.json` → asymptotic CLs Brazil bands. No Python, no server.

#### Visualization

- `plot_cls_curve()`, `plot_brazil_limits()`, `plot_profile_curve()`.
- `nextstat viz distributions`, `viz pulls`, `viz corr`, `viz ranking` subcommands.
- Kalman: `plot_kalman_states()`, `plot_forecast_bands()`.

#### CLI & Infrastructure

- Structured logging (`--log-level`), reproducible run bundles (`--bundle`).
- `fit()` supports `init_pars=` for warm-start MLE.
- CI: pyhf parity gate on push/PR, TREx baseline refresh (nightly), HEPData workspace tests.
- Apex2 validation: NLL parity, bias/pulls regression, SBC calibration, NUTS quality gates.
- **`nextstat-server`** — self-hosted REST API for shared GPU inference. `POST /v1/fit` (workspace → FitResult), `POST /v1/ranking` (NP impacts), `GET /v1/health`. `--gpu cuda|metal`, `--port`, `--host`, `--threads`.
- **`nextstat.remote`** — pure-Python thin client (httpx). `client = nextstat.remote.connect("http://gpu-server:3742")`, then `client.fit(workspace)`, `client.ranking(workspace)`, `client.health()`. Typed dataclass results.
- **Batch API** — `POST /v1/batch/fit` fits up to 100 workspaces in one request; `POST /v1/batch/toys` runs GPU-accelerated toy fitting (CUDA/Metal/CPU). `client.batch_fit(workspaces)`, `client.batch_toys(workspace, n_toys=1000)`.
- **Model cache** — `POST /v1/models` uploads a workspace and returns a `model_id` (SHA-256); subsequent `/v1/fit` and `/v1/ranking` calls accept `model_id=` to skip re-parsing. `GET /v1/models`, `DELETE /v1/models/:id`. LRU eviction (64 models).
- **Docker & Helm** — multi-stage `Dockerfile` for CPU and CUDA builds, Helm chart with health probes, GPU resource requests, configurable replicas.

### Fixed

- CUDA batch toys (`--gpu cuda`) crash when some toys converge before others.
- GPU profiled session (`ProfiledDifferentiableSession`) convergence failure near parameter bounds.
- Optimizer early-stop with negative NLL (`target_cost(0.0)` removed).
- `kalman_simulate()`: `init="sample|mean"` and `x0=...` support.
- StatError: incorrect `sqrt(sumw2)` propagation with zero nominal counts.
- Metal GPU: scratch buffer reuse (~40% less allocation overhead).
- HistFactory XML: strip `<!DOCTYPE>` declarations before parsing.
- CUDA/Metal signal gradient race condition: incorrect accumulation when multiple samples contribute to the same bin.
- 10 missing Python re-exports in `__init__.py`: `has_metal`, `read_root_histogram`, `workspace_audit`, `cls_curve`, `profile_curve`, `kalman_filter/smooth/em/forecast/simulate`.

---

## [0.1.0] — 2026-02-05

Initial public release.

### Core Engine

- HistFactory workspace data model with full pyhf JSON compatibility.
- Poisson NLL with all modifier types (histosys, normsys, shapesys, staterror, lumi) + Barlow-Beeston.
- SIMD-accelerated NLL via `wide::f64x4`.
- Automatic differentiation: forward-mode (dual numbers) and reverse-mode (tape AD).

### Frequentist Inference

- MLE via L-BFGS-B with Hessian-based uncertainties.
- Asymptotic CLs hypothesis testing (q-tilde test statistic).
- Profile likelihood scans, CLs upper limits (bisection + linear scan), Brazil bands.
- Batch MLE, toy studies, nuisance parameter ranking.

### Bayesian Sampling

- No-U-Turn Sampler (NUTS) with dual averaging.
- HMC diagnostics: divergences, tree depth, step size, E-BFMI.
- Rank-normalized folded R-hat + improved ESS (Geyer IMS + variogram).
- Python: `sample()` returning ArviZ-compatible dict.

### Regression & GLM

- Linear, logistic, Poisson, negative binomial regression.
- Ridge regression (MAP/L2), separation detection, exposure/offset support.
- Cross-validation (`kfold_indices`, `cross_val_score`) and metrics (RMSE, log-loss, Poisson deviance).

### Hierarchical Models

- Random intercepts/slopes, correlated effects (LKJ + Cholesky), non-centered parameterization.
- Posterior Predictive Checks.

### Time Series

- Linear-Gaussian Kalman filter + RTS smoother.
- EM parameter estimation, multi-step-ahead forecasting with prediction intervals.
- Local-level, local-trend, AR(1) builders. Missing observation handling.

### Probability Distributions

- Normal, StudentT, Bernoulli, Binomial, Poisson, NegativeBinomial, Gamma, Exponential, Weibull, Beta.
- Bijector/transform layer: Identity, Exp, Softplus, Sigmoid, Affine.

### Visualization

- Profile likelihood curves and CLs Brazil band plots.
- CLI: `viz profile`, `viz cls`. Python: `viz_profile_curve()`, `viz_cls_curve()`.

### Python Bindings & CLI

- `nextstat` Python package (PyO3/maturin) with `Model`, `FitResult` classes.
- `nextstat` CLI: `fit`, `hypotest`, `upper-limit`, `scan`, `version`.
- CI workflows + GitHub release pipeline (multi-arch wheels + CLI binary).

### Validation (Apex2)

- Master report aggregator with NLL parity, GLM benchmarks, bias/pulls regression, SBC calibration, NUTS quality gates.
- Nightly slow CI workflow.
