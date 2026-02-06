# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 9 — Pharma & Social Sciences Domain Packs

**Pack A: Survival Analysis**
- Parametric survival models with right-censoring (`ns-inference::survival`):
  Exponential (`log_rate`), Weibull (`log_k`, `log_lambda`), LogNormal AFT (`mu`, `log_sigma`).
- Cox Proportional Hazards model with Efron/Breslow ties handling (`CoxPhModel`).
- Python surface: `nextstat.survival.exponential()`, `.weibull()`, `.lognormal_aft()`, `.cox_ph()`.
- All survival models integrate with `nextstat.fit()` and `nextstat.sample()`.

**Pack B: Longitudinal / Mixed-Effects**
- Linear Mixed Model with analytic marginal likelihood (`LmmMarginalModel`):
  random intercept, or random intercept + one random slope (diagonal RE covariance).
- Laplace approximation utilities for approximate posteriors (`ns-inference::laplace`).
- Python surface: `nextstat.LmmMarginalModel(...)` with `fit()` / `sample()` integration.
- LMM parameter recovery smoke tests and tutorial (`docs/tutorials/phase-9-lmm.md`).

#### Infrastructure
- Release pipeline hardening: test gate, version validation, multi-arch wheels, sdist,
  CHANGELOG-based release notes, optional PyPI publish.
- Structured logging with `tracing` and `--log-level` CLI flag.
- `ns-cli`: reproducible run bundle (`--bundle` flag).
- Apex2 baseline recorder (`tests/record_baseline.py`) with full environment fingerprint
  (hostname, Python/pyhf/nextstat/numpy versions, git commit, CPU, platform).
- Apex2 baseline comparison (`tests/compare_with_latest_baseline.py`) with per-type manifests
  and `--require-same-host` strict mode.
- NUTS quality report (`tests/apex2_nuts_quality_report.py`) integrated into master report.
- Bias/pulls report hardened: per-parameter output, `--params poi|all`, `--min-used-abs/frac`,
  try/except around toy-loop fits, `skipped` status on insufficient valid toys.
- Slow test ergonomics: `sbc` pytest marker, `NS_RUN_SBC=1` gate, reduced default toy counts.

#### Visualization
- `plot_cls_curve()` now respects `nsigma_order` + `cls_exp` dynamically.
- `plot_brazil_limits()` for observed/expected upper-limit Brazil band.
- `plot_profile_curve()` supports `y="q_mu"` (default) and `y="twice_delta_nll"`.

### Fixed
- Optimizer early-stop bug: removed `target_cost(0.0)` that broke models with NLL < 0.
- `kalman_simulate()` now supports `init="sample|mean"` and `x0=...` for custom start.

## [0.1.0] - 2026-02-05

### Added

#### Phase 1 — Core Engine
- `ns-core`: workspace data model (Channel, Sample, Parameter, Modifier, Measurement).
- `ns-core`: Poisson + modifier evaluation (histosys, normsys, shapesys, staterror, lumi).
- `ns-core`: negative log-likelihood (NLL) computation with constraint terms.
- `ns-translate`: pyhf JSON workspace parser with full HistFactory support.
- `ns-compute`: SIMD-accelerated Poisson NLL via `wide::f64x4`.
- `ns-py`: Python bindings (`nextstat` package) with `Model`, `FitResult` classes.
- `ns-cli`: `nextstat` CLI with `fit`, `hypotest`, `upper-limit`, `scan`, `version` commands.
- CI workflows for Rust (test + clippy) and Python (maturin build + pytest).
- GitHub release workflow with wheel and CLI binary artifacts.

#### Phase 2A — SIMD & PreparedModel
- `PreparedModel` with cached observed data, ln-factorials, and observation masks.
- SIMD Poisson NLL path (f64x4) with lane-by-lane scalar `ln()` for accuracy.
- `fit_minimum()` extracted for fast repeated minimizations (profile, toys).

#### Phase 2B — Automatic Differentiation
- `ns-ad`: dual-number forward-mode AD (`Dual<f64>`).
- `ns-ad`: reverse-mode tape AD (`Tape`, `TapeVar`).
- Generic `nll_generic<T: Scalar>` path supporting both AD backends.

#### Phase 3.1 — Frequentist Inference
- `ns-inference`: MLE fitting via `argmin` (L-BFGS-B with Hessian uncertainties).
- `ns-inference`: asymptotic CLs hypothesis testing (q-tilde test statistic).
- `ns-inference`: profile likelihood scan over POI values.
- `ns-inference`: CLs upper limits via bisection and linear scan.
- `ns-inference`: expected CLs band (Brazil band) computation.
- `ns-inference`: batch MLE fitting (`fit_batch`), toy studies (`fit_toys`), NP ranking.
- `ns-py`: `mle.fit_batch()`, `mle.fit_toys()`, `mle.ranking()` Python bindings.

#### Phase 3.2 — Bayesian Sampling
- `ns-inference`: No-U-Turn Sampler (NUTS) with dual averaging.
- `ns-inference`: HMC diagnostics (divergences, tree depth, step size).
- Rank-normalized folded R-hat + improved ESS (Geyer IMS + variogram).
- E-BFMI energy diagnostic.
- Overdispersed chain initialization option.
- Non-slow sampling quality summary.
- `ns-py`: `sample()` Python binding returning ArviZ-compatible dict.

#### Phase 3.4 — Visualization Artifacts
- `ns-viz`: `ProfileCurveArtifact` (q_mu vs mu, with sigma lines and best-fit).
- `ns-viz`: `ClsCurveArtifact` (observed + expected CLs with Brazil band).
- `ns-cli`: `viz profile` and `viz cls` subcommands.
- `ns-py`: `viz_profile_curve()` and `viz_cls_curve()` Python helpers.

#### Phase 5 — Universal Model API
- `ns-core`: `LogDensityModel` trait for generic inference across all model types.
- `ns-prob` crate: probability distribution library (Normal, StudentT, Bernoulli, Binomial,
  Poisson, NegativeBinomial, Gamma, Exponential, Weibull, Beta).
- `ns-prob`: bijector/transform layer (Identity, Exp, Softplus, Sigmoid, Affine).
- `ns-inference`: `ComposedGlmModel` builder for composing GLM-style models.
- `ns-inference`: generic posterior, HMC, and NUTS over any `LogDensityModel`.
- `ns-inference`: parameter transforms with log-Jacobian for constrained parameters.
- `ns-py`: `GaussianMeanModel` for non-HEP Bayesian estimation.
- `ns-py`: `GlmSpec` data adapter (`nextstat.data`) for design matrices and responses.

#### Phase 6 — Regression & GLM Pack
- Linear, logistic, Poisson, and negative binomial regression models (`ns-inference::regression`).
- Ridge regression via Normal prior on coefficients (MAP/L2).
- Separation detection and stabilization warnings for logistic regression.
- Exposure/offset support for Poisson regression.
- Python GLM surface: `nextstat.glm.linear`, `.logistic`, `.poisson`, `.negbin`.
- Cross-validation utilities: `kfold_indices()`, `cross_val_score()` (`nextstat.glm.cv`).
- Metrics: RMSE, log-loss, Poisson deviance (`nextstat.glm.metrics`).
- Golden regression test fixtures with statsmodels parity (`tests/fixtures/regression/`).
- Criterion benchmarks: `regression_benchmark.rs`, `glm_fit_predict_benchmark.rs`.

#### Phase 7 — Hierarchical / Multilevel Models
- Random intercept support in `ComposedGlmModel` (Normal prior, partial pooling).
- Random slopes / varying coefficients.
- Non-centered parameterization for improved NUTS mixing.
- Correlated random effects via LKJ prior + Cholesky decomposition.
- Python surface: `nextstat.hier` with `linear_random_intercept()`, `linear_random_slope()`,
  `logistic_random_intercept()`, `logistic_correlated_intercept_slope()`, etc.
- Posterior Predictive Checks: `nextstat.ppc.ppc_glm_from_sample()` with replicate generation.
- Criterion benchmark: `hier_benchmark.rs`.

#### Phase 8 — Time Series & State Space Models
- Linear-Gaussian Kalman filter and RTS smoother (`ns-inference::timeseries`).
- EM parameter estimation for state-space models (Q, R, optionally F, H).
- Multi-step-ahead Kalman forecasting with prediction intervals.
- Standard model builders: local-level, local-trend.
- AR(1) parameter transform (softplus + bounds).
- Missing observation handling (skip update, correct likelihood).
- State simulation from model with controllable seed.
- Python surface: `nextstat.timeseries` with `kalman_filter()`, `kalman_smooth()`,
  `kalman_em()`, `kalman_fit()`, `kalman_forecast()`, `kalman_simulate()`.
- Visualization: `plot_kalman_states()`, `plot_forecast_bands()`.
- Criterion benchmarks: `kalman_benchmark.rs`, `em_benchmark.rs`.

#### Apex2 Validation System
- Master report aggregator (`tests/apex2_master_report.py`) with exit code policy.
- pyhf NLL/expected_data parity runner.
- P6 GLM benchmark runner with baseline comparison and slowdown detection.
- Bias/pulls regression runner (NextStat vs pyhf).
- SBC posterior calibration runner (NUTS).
- NUTS quality runner (divergence/R-hat/ESS/E-BFMI gates).
- ROOT/TRExFitter parity runner (optional).
- Nightly slow CI workflow (`.github/workflows/apex2-nightly-slow.yml`).
