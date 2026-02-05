# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Release pipeline hardening: test gate, version validation, multi-arch wheels, sdist,
  CHANGELOG-based release notes, optional PyPI publish.
- Structured logging with `tracing` and `--log-level` CLI flag.

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

#### Phase 3.2 — Bayesian Sampling
- `ns-inference`: No-U-Turn Sampler (NUTS) with dual averaging.
- `ns-inference`: HMC diagnostics (divergences, tree depth, step size, ESS).
- `ns-py`: `sample_nuts()` Python binding returning ArviZ-compatible dict.

#### Phase 3.4 (partial) — Visualization Artifacts
- `ns-viz`: `ProfileCurveArtifact` (q_mu vs mu, with sigma lines and best-fit).
- `ns-viz`: `ClsCurveArtifact` (observed + expected CLs with Brazil band).
- `ns-cli`: `viz profile` and `viz cls` subcommands.
- `ns-py`: `viz_profile_curve()` and `viz_cls_curve()` Python helpers.
