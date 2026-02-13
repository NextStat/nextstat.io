---
title: "PAGE 2026 Abstract Draft — NextStat Pharmacometrics Benchmark Poster"
status: draft
last_updated: 2026-02-12
target: PAGE 2026 (Population Approach Group in Europe)
category: poster
---

# NextStat: A Compiled Pharmacometrics Engine with FOCE/FOCEI, Correlated Random Effects, and Reproducible Benchmark Infrastructure

## Authors

NextStat Team

## Objectives

To introduce NextStat, an open-source compiled pharmacometrics engine implemented in Rust, and present benchmark results demonstrating parameter recovery on classic PK datasets using FOCE/FOCEI estimation with correlated random effects, stepwise covariate modeling, and integrated diagnostic infrastructure.

## Methods

NextStat implements a native FOCE/FOCEI estimator with:

1. **Two-level optimization** — per-subject ETA optimization via damped Newton-Raphson (inner) and EM-like population parameter updates with ridge-regularized empirical Ω (outer).

2. **Full Ω parameterization** — Cholesky-decomposed variance–covariance matrix (Ω = L·Lᵀ), guaranteeing positive-definiteness by construction. Supports diagonal, correlation-based, and full covariance constructors.

3. **Stepwise Covariate Modeling (SCM)** — automated forward selection (α = 0.05, ΔOFV > 3.84) and backward elimination (α = 0.01, ΔOFV > 6.63) with power, proportional, and exponential covariate–parameter relationships.

4. **VPC and GOF diagnostics** — simulation-based Visual Predictive Check (N = 200 replicates, configurable quantiles and prediction intervals) and standard goodness-of-fit metrics (PRED, IPRED, IWRES, CWRES).

Benchmarks used synthetic datasets mimicking three classic pharmacometric compounds with known true parameters:

| Dataset | Drug | Subjects | Obs/subj | True CL (L/h) | True V (L) | True Ka (h⁻¹) |
|---------|------|----------|----------|----------------|------------|----------------|
| A | Warfarin | 32 | 10 | 0.133 | 8.0 | 0.8 |
| B | Theophylline | 12 | 11 | 0.04 | 0.5 | 1.5 |
| C | Phenobarbital | 40 | 8 | 0.005 | 0.7 | 2.0 |
| D | Warfarin (corr.) | 40 | 10 | 0.133 | 8.0 | 0.8 |

Dataset D includes a CL–V correlation of 0.6 in the true Ω matrix to validate correlated random effects recovery.

All datasets used additive error models with known σ. Data generation used log-normal inter-individual variability on CL, V, and Ka.

**Estimation protocol**: FOCEI with ΔOFV convergence tolerance 1e-4, max 100 outer / 20 inner iterations. Initialization at true population parameters.

**Recovery criterion**: fitted θ within 3× true ω of the true value per parameter.

## Results

All four benchmarks converged and recovered true parameters within tolerance:

| Dataset | Converged | OFV finite | θ recovery | GOF (IWRES ~0) | VPC (median in PI) |
|---------|-----------|------------|------------|----------------|-------------------|
| Warfarin | ✓ | ✓ | ✓ (3/3 params) | ✓ | ✓ |
| Theophylline | ✓ | ✓ | ✓ (3/3 params) | ✓ | ✓ |
| Phenobarbital | ✓ | ✓ | ✓ (3/3 params) | ✓ | ✓ |
| Warfarin (corr.) | ✓ | ✓ | ✓ (3/3 params) | — | — |

The correlated Warfarin benchmark recovered the CL–V correlation structure from the full Ω matrix.

**Infrastructure**: All results are exported as structured JSON artifacts (schema v2.0.0) with provenance bundles capturing git revision, random seeds, dataset hashes, and environment metadata. CSV exports are available for fixed effects, random effects, GOF records, VPC bins, and SCM trace.

**Test coverage**: 46 automated tests (13 FOCE + 7 SCM + 7 VPC + 15 artifacts + 4 integration benchmarks), all passing.

## Conclusions

NextStat demonstrates a viable compiled alternative to established pharmacometrics tools for FOCE/FOCEI estimation with:

- Full Ω covariance modeling via Cholesky parameterization
- Automated covariate selection with auditable traces
- Integrated VPC/GOF diagnostics
- Reproducible benchmark infrastructure with structured artifact export

The compiled Rust implementation offers potential for significant performance advantages as problem sizes scale, while the artifact schema and provenance bundles support regulated-environment reproducibility requirements.

**Source code**: https://github.com/nextstat-io/nextstat (Apache-2.0 + commercial dual license)

**Benchmarks**: `cargo test -p ns-inference --test pharma_benchmark`

---

## Submission Checklist

- [ ] All numbers traceable to committed artifacts/test output
- [ ] Benchmark tests pass on CI (`cargo test -p ns-inference --test pharma_benchmark`)
- [ ] Unit tests pass (`cargo test -p ns-inference --lib`)
- [ ] Artifact JSON schema version matches abstract (v2.0.0)
- [ ] RunBundle provenance captures git rev at submission time
- [ ] Figures/tables generated from benchmark artifacts (not hand-crafted)
- [ ] Repro bundle: `pharma_benchmark.rs` + `artifacts.rs` + `RunBundle::auto_detect()`
- [ ] Abstract word count within PAGE limits
- [ ] Co-author approvals

## Repro Bundle Reference

| Artifact | Path |
|----------|------|
| Benchmark tests | `crates/ns-inference/tests/pharma_benchmark.rs` |
| FOCE estimator | `crates/ns-inference/src/foce.rs` |
| SCM module | `crates/ns-inference/src/scm.rs` |
| VPC/GOF module | `crates/ns-inference/src/vpc.rs` |
| Artifact schema | `crates/ns-inference/src/artifacts.rs` |
| Tutorial pack | `docs/pharmacometrics/phase2-tutorial.md` |
| Suite runbook | `docs/benchmarks/suites/pharma.md` |
| Blog draft | `docs/blog/pharma-benchmarks-pk-nlme.md` |
