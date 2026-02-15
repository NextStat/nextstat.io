---
title: "Benchmark Suite: Pharma (PK / NLME)"
description: "Pharmacometrics benchmark suite for NextStat: FOCE/FOCEI estimation, correlated random effects, stepwise covariate modeling, VPC/GOF diagnostics — with correctness-gated protocols for regulated-industry validation."
status: active
last_updated: 2026-02-12
keywords:
  - NLME benchmark
  - pharmacometrics performance
  - PK model benchmark
  - population PK fitting
  - FOCE FOCEI
  - correlated random effects
  - stepwise covariate modeling
  - VPC GOF
  - NONMEM alternative
  - Monolix comparison
  - pharmaceutical software validation
  - NextStat pharma
---

# Pharma Benchmark Suite (PK / NLME)

This suite benchmarks NextStat's pharmacometrics stack:

- **Phase 1** — 1/2-compartment PK models, FOCE/FOCEI estimation, NONMEM dataset ingest
- **Phase 2** — Correlated random effects (full Ω), SCM, VPC/GOF diagnostics
- **Phase 3** — SAEM algorithm, PD models (Emax, sigmoid Emax, indirect response I–IV), adaptive ODE solvers (RK45 + ESDIRK stiff), IQ/OQ/PQ validation, 21 CFR Part 11 compliance

The goal is to publish **reproducible parameter recovery** and **fit cost** for realistic problem sizes, with explicit protocols and synthetic datasets.

## Implemented benchmarks

Four integration tests in `crates/ns-inference/tests/pharma_benchmark.rs`:

| Benchmark | Drug | Subjects | Obs/subj | Model | Omega | Status |
|-----------|------|----------|----------|-------|-------|--------|
| `benchmark_warfarin` | Warfarin | 32 | 10 | 1-cpt oral | Diagonal | ✅ Pass |
| `benchmark_theophylline` | Theophylline | 12 | 11 | 1-cpt oral | Diagonal | ✅ Pass |
| `benchmark_phenobarbital` | Phenobarbital | 40 | 8 | 1-cpt oral (fast Ka) | Diagonal | ✅ Pass |
| `benchmark_warfarin_correlated_omega` | Warfarin | 40 | 10 | 1-cpt oral | Full (CL–V corr=0.6) | ✅ Pass |

### Run locally

```bash
cargo test -p ns-inference --test pharma_benchmark -- --nocapture
```

### True parameters (synthetic datasets)

**Warfarin** — CL=0.133 L/h, V=8.0 L, Ka=0.8 h⁻¹, ω²=[0.30, 0.25, 0.30], σ=0.5 mg/L

**Theophylline** — CL=0.04 L/h, V=0.5 L, Ka=1.5 h⁻¹, ω²=[0.25, 0.20, 0.30], σ=0.3 mg/L

**Phenobarbital** — CL=0.005 L/h, V=0.7 L, Ka=2.0 h⁻¹, ω²=[0.20, 0.15, 0.25], σ=0.2 mg/L

**Warfarin (correlated)** — CL=0.133, V=8.0, Ka=0.8, corr(CL,V)=0.6, ω=[0.30, 0.25, 0.30], σ=0.5

### Recovery tolerance

All benchmarks validate that fitted θ is within **3× true ω** of the true value per parameter. This is conservative for population-level recovery with realistic between-subject variability.

## Estimation protocol

- **Estimator**: `FoceEstimator::focei()` (FOCE with interaction)
- **Inner optimizer**: Damped Newton-Raphson (per-subject ETA optimization)
- **Outer loop**: EM-like alternation with empirical Ω update (ridge-regularized)
- **Convergence**: ΔOFV < 1e-4, max 100 outer iterations, max 20 inner iterations
- **Error model**: Additive (`ErrorModel::Additive(σ)`)
- **Initialization**: True population parameters (best-case scenario for recovery)

## Diagnostics (Phase 2)

Each benchmark runs GOF and VPC diagnostics after fitting:

### GOF (Goodness-of-Fit)

`gof_1cpt_oral()` computes per-observation:
- **PRED** — population prediction (η = 0)
- **IPRED** — individual prediction (at conditional mode η̂)
- **IWRES** — individual weighted residual
- **CWRES** — conditional weighted residual

Validation gate: IWRES mean within ±1.0 of zero.

### VPC (Visual Predictive Check)

`vpc_1cpt_oral()` with 200 simulation replicates:
- Quantiles: 5th, 50th, 95th percentile
- 10 time bins
- 90% prediction intervals on simulated quantiles

Validation gate: observed median within simulated PI.

## Correlated random effects

`OmegaMatrix` stores the full Ω via Cholesky factor L (Ω = L·Lᵀ). Constructors:
- `from_diagonal()` — backward-compatible independent RE
- `from_correlation()` — SDs + correlation matrix
- `from_covariance()` — full covariance matrix

The warfarin correlated benchmark validates recovery of a CL–V correlation of 0.6.

## Stepwise Covariate Modeling (SCM)

`ScmEstimator` with forward selection + backward elimination:
- Forward: add covariates with ΔOFV > 3.84 (α = 0.05)
- Backward: remove covariates with ΔOFV < 6.63 (α = 0.01)
- Relationships: Power, Proportional, Exponential
- 7/7 unit tests pass (selection + rejection + input validation)

## Artifact schema (v2.0.0)

All results export to `NlmeArtifact` (JSON authoritative):

```json
{
  "schema_version": "2.0.0",
  "model_label": "warfarin_1cpt_oral",
  "fixed_effects": { "names": ["CL","V","Ka"], "estimates": [...] },
  "random_effects": { "sds": [...], "covariance": [[...]], "correlation": [[...]] },
  "ofv": 123.456,
  "converged": true,
  "run_bundle": {
    "nextstat_version": "0.9.0",
    "git_rev": "abc1234",
    "seeds": { "foce": 42, "vpc": 123 },
    "datasets": [{ "label": "warfarin_32subj", "n_subjects": 32, "n_obs": 320 }]
  }
}
```

CSV exports: `fixed_effects_csv()`, `random_effects_csv()`, `gof_csv()`, `vpc_csv()`, `scm_csv()`.

## Provenance (RunBundle)

Every benchmark run captures:
- NextStat version + git rev + dirty flag
- Rust toolchain + target triple + OS
- Random seeds (keyed by purpose)
- Dataset provenance (label, hash, n_subjects, n_obs, source)
- Reference tool versions (for parity comparison)

## Test counts

| Module | Tests | Status |
|--------|-------|--------|
| `foce` (OmegaMatrix + FOCE) | 13 | ✅ |
| `scm` (SCM) | 7 | ✅ |
| `vpc` (VPC + GOF) | 7 | ✅ |
| `artifacts` (schema + export + bundle) | 15 | ✅ |
| `pharma_benchmark` (integration) | 4 | ✅ |
| `saem` (SAEM algorithm) | 5 | ✅ |
| `pd` (Emax, sigmoid Emax, IDR) | 19 | ✅ |
| `ode_adaptive` (RK45 + ESDIRK) | 12 | ✅ |
| `phase3_benchmark` (integration) | 7 | ✅ |
| **Total** | **89** | ✅ |

## Phase 3: SAEM, PD models, ODE solvers

### SAEM (Stochastic Approximation EM)

`SaemEstimator` — Monolix-class algorithm with:
- Metropolis-Hastings E-step with adaptive proposal variance
- Stochastic approximation with burn-in (γ=1) and estimation (γ=1/k) phases
- Closed-form M-step for θ and Ω (re-centering + empirical covariance)
- `SaemDiagnostics`: acceptance rates, OFV trace

Integration tests in `crates/ns-inference/tests/phase3_benchmark.rs`:

| Benchmark | Description | Status |
|-----------|-------------|--------|
| `benchmark_saem_warfarin` | SAEM fit on 32-subject Warfarin, parameter recovery | ✅ Pass |
| `benchmark_saem_vs_foce_parity` | θ̂ agreement between SAEM and FOCE | ✅ Pass |

### PD models

| Model | Formula | Tests |
|-------|---------|-------|
| `EmaxModel` | E = E0 + Emax·C/(EC50 + C) | 6 |
| `SigmoidEmaxModel` | E = E0 + Emax·C^γ/(EC50^γ + C^γ) | 4 |
| `IndirectResponseModel` Types I–IV | ODE-based production/loss modulation | 9 |

Integration benchmarks:
- `benchmark_emax_dose_response` — monotonicity, E(0)=E0, E(EC50)=Emax/2
- `benchmark_sigmoid_emax_hill_coefficients` — steepness vs γ
- `benchmark_indirect_response_all_types` — direction check for all 4 IDR types

### Adaptive ODE solvers

| Solver | Method | Use case | Tests |
|--------|--------|----------|-------|
| `rk45()` | Dormand–Prince 4(5) | Non-stiff PK/PD, Michaelis–Menten | 8 |
| `esdirk4()` | L-stable SDIRK2 | Stiff systems (transit ktr > 100) | 4 |

Integration benchmarks:
- `benchmark_ode_transit_chain_rk45_vs_esdirk` — cross-solver parity (7 cpts)
- `benchmark_ode_stiff_transit_high_ktr` — ktr=100, 10 compartments

### Validation & compliance

- **IQ/OQ/PQ protocol** — `docs/validation/iq-oq-pq-protocol.md` (NS-VAL-001)
- **21 CFR Part 11** — `docs/validation/21cfr-part11-compliance.md` (NS-REG-001)

### R package (ns-r)

R wrappers for pharma functions in `bindings/ns-r/R/pharma.R`:
- `ns_foce()`, `ns_saem()` — NLME estimation
- `ns_emax()`, `ns_sigmoid_emax()`, `ns_idr()` — PD models
- `ns_vpc()`, `ns_gof()`, `ns_scm()` — diagnostics

## Baselines (planned)

- **nlmixr2** (R) — FOCE/FOCEI parity on same synthetic datasets
- **Monolix** (SAEM) — SAEM parity on Warfarin/Theophylline
- **Torsten (Stan)** — for selected PK/NLME workflows

## Scaling axes

- Number of subjects (12–40 currently; plan to test 100–1000)
- Observations per subject (8–11 currently)
- Random-effects dimension (3 diagonal, 3 correlated)
- Error model type (additive; proportional/combined supported)
- PD model complexity (direct effect → ODE-based IDR)

## Related reading

- [Phase 2 Tutorial Pack](/docs/pharmacometrics/phase2-tutorial) — SCM + correlated RE + VPC interpretation
- [PK Baseline Tutorial](/docs/tutorials/phase-13-pk) — 1-compartment PK model walkthrough
- [NLME Baseline Tutorial](/docs/tutorials/phase-13-nlme) — population PK with random effects
- [Public Benchmarks Specification](/docs/public-benchmarks) — canonical spec
- [Pharma Benchmarks blog post](/blog/pharma-benchmarks-pk-nlme) — methodology
