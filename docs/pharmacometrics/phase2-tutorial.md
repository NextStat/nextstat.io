# Pharmacometrics Phase 2 Tutorial Pack

## Overview

This tutorial covers the Phase 2 Credibility features added to NextStat's
pharmacometrics engine:

1. **Correlated Random Effects** — full Ω matrix with Cholesky parameterization
2. **Stepwise Covariate Modeling (SCM)** — forward selection + backward elimination
3. **VPC & GOF Diagnostics** — Visual Predictive Check and Goodness-of-Fit

All examples use the 1-compartment oral PK model with FOCE/FOCEI estimation.

---

## 1  Correlated Random Effects

### 1.1  Background

In population PK, random effects (ETAs) capture inter-individual variability
in PK parameters. The standard assumption is:

    η_i ~ N(0, Ω)

where Ω is the variance–covariance matrix of random effects. Phase 1 used a
**diagonal Ω** (independent random effects). Phase 2 introduces a **full Ω**
that models correlations between random effects (e.g., CL–V correlation).

### 1.2  Cholesky Parameterization

Ω is stored as its Cholesky factor **L** (lower triangular), so Ω = L·Lᵀ
is always positive-definite by construction:

    Ω = L · Lᵀ

This avoids the need for constrained optimization on the Ω elements.

### 1.3  API

```rust
use ns_inference::{OmegaMatrix, FoceEstimator, ErrorModel};

// Diagonal (independent) — backward compatible
let omega = OmegaMatrix::from_diagonal(&[0.25, 0.20, 0.30]).unwrap();

// From correlation matrix (CL–V correlation = 0.6)
let corr = vec![
    vec![1.0, 0.6, 0.0],
    vec![0.6, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
];
let omega = OmegaMatrix::from_correlation(&[0.25, 0.20, 0.30], &corr).unwrap();

// From full covariance matrix
let cov = vec![
    vec![0.0625, 0.030, 0.0],
    vec![0.030,  0.040, 0.0],
    vec![0.0,    0.0,   0.09],
];
let omega = OmegaMatrix::from_covariance(&cov).unwrap();

// Fit with correlated omega
let estimator = FoceEstimator::focei();
let result = estimator.fit_1cpt_oral_correlated(
    &times, &y, &subject_idx, n_subjects,
    dose, bioav, ErrorModel::Additive(sigma),
    &[cl_init, v_init, ka_init],
    omega,
).unwrap();

// Inspect results
println!("Estimated SDs: {:?}", result.omega);
println!("Correlation matrix: {:?}", result.correlation);
println!("Full Ω: {:?}", result.omega_matrix.to_matrix());
```

### 1.4  Interpreting Results

The `FoceResult` now includes:

| Field | Description |
|-------|-------------|
| `omega` | Diagonal SDs (backward compatible) |
| `omega_matrix` | Full Ω as `OmegaMatrix` (Cholesky-stored) |
| `correlation` | Extracted correlation matrix |

A positive CL–V correlation (common in PK) means subjects with higher
clearance also tend to have higher volume of distribution.

### 1.5  Mathematical Details

The prior term in the FOCE inner objective changes from:

    Diagonal: 0.5 · Σ_k η_k² / ω_k²

to:

    Full: 0.5 · ηᵀ Ω⁻¹ η

The inverse quadratic form is computed efficiently via forward-substitution
on L (no explicit matrix inversion):

    L z = η  →  ηᵀ Ω⁻¹ η = |z|²

The Ω log-determinant for the OFV is:

    log|Ω| = 2 · Σ_i ln(L_ii)

---

## 2  Stepwise Covariate Modeling (SCM)

### 2.1  Background

Covariate modeling identifies patient characteristics (weight, age, renal
function) that explain part of the inter-individual variability in PK
parameters. SCM uses a two-stage procedure:

1. **Forward selection** — greedily add the most significant covariate
   relationship until no more pass the threshold (default: p < 0.05, ΔOFV > 3.84)
2. **Backward elimination** — remove the least significant covariate
   from the full forward model until all remaining are significant
   (default: p < 0.01, ΔOFV > 6.63)

### 2.2  Supported Relationships

| Type | Formula | Use case |
|------|---------|----------|
| **Power** | TV(P) = θ_P · (COV/center)^θ_cov | Allometric scaling (weight) |
| **Proportional** | TV(P) = θ_P · (1 + θ_cov · (COV − center)) | Linear effects |
| **Exponential** | TV(P) = θ_P · exp(θ_cov · (COV − center)) | Age, eGFR |

### 2.3  API

```rust
use ns_inference::{
    ScmEstimator, ScmConfig, CovariateCandidate, CovariateRelationship,
    OmegaMatrix, ErrorModel,
};

// Define candidate covariates
let candidates = vec![
    CovariateCandidate {
        name: "WT_on_CL".to_string(),
        param_index: 0,  // CL
        values: weights.clone(),  // per-subject weights
        center: 70.0,            // median weight
        relationship: CovariateRelationship::Power,
    },
    CovariateCandidate {
        name: "AGE_on_CL".to_string(),
        param_index: 0,  // CL
        values: ages.clone(),
        center: 40.0,
        relationship: CovariateRelationship::Exponential,
    },
    CovariateCandidate {
        name: "WT_on_V".to_string(),
        param_index: 1,  // V
        values: weights.clone(),
        center: 70.0,
        relationship: CovariateRelationship::Power,
    },
];

let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
let scm = ScmEstimator::with_defaults();

let result = scm.run_1cpt_oral(
    &times, &y, &subject_idx, n_subjects,
    dose, bioav, ErrorModel::Additive(sigma),
    &[cl_init, v_init, ka_init],
    &omega,
    &candidates,
).unwrap();

// Inspect results
println!("Base OFV: {:.2}", result.base_ofv);
println!("Final OFV: {:.2}", result.ofv);
println!("Selected covariates:");
for step in &result.selected {
    println!("  {} (ΔOFV = {:.2}, p = {:.4}, coeff = {:.3})",
        step.name, step.delta_ofv, step.p_value, step.coefficient);
}
```

### 2.4  Interpreting Results

| `ScmResult` field | Description |
|-------------------|-------------|
| `selected` | Final retained covariates after backward elimination |
| `forward_trace` | Every covariate added during forward selection |
| `backward_trace` | Every covariate tested for removal |
| `theta` | Final params: `[CL, V, Ka, cov_coeff_1, ...]` |
| `base_ofv` | OFV of the no-covariate base model |
| `ofv` | OFV of the final covariate model |

A typical allometric weight–CL exponent is ~0.75 (3/4 power law). If the
SCM selects WT on CL with coefficient near 0.75, it validates the biological
plausibility of the model.

### 2.5  Customizing Thresholds

```rust
let config = ScmConfig {
    forward_alpha: 0.01,   // stricter forward: ΔOFV > 6.63
    backward_alpha: 0.001, // stricter backward: ΔOFV > 10.83
    ..ScmConfig::default()
};
let scm = ScmEstimator::new(config);
```

---

## 3  VPC & GOF Diagnostics

### 3.1  Goodness-of-Fit (GOF)

GOF diagnostics compare observed vs predicted data at the individual and
population level.

```rust
use ns_inference::{gof_1cpt_oral, ErrorModel};

let records = gof_1cpt_oral(
    &times, &y, &subject_idx,
    dose, bioav,
    &result.theta,
    &result.eta,
    &ErrorModel::Additive(sigma),
).unwrap();

for r in &records {
    println!("Subject {} t={:.1}: DV={:.2} PRED={:.2} IPRED={:.2} IWRES={:.2} CWRES={:.2}",
        r.subject, r.time, r.dv, r.pred, r.ipred, r.iwres, r.cwres);
}
```

| Diagnostic | Formula | Expected |
|------------|---------|----------|
| **PRED** | C(θ, η=0, t) | Population-level prediction |
| **IPRED** | C(θ, η̂_i, t) | Individual prediction at conditional mode |
| **IWRES** | (DV − IPRED) / σ(IPRED) | ~N(0,1) if model is correct |
| **CWRES** | (DV − PRED) / √Var_pop | ~N(0,1) under correct model |

**Key plots** (generate with your preferred plotting library):

- **DV vs PRED** — points should scatter around identity line
- **DV vs IPRED** — tighter scatter than PRED (individual fits)
- **IWRES vs time** — should be uniform band around 0, no trend
- **CWRES vs PRED** — should be uniform, no funnel shape (heteroscedasticity)
- **QQ plot of IWRES** — should follow 45° line

### 3.2  Visual Predictive Check (VPC)

VPC simulates many datasets from the fitted model and compares observed
quantiles against simulated prediction intervals.

```rust
use ns_inference::{vpc_1cpt_oral, VpcConfig, OmegaMatrix, ErrorModel};

let config = VpcConfig {
    n_sim: 200,                           // simulation replicates
    quantiles: vec![0.05, 0.50, 0.95],   // percentiles to track
    n_bins: 10,                           // time bins
    seed: 42,
    pi_level: 0.90,                       // 90% PI on simulated quantiles
};

let vpc = vpc_1cpt_oral(
    &times, &y, &subject_idx, n_subjects,
    dose, bioav,
    &result.theta,
    &result.omega_matrix,
    &ErrorModel::Additive(sigma),
    &config,
).unwrap();

for bin in &vpc.bins {
    println!("Time {:.1} (n={})", bin.time, bin.n_obs);
    println!("  Obs quantiles:  {:?}", bin.obs_quantiles);
    println!("  Sim PI lower:   {:?}", bin.sim_pi_lower);
    println!("  Sim PI median:  {:?}", bin.sim_pi_median);
    println!("  Sim PI upper:   {:?}", bin.sim_pi_upper);
}
```

### 3.3  Interpreting VPC

A well-specified model shows:

- **Observed median** falls within the 90% PI of simulated medians
- **Observed 5th/95th** fall within their respective PIs
- No systematic time-dependent bias

**Warning signs:**

| Pattern | Diagnosis |
|---------|-----------|
| Observed median above all simulated PIs | Model under-predicts |
| Observed 5th/95th outside PIs | Variability model (Ω) is wrong |
| Systematic drift over time | Structural model mis-specification |
| Funnel shape in extremes | Error model needs proportional component |

---

## 4  End-to-End Example: Warfarin Pop PK

```rust
use ns_inference::{
    FoceEstimator, FoceConfig, OmegaMatrix, ErrorModel,
    ScmEstimator, CovariateCandidate, CovariateRelationship,
    gof_1cpt_oral, vpc_1cpt_oral, VpcConfig,
};

// 1. Fit base model with correlated omega
let omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
let estimator = FoceEstimator::focei();
let base = estimator.fit_1cpt_oral_correlated(
    &times, &y, &subject_idx, n_subjects,
    dose, 1.0, ErrorModel::Additive(sigma),
    &[0.1, 5.0, 0.5], omega,
).unwrap();

// 2. Run SCM with weight covariate
let candidates = vec![
    CovariateCandidate {
        name: "WT_on_CL".into(),
        param_index: 0,
        values: weights.clone(),
        center: 70.0,
        relationship: CovariateRelationship::Power,
    },
];
let scm = ScmEstimator::with_defaults();
let scm_result = scm.run_1cpt_oral(
    &times, &y, &subject_idx, n_subjects,
    dose, 1.0, ErrorModel::Additive(sigma),
    &base.theta, &base.omega_matrix, &candidates,
).unwrap();

// 3. GOF on final model
let gof = gof_1cpt_oral(
    &times, &y, &subject_idx, dose, 1.0,
    &scm_result.theta, &base.eta, &ErrorModel::Additive(sigma),
).unwrap();

// 4. VPC on final model
let vpc = vpc_1cpt_oral(
    &times, &y, &subject_idx, n_subjects, dose, 1.0,
    &scm_result.theta, &scm_result.omega,
    &ErrorModel::Additive(sigma),
    &VpcConfig { n_sim: 200, n_bins: 8, seed: 42, ..VpcConfig::default() },
).unwrap();
```

---

## 5  Benchmark Suite

Three classic pharmacometric datasets are available as integration tests:

| Dataset | Drug | Subjects | Model | Test |
|---------|------|----------|-------|------|
| **Warfarin** | warfarin | 32 | 1-cpt oral | `benchmark_warfarin` |
| **Theophylline** | theophylline | 12 | 1-cpt oral | `benchmark_theophylline` |
| **Phenobarbital** | phenobarbital | 40 | 1-cpt oral (fast Ka) | `benchmark_phenobarbital` |
| **Warfarin (corr.)** | warfarin | 40 | 1-cpt oral + corr. Ω | `benchmark_warfarin_correlated_omega` |

Run benchmarks:

```bash
cargo test -p ns-inference --test pharma_benchmark -- --nocapture
```

---

## References

1. `crates/ns-inference/src/foce.rs` — OmegaMatrix, FOCE/FOCEI estimator
2. `crates/ns-inference/src/scm.rs` — Stepwise Covariate Modeling
3. `crates/ns-inference/src/vpc.rs` — VPC and GOF diagnostics
4. `crates/ns-inference/tests/pharma_benchmark.rs` — Benchmark suite
5. `crates/ns-inference/src/pk.rs` — PK models and error models
6. `crates/ns-inference/src/dosing.rs` — Dosing regimen abstraction
7. `crates/ns-inference/src/nonmem.rs` — NONMEM dataset reader
