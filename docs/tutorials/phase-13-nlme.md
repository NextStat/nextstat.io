---
title: "Tutorial: Population PK with NLME (FOCE/FOCEI + SAEM)"
description: "End-to-end guide to population pharmacokinetic modeling in NextStat: FOCE/FOCEI and SAEM estimation, correlated random effects, GOF/VPC diagnostics, stepwise covariate modeling, and result interpretation."
status: stable
last_updated: 2026-02-12
---

# Tutorial: Population PK with NLME (FOCE/FOCEI + SAEM)

This tutorial covers the full nonlinear mixed-effects (NLME) estimation pipeline in NextStat — from raw data to diagnostics-grade results.

## Table of contents

1. [Statistical model](#statistical-model)
2. [FOCE/FOCEI estimation](#focefocei-estimation)
3. [SAEM estimation](#saem-estimation)
4. [Correlated random effects](#correlated-random-effects)
5. [Goodness-of-fit diagnostics](#goodness-of-fit-diagnostics)
6. [Visual predictive check](#visual-predictive-check)
7. [Stepwise covariate modeling](#stepwise-covariate-modeling)
8. [Interpreting results](#interpreting-results)
9. [R interface](#r-interface)
10. [End-to-end example](#end-to-end-example)

---

## Statistical model

Population PK uses a hierarchical model with two levels:

**Structural model** (1-compartment oral):

```
C_i(t) = (F·Dose·Ka_i / (V_i·(Ka_i − Ke_i))) · (exp(−Ke_i·t) − exp(−Ka_i·t))
where Ke_i = CL_i / V_i
```

**Individual parameters** (log-normal inter-individual variability):

```
CL_i = CL_pop · exp(η_CL,i)
V_i  = V_pop  · exp(η_V,i)
Ka_i = Ka_pop · exp(η_Ka,i)

η_i ~ MVN(0, Ω)
```

**Observation model**:

```
y_ij = C_i(t_ij) + ε_ij   (additive)
y_ij = C_i(t_ij)·(1+ε_ij) (proportional)
```

---

## FOCE/FOCEI estimation

`FoceEstimator` implements the industry-standard First-Order Conditional Estimation.

### Configuration

```rust
use ns_inference::{FoceConfig, FoceEstimator};

// FOCEI (default, recommended)
let est = FoceEstimator::focei();

// FOCE (no interaction term)
let est = FoceEstimator::foce();

// Custom configuration
let est = FoceEstimator::new(FoceConfig {
    max_outer_iter: 100,  // population parameter iterations
    max_inner_iter: 20,   // per-subject ETA optimization (Newton)
    tol: 1e-4,            // ΔOFV convergence criterion
    interaction: true,     // FOCEI (accounts for η-dependence in error model)
});
```

### Fitting

```rust
use ns_inference::{FoceEstimator, ErrorModel};

let fit = FoceEstimator::focei().fit_1cpt_oral(
    &times,        // observation times (flattened)
    &dv,           // observed concentrations (flattened)
    &subject_idx,  // subject index per observation (0-based)
    n_subjects,    // number of unique subjects
    100.0,         // dose
    1.0,           // bioavailability
    ErrorModel::Additive(0.5),
    &[0.133, 8.0, 0.8],   // theta_init: [CL, V, Ka]
    &[0.30, 0.25, 0.30],  // omega_init: [ω_CL, ω_V, ω_Ka]
).unwrap();

println!("θ = {:?}", fit.theta);    // population parameters
println!("ω = {:?}", fit.omega);    // random effect SDs
println!("OFV = {:.2}", fit.ofv);   // -2·logL
println!("converged = {}", fit.converged);
println!("iterations = {}", fit.n_iter);
```

### FoceResult fields

| Field | Type | Description |
|-------|------|-------------|
| `theta` | `Vec<f64>` | Population fixed effects [CL, V, Ka] |
| `omega` | `Vec<f64>` | Random effect SDs [ω_CL, ω_V, ω_Ka] |
| `omega_matrix` | `OmegaMatrix` | Full Ω (Cholesky-parameterized) |
| `correlation` | `Vec<Vec<f64>>` | Correlation matrix from Ω |
| `eta` | `Vec<Vec<f64>>` | Conditional modes η̂ per subject |
| `ofv` | `f64` | Objective function value (−2·log L) |
| `converged` | `bool` | Convergence flag |
| `n_iter` | `usize` | Number of outer iterations |

---

## SAEM estimation

`SaemEstimator` implements the Stochastic Approximation EM algorithm (Monolix-class). More robust than FOCE for complex nonlinear models.

```rust
use ns_inference::{SaemConfig, SaemEstimator};

let config = SaemConfig {
    n_burn: 200,       // burn-in iterations (step size γ = 1)
    n_iter: 100,       // estimation iterations (step size γ = 1/k)
    n_chains: 1,       // MCMC chains per subject
    proposal_sd: 0.1,  // initial MH proposal SD
    tol: 1e-4,
    seed: 42,
};

let est = SaemEstimator::new(config);
let result = est.fit_1cpt_oral(
    &times, &dv, &subject_idx, n_subjects,
    100.0, 1.0, ErrorModel::Additive(0.5),
    &[0.133, 8.0, 0.8], &[0.3, 0.25, 0.3],
).unwrap();

// result.theta, result.omega, result.ofv, result.converged
// result.diagnostics.acceptance_rates — per-subject MCMC acceptance
// result.diagnostics.ofv_trace — OFV convergence history
```

### When to use SAEM vs FOCE

| Criterion | FOCE/FOCEI | SAEM |
|-----------|-----------|------|
| **Speed** | Faster (deterministic) | Slower (stochastic) |
| **Convergence** | Can fail for complex models | More robust |
| **Diagnostics** | OFV only | OFV trace + acceptance rates |
| **Regulatory** | Gold standard (NONMEM) | Accepted (Monolix, nlmixr2) |
| **Best for** | Standard PK, well-behaved data | PD/ODE models, flat likelihood |

---

## Correlated random effects

`OmegaMatrix` stores the full Ω via Cholesky factor L (Ω = L·Lᵀ).

### Constructors

```rust
use ns_inference::OmegaMatrix;

// Diagonal (independent random effects)
let om = OmegaMatrix::from_diagonal(&[0.3, 0.25, 0.3]).unwrap();

// From SDs + correlation matrix
let sds = vec![0.3, 0.25, 0.3];
let corr = vec![
    vec![1.0, 0.6, 0.0],
    vec![0.6, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
];
let om = OmegaMatrix::from_correlation(&sds, &corr).unwrap();

// From full covariance matrix
let cov = vec![
    vec![0.09,  0.045, 0.0],
    vec![0.045, 0.0625, 0.0],
    vec![0.0,   0.0,    0.09],
];
let om = OmegaMatrix::from_covariance(&cov).unwrap();
```

### Fitting with correlated Ω

```rust
let om_init = OmegaMatrix::from_correlation(
    &[0.3, 0.25, 0.3],
    &[vec![1.0, 0.6, 0.0], vec![0.6, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
).unwrap();

let fit = FoceEstimator::focei().fit_1cpt_oral_correlated(
    &times, &dv, &subject_idx, n_subjects,
    100.0, 1.0, ErrorModel::Additive(0.5),
    &[0.133, 8.0, 0.8], om_init,
).unwrap();

// Inspect estimated correlation
for row in &fit.correlation {
    println!("{:.3?}", row);
}
```

---

## Goodness-of-fit diagnostics

`gof_1cpt_oral()` computes per-observation diagnostic quantities after fitting.

```rust
use ns_inference::gof_1cpt_oral;

let gof = gof_1cpt_oral(
    &times, &dv, &subject_idx,
    100.0, 1.0,
    &fit.theta, &fit.eta,
    &ErrorModel::Additive(0.5),
).unwrap();

for rec in &gof {
    println!(
        "subj={} t={:.1} DV={:.2} PRED={:.2} IPRED={:.2} IWRES={:.3} CWRES={:.3}",
        rec.subject, rec.time, rec.dv, rec.pred, rec.ipred, rec.iwres, rec.cwres
    );
}
```

### Diagnostic quantities

| Quantity | Formula | Interpretation |
|----------|---------|---------------|
| **PRED** | C(t; θ_pop, η=0) | Population prediction |
| **IPRED** | C(t; θ_pop, η̂_i) | Individual prediction |
| **IWRES** | (DV − IPRED) / σ(IPRED) | Individual weighted residual |
| **CWRES** | FOCE-based conditional residual | Should be ~ N(0, 1) |

### Standard GOF plots (interpretation)

- **DV vs PRED** — systematic bias → misspecified structural model
- **DV vs IPRED** — poor individual fit → inadequate random effects
- **IWRES vs TIME** — trending pattern → time-dependent bias
- **CWRES vs PRED** — heteroscedasticity → wrong error model
- **QQ-plot of CWRES** — departure from N(0,1) → model misspecification

---

## Visual predictive check

`vpc_1cpt_oral()` simulates replicates and computes prediction intervals.

```rust
use ns_inference::{VpcConfig, vpc_1cpt_oral};

let vpc = vpc_1cpt_oral(
    &times, &dv, &subject_idx,
    n_subjects, 100.0, 1.0,
    &fit.theta, &fit.omega_matrix,
    &ErrorModel::Additive(0.5),
    &VpcConfig {
        n_sim: 200,     // simulation replicates
        n_bins: 10,     // time bins
        quantiles: vec![0.05, 0.50, 0.95],
        seed: 123,
    },
).unwrap();

// vpc.bins: Vec<VpcBin> — per-bin observed + simulated quantiles
for bin in &vpc.bins {
    println!(
        "t=[{:.1}, {:.1}] obs_median={:.2} sim_PI=[{:.2}, {:.2}]",
        bin.time_lo, bin.time_hi,
        bin.observed_quantiles[1],
        bin.simulated_pi_lo[1], bin.simulated_pi_hi[1],
    );
}
```

### VPC interpretation

- **Observed median inside simulated PI** → good model fit
- **Observed 5th/95th outside PI** → variability misspecified (check Ω, error model)
- **Systematic shift** → structural model or covariate effect missing

---

## Stepwise covariate modeling

`ScmEstimator` performs forward selection + backward elimination.

```rust
use ns_inference::{
    ScmConfig, ScmEstimator, CovariateCandidate, CovariateRelationship,
};

let candidates = vec![
    CovariateCandidate {
        name: "WT on CL".into(),
        param_index: 0,  // CL
        values: weights.clone(),  // per-subject
        center: 70.0,    // reference (median weight)
        relationship: CovariateRelationship::Power,
    },
    CovariateCandidate {
        name: "AGE on V".into(),
        param_index: 1,  // V
        values: ages.clone(),
        center: 45.0,
        relationship: CovariateRelationship::Exponential,
    },
];

let scm = ScmEstimator::new(ScmConfig {
    forward_alpha: 0.05,   // ΔOFV > 3.84 to add
    backward_alpha: 0.01,  // ΔOFV > 6.63 to keep
    foce: FoceConfig::default(),
});

let result = scm.run(
    &times, &dv, &subject_idx, n_subjects,
    100.0, 1.0, ErrorModel::Additive(0.5),
    &[0.133, 8.0, 0.8], &[0.3, 0.25, 0.3],
    &candidates,
).unwrap();

println!("Base OFV: {:.2}", result.base_ofv);
println!("Final OFV: {:.2}", result.ofv);
println!("Selected covariates:");
for step in &result.selected {
    println!("  {} — ΔOFV={:.2}, p={:.4}, coeff={:.4}",
        step.name, step.delta_ofv, step.p_value, step.coefficient);
}
```

### SCM interpretation

| Threshold | α | ΔOFV (χ²₁) | Used for |
|-----------|---|------------|----------|
| Forward | 0.05 | 3.84 | Adding covariates |
| Backward | 0.01 | 6.63 | Removing covariates |

The more stringent backward threshold guards against overfitting.

---

## Interpreting results

### Parameter recovery checklist

1. **θ̂ within ±3ω of true** — adequate recovery at population level
2. **OFV decrease** with covariates — covariate effects are real
3. **CWRES ~ N(0,1)** — observation model is correct
4. **VPC median inside PI** — structural model captures central tendency
5. **Ω estimates** — positive and physiologically plausible

### Common pitfalls

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Non-convergence | Bad initial values | Use literature θ₀ |
| θ̂ at boundary | Over-parameterized | Reduce model complexity |
| Ω → 0 for a parameter | No IIV on that parameter | Remove η from model |
| CWRES trending | Wrong error model | Try proportional/combined |
| VPC underpredicts tails | Ω too small | Allow correlated Ω |

---

## R interface

```r
library(nextstat)

# FOCE/FOCEI estimation
fit <- ns_foce(
  times = dat$TIME, dv = dat$DV, id = dat$ID - 1,
  n_subjects = length(unique(dat$ID)),
  dose = 100, theta_init = c(0.133, 8, 0.8),
  omega_init = c(0.3, 0.25, 0.3), sigma = 0.5
)
fit$theta   # population parameters
fit$omega   # random effect SDs
fit$ofv     # objective function value

# SAEM estimation
fit_saem <- ns_saem(
  times = dat$TIME, dv = dat$DV, id = dat$ID - 1,
  n_subjects = length(unique(dat$ID)),
  dose = 100, theta_init = c(0.133, 8, 0.8),
  omega_init = c(0.3, 0.25, 0.3), sigma = 0.5, seed = 42
)

# VPC
vpc <- ns_vpc(dat$TIME, dat$DV, dat$ID-1, 32, 100,
  theta = fit$theta, omega = fit$omega, sigma = 0.5)

# GOF diagnostics
gof <- ns_gof(dat$TIME, dat$DV, dat$ID-1, 32, 100,
  theta = fit$theta, omega = fit$omega, eta = fit$eta, sigma = 0.5)
```

---

## End-to-end example

### Warfarin population PK (32 subjects)

```rust
use ns_inference::{
    FoceEstimator, ErrorModel, OmegaMatrix,
    gof_1cpt_oral, vpc_1cpt_oral, VpcConfig,
};

// 1. Prepare data (32 subjects, 10 observations each)
let (times, dv, subject_idx) = generate_warfarin_data(32, 10, seed);

// 2. Fit with FOCEI
let fit = FoceEstimator::focei().fit_1cpt_oral(
    &times, &dv, &subject_idx, 32,
    100.0, 1.0,
    ErrorModel::Additive(0.5),
    &[0.133, 8.0, 0.8],   // true θ as init
    &[0.30, 0.25, 0.30],  // true ω as init
).unwrap();
assert!(fit.converged);

// 3. GOF diagnostics
let gof = gof_1cpt_oral(
    &times, &dv, &subject_idx, 100.0, 1.0,
    &fit.theta, &fit.eta, &ErrorModel::Additive(0.5),
).unwrap();
let mean_iwres: f64 = gof.iter().map(|r| r.iwres).sum::<f64>() / gof.len() as f64;
assert!(mean_iwres.abs() < 1.0, "IWRES mean should be near zero");

// 4. VPC
let vpc = vpc_1cpt_oral(
    &times, &dv, &subject_idx, 32, 100.0, 1.0,
    &fit.theta, &fit.omega_matrix, &ErrorModel::Additive(0.5),
    &VpcConfig { n_sim: 200, n_bins: 10, quantiles: vec![0.05, 0.5, 0.95], seed: 123 },
).unwrap();
// Check: observed median within simulated PI for each bin
```

---

## Test coverage

| Module | Tests | Status |
|--------|-------|--------|
| `foce` (FOCE/FOCEI + OmegaMatrix) | 13 | ✅ |
| `saem` (SAEM algorithm) | 5 | ✅ |
| `scm` (stepwise covariate modeling) | 7 | ✅ |
| `vpc` (VPC + GOF diagnostics) | 7 | ✅ |
| `pharma_benchmark` (integration) | 4 | ✅ |
| `phase3_benchmark` (SAEM/PD/ODE) | 7 | ✅ |

```bash
cargo test -p ns-inference -- foce::tests saem::tests scm::tests vpc::tests
cargo test -p ns-inference --test pharma_benchmark --test phase3_benchmark
```

## Related

- [PK tutorial](phase-13-pk.md) — single-subject PK models, dosing, NONMEM reader
- [Pharma benchmark suite](/docs/benchmarks/suites/pharma.md) — Warfarin/Theophylline parity
- [IQ/OQ/PQ validation](/docs/validation/iq-oq-pq-protocol.md) — GxP qualification
