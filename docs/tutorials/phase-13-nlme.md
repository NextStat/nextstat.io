# Phase 13: NLME Baseline (Population + Individual Random Effects)

This tutorial documents the **baseline** nonlinear mixed-effects (NLME) implementation for the
Phase 13 1-compartment oral PK model.

## What is implemented

Model components:

- **Population parameters**: `cl_pop`, `v_pop`, `ka_pop`
- **Log-normal individual random effects** (diagonal Omega baseline):
  - `cl_i = cl_pop * exp(eta_cl[i])`
  - `v_i = v_pop * exp(eta_v[i])`
  - `ka_i = ka_pop * exp(eta_ka[i])`
- **Random-effects priors**:
  - `eta_*[i] ~ Normal(0, omega_*)` (independent across parameters and subjects)
- **Observation model** (same baseline as individual PK):
  - `y_ij ~ Normal(C(t_ij; cl_i, v_i, ka_i), sigma)` with fixed `sigma`

Fitting:
- Baseline fit is **joint MAP** over population params, omegas, and all subject-level `eta`s.
- A generic **Laplace approximation** is available post-fit via `ns_inference::laplace_log_marginal(...)`.

## Limitations vs “full” NLME workflows (NONMEM / Stan / etc.)

This baseline is intentionally minimal:

- No FOCE/FOCEI-style marginalization over random effects (i.e. we optimize the full joint objective).
- No correlated random effects (Omega is diagonal).
- No rich dosing regimens (baseline assumes a single oral dose at `t=0` with fixed `dose` and `F`).
- Observation noise is additive Normal with fixed `sigma` (no proportional/combined error model yet).
- Exposed in the Python bindings as `nextstat.OneCompartmentOralPkNlmeModel` (and `OneCompartmentOralPkModel`).

## Rust usage (MAP + Laplace)

```rust
use ns_inference::{MaximumLikelihoodEstimator, OneCompartmentOralPkNlmeModel, LloqPolicy, laplace_log_marginal};

let n_subjects = 3usize;
let times = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

// Flattened observations: (t_k, y_k, subject_idx_k)
let mut t_all = Vec::new();
let mut y_all = Vec::new();
let mut sid_all = Vec::new();
for sid in 0..n_subjects {
    for &t in &times {
        t_all.push(t);
        y_all.push(0.0); // fill with observations
        sid_all.push(sid);
    }
}

let model = OneCompartmentOralPkNlmeModel::new(
    t_all,
    y_all,
    sid_all,
    n_subjects,
    100.0, // dose
    1.0,   // bioavailability
    0.05,  // sigma
    None,  // lloq
    LloqPolicy::Censored,
).unwrap();

let mle = MaximumLikelihoodEstimator::new();
let fit = mle.fit(&model).unwrap(); // MAP for this NLME objective

let lap = laplace_log_marginal(&model, &fit.parameters).unwrap();
println!("nll(mode)={} logZ~={}", lap.nll_at_mode, lap.log_marginal);
```
