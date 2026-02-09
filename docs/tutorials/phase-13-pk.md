---
title: "Phase 13: PK Baseline (1-compartment, oral)"
status: stable
---

# Phase 13: PK Baseline (1-compartment, oral)

This tutorial documents the **baseline** 1-compartment PK model with first-order absorption.

## What is implemented

Model parameters (per fit):
- `cl`: clearance
- `v`: central volume
- `ka`: absorption rate

Inputs:
- `times`: observation times (>= 0)
- `y`: observed concentrations (>= 0)
- `dose`: single oral dose at `t=0`
- `bioavailability` (`F`): scalar multiplier on the dose

Observation model:
- `y_i ~ Normal(C(t_i; cl, v, ka), sigma)` with fixed `sigma`

LLOQ handling (optional):
- `lloq=None` disables LLOQ handling.
- `lloq_policy`:
  - `"ignore"`: drop points below LLOQ
  - `"replace_half"`: replace with `LLOQ/2`
  - `"censored"`: left-censored likelihood term `P(Y < LLOQ)`

## Python quickstart

```python
import nextstat

times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
y = [0.12, 0.18, 0.20, 0.14, 0.08, 0.03]  # example concentrations

model = nextstat.OneCompartmentOralPkModel(
    times,
    y,
    dose=100.0,
    bioavailability=1.0,
    sigma=0.05,
    lloq=None,
    lloq_policy="censored",
)

mle = nextstat.MaximumLikelihoodEstimator()
fit = mle.fit(model)

print("params:", dict(zip(model.parameter_names(), fit.bestfit)))
print("nll:", fit.nll)
print("pred:", model.predict(fit.bestfit))
```

## Rust quickstart

```rust
use ns_inference::{MaximumLikelihoodEstimator, OneCompartmentOralPkModel, LloqPolicy};

let times = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
let y = vec![0.12, 0.18, 0.20, 0.14, 0.08, 0.03];

let model = OneCompartmentOralPkModel::new(
    times,
    y,
    100.0, // dose
    1.0,   // bioavailability
    0.05,  // sigma
    None,  // lloq
    LloqPolicy::Censored,
).unwrap();

let mle = MaximumLikelihoodEstimator::new();
let fit = mle.fit(&model).unwrap();
println!("nll={} params={:?}", fit.nll, fit.parameters);
```

## Related

- NLME baseline: `docs/tutorials/phase-13-nlme.md`

