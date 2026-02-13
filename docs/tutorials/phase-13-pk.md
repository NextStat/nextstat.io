---
title: "Tutorial: Pharmacokinetic (PK) Modeling"
description: "End-to-end guide to PK modeling in NextStat: 1- and 2-compartment models, dosing regimens, error models, NONMEM dataset ingest, and LLOQ handling."
status: stable
last_updated: 2026-02-12
---

# Tutorial: Pharmacokinetic (PK) Modeling

This tutorial covers the full PK modeling API in NextStat, from single-subject fits to multi-dose regimens with real-world NONMEM datasets.

## Table of contents

1. [Structural models](#structural-models)
2. [Error models](#error-models)
3. [Dosing regimens](#dosing-regimens)
4. [NONMEM dataset reader](#nonmem-dataset-reader)
5. [LLOQ handling](#lloq-handling)
6. [End-to-end examples](#end-to-end-examples)
7. [R interface](#r-interface)

---

## Structural models

NextStat provides analytical PK models with closed-form concentration solutions.

### 1-compartment oral

Parameters: `CL` (clearance), `V` (central volume), `Ka` (absorption rate).

```
dA_gut/dt  = −Ka · A_gut
dA_cent/dt =  Ka · A_gut − (CL/V) · A_cent
C(t)       =  A_cent(t) / V
```

**Rust:**

```rust
use ns_inference::{OneCompartmentOralPkModel, ErrorModel, LloqPolicy, MaximumLikelihoodEstimator};

let model = OneCompartmentOralPkModel::with_error_model(
    vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0],   // times
    vec![0.12, 0.18, 0.20, 0.14, 0.08, 0.03], // observed concentrations
    100.0,                    // dose (mg)
    1.0,                      // bioavailability (F)
    ErrorModel::Additive(0.05), // σ = 0.05 mg/L
    None,                     // no LLOQ
    LloqPolicy::Ignore,
).unwrap();

let mle = MaximumLikelihoodEstimator::new();
let fit = mle.fit(&model).unwrap();
// fit.parameters = [CL, V, Ka]
println!("CL={:.4} V={:.4} Ka={:.4}", fit.parameters[0], fit.parameters[1], fit.parameters[2]);
```

**Python:**

```python
import nextstat

model = nextstat.OneCompartmentOralPkModel(
    [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    [0.12, 0.18, 0.20, 0.14, 0.08, 0.03],
    dose=100.0, bioavailability=1.0, sigma=0.05,
)
fit = nextstat.MaximumLikelihoodEstimator().fit(model)
print(dict(zip(model.parameter_names(), fit.bestfit)))
```

### 2-compartment IV bolus

Parameters: `CL`, `V1` (central), `V2` (peripheral), `Q` (intercompartmental clearance).

Analytical bi-exponential solution with eigenvalue decomposition.

```rust
use ns_inference::{TwoCompartmentIvPkModel, ErrorModel, LloqPolicy};

let model = TwoCompartmentIvPkModel::new(
    times, y,
    500.0,                          // dose (mg, IV bolus)
    ErrorModel::Proportional(0.1),  // 10% proportional error
    None, LloqPolicy::Ignore,
).unwrap();
// fit.parameters = [CL, V1, V2, Q]
```

### 2-compartment oral

Parameters: `CL`, `V1`, `V2`, `Q`, `Ka`.

Analytical tri-exponential solution (superposition of α, β, Ka terms).

```rust
use ns_inference::{TwoCompartmentOralPkModel, ErrorModel, LloqPolicy};

let model = TwoCompartmentOralPkModel::new(
    times, y,
    200.0,                    // dose (mg)
    0.85,                     // bioavailability
    ErrorModel::Combined { sigma_add: 0.1, sigma_prop: 0.05 },
    Some(0.01), LloqPolicy::Censored,
).unwrap();
// fit.parameters = [CL, V1, V2, Q, Ka]
```

---

## Error models

NONMEM-style residual error models control the observation noise structure.

| Variant | Formula | Variance |
|---------|---------|----------|
| `Additive(σ)` | y = f + ε | Var = σ² |
| `Proportional(σ)` | y = f·(1 + ε) | Var = (σ·f)² |
| `Combined { σ_add, σ_prop }` | y = f·(1 + ε₁) + ε₂ | Var = σ_add² + (σ_prop·f)² |

Each variant provides:
- `variance(f)` — observation noise variance at predicted `f`
- `nll_obs(y, f)` — per-observation NLL contribution
- `dnll_obs_df(y, f)` — analytical gradient for FOCEI

**Usage:**

```rust
use ns_inference::ErrorModel;

let em = ErrorModel::Combined { sigma_add: 0.5, sigma_prop: 0.1 };
em.validate().unwrap();
let var = em.variance(10.0); // 0.5² + (0.1·10)² = 1.25
```

---

## Dosing regimens

The `DosingRegimen` abstraction supports multi-dose schedules with superposition.

### Dose routes

| Route | Description |
|-------|-------------|
| `IvBolus` | Instantaneous injection into central compartment |
| `Oral { bioavailability }` | First-order absorption with fraction F ∈ (0, 1] |
| `Infusion { duration }` | Zero-order input over specified duration |

### Constructors

```rust
use ns_inference::{DosingRegimen, DoseRoute, DoseEvent};

// Single IV bolus at t=0
let reg = DosingRegimen::single_iv_bolus(500.0).unwrap();

// Single oral dose at t=0
let reg = DosingRegimen::single_oral(200.0, 0.85).unwrap();

// Single IV infusion (1 hour) at t=0
let reg = DosingRegimen::single_infusion(500.0, 1.0).unwrap();

// Repeated oral: 200 mg every 12h for 7 days
let reg = DosingRegimen::repeated(
    200.0, 12.0, 14,
    DoseRoute::Oral { bioavailability: 0.85 },
).unwrap();

// Mixed regimen: loading IV + maintenance oral
let reg = DosingRegimen::from_events(vec![
    DoseEvent { time: 0.0, amount: 500.0, route: DoseRoute::Infusion { duration: 1.0 } },
    DoseEvent { time: 12.0, amount: 200.0, route: DoseRoute::Oral { bioavailability: 0.8 } },
    DoseEvent { time: 24.0, amount: 200.0, route: DoseRoute::Oral { bioavailability: 0.8 } },
]).unwrap();
```

### Computing concentration profiles

```rust
// 1-compartment via superposition
let concs = reg.predict_1cpt(cl, v, ka, &observation_times);

// 2-compartment IV via superposition
let concs = reg.predict_2cpt_iv(cl, v1, v2, q, &observation_times);

// 2-compartment oral via superposition
let concs = reg.predict_2cpt_oral(cl, v1, v2, q, ka, &observation_times);
```

---

## NONMEM dataset reader

`NonmemDataset` parses standard NONMEM-format CSV files.

### Required columns

| Column | Description |
|--------|-------------|
| `ID` | Subject identifier |
| `TIME` | Observation/dosing time |
| `DV` | Dependent variable (observed concentration) |

### Optional columns (with defaults)

| Column | Default | Description |
|--------|---------|-------------|
| `AMT` | 0 | Dose amount |
| `EVID` | Inferred (1 if AMT > 0, else 0) | Event ID |
| `MDV` | Inferred (1 if EVID ≠ 0, else 0) | Missing DV flag |
| `CMT` | 1 | Compartment (1 = depot/oral, 2 = central/IV) |
| `RATE` | 0 | Infusion rate (>0 → infusion, 0 → bolus) |

### Example

```rust
use ns_inference::NonmemDataset;

let csv = "ID,TIME,AMT,EVID,DV,CMT
1,0,100,1,.,1
1,0.5,0,0,2.1,1
1,1.0,0,0,3.8,1
1,2.0,0,0,3.2,1
2,0,100,1,.,1
2,0.5,0,0,1.9,1
2,1.0,0,0,4.1,1
2,2.0,0,0,2.9,1";

let ds = NonmemDataset::from_csv(csv).unwrap();
assert_eq!(ds.n_subjects(), 2);

// Extract observations (EVID=0, MDV=0)
let (times, dv, subj_idx) = ds.observation_data();

// Build per-subject dosing regimens
let regimens = ds.all_dosing_regimens().unwrap();
// Or with custom bioavailability:
let regimens = ds.all_dosing_regimens_with_bioav(0.85).unwrap();

// Per-subject access
let subj_1_reg = ds.dosing_regimen("1").unwrap();
let subj_1_recs = ds.subject_records("1");
```

### Automatic dose route mapping

| CMT | RATE | Route |
|-----|------|-------|
| 1 | — | `Oral { bioavailability }` |
| 2 | 0 | `IvBolus` |
| 2 | > 0 | `Infusion { duration = AMT / RATE }` |

---

## LLOQ handling

Below–limit-of-quantification observations are handled via `LloqPolicy`:

| Policy | Behavior |
|--------|----------|
| `Ignore` | Drop observations below LLOQ |
| `ReplaceHalf` | Replace with LLOQ/2 |
| `Censored` | Left-censored likelihood: −ln Φ((LLOQ − f) / σ(f)) |

The `Censored` policy is the most statistically appropriate (M3 method) and is recommended for regulatory submissions.

```rust
let model = OneCompartmentOralPkModel::with_error_model(
    times, y, dose, bioav,
    ErrorModel::Additive(0.5),
    Some(0.05),              // LLOQ = 0.05 mg/L
    LloqPolicy::Censored,   // M3 method
).unwrap();
```

---

## End-to-end examples

### Example 1: Warfarin single-subject fit

```rust
use ns_inference::{
    OneCompartmentOralPkModel, ErrorModel, LloqPolicy, MaximumLikelihoodEstimator,
};

let times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0];
let y = vec![1.2, 3.8, 6.5, 8.1, 7.2, 5.9, 3.1, 1.8, 1.0, 0.3];

let model = OneCompartmentOralPkModel::with_error_model(
    times, y,
    100.0, 1.0,
    ErrorModel::Additive(0.5),
    None, LloqPolicy::Ignore,
).unwrap();

let fit = MaximumLikelihoodEstimator::new().fit(&model).unwrap();
println!("CL = {:.4} L/h", fit.parameters[0]);
println!("V  = {:.4} L",   fit.parameters[1]);
println!("Ka = {:.4} h⁻¹", fit.parameters[2]);
println!("NLL = {:.2}",     fit.nll);
```

### Example 2: Multi-dose 2-compartment from NONMEM file

```rust
use ns_inference::{NonmemDataset, DosingRegimen, TwoCompartmentOralPkModel, ErrorModel, LloqPolicy};

let csv = std::fs::read_to_string("warfarin_study.csv").unwrap();
let ds = NonmemDataset::from_csv(&csv).unwrap();
let (times, dv, _) = ds.observation_data();

// Use subject 1's dosing regimen for prediction
let reg = ds.dosing_regimen("1").unwrap();
let cl = 0.133; let v1 = 8.0; let v2 = 15.0; let q = 0.5; let ka = 0.8;
let pred = reg.predict_2cpt_oral(cl, v1, v2, q, ka, &times);
```

---

## R interface

The `nextstat` R package provides wrappers for PK workflows:

```r
library(nextstat)

# Single-subject 1-cpt oral fit
model <- nextstat.OneCompartmentOralPkModel(
  times = c(0.5, 1, 2, 4, 8, 12, 24),
  y = c(1.2, 3.8, 6.5, 8.1, 7.2, 5.9, 3.1),
  dose = 100, bioavailability = 1.0, sigma = 0.5
)
fit <- nextstat.MaximumLikelihoodEstimator()$fit(model)
```

For population PK (FOCE/FOCEI), see [NLME tutorial](phase-13-nlme.md).

---

## Test coverage

| Module | Tests | Status |
|--------|-------|--------|
| `pk` (1-cpt + 2-cpt models) | 17 | ✅ |
| `dosing` (regimens + superposition) | 11 | ✅ |
| `nonmem` (dataset reader) | 12 | ✅ |

```bash
cargo test -p ns-inference -- pk::tests dosing::tests nonmem::tests
```

## Related

- [NLME tutorial](phase-13-nlme.md) — population PK with FOCE/FOCEI + random effects
- [Pharma benchmark suite](/docs/benchmarks/suites/pharma.md) — Warfarin/Theophylline parity
