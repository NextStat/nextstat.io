---
title: "Stepwise Covariate Modeling (SCM) with NextStat"
description: "Complete guide to SCM forward selection and backward elimination for population PK covariate analysis. Covers data preparation, API usage, result interpretation, and clinical examples."
status: stable
last_updated: 2026-02-21
---

# Stepwise Covariate Modeling (SCM) with NextStat

This tutorial covers the full workflow for covariate analysis in population
pharmacokinetics using NextStat's SCM implementation. It is aimed at
pharmacometricians who want to identify statistically significant covariate
effects on PK parameters (clearance, volume, absorption rate constant).

## Table of contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Data preparation](#3-data-preparation)
4. [Running SCM](#4-running-scm)
5. [Interpreting results](#5-interpreting-results)
6. [Covariate relationship types](#6-covariate-relationship-types)
7. [Clinical example: allometric scaling](#7-clinical-example-allometric-scaling)
8. [Complete runnable example](#8-complete-runnable-example)
9. [Advanced configuration](#9-advanced-configuration)
10. [Comparison with NONMEM SCM](#10-comparison-with-nonmem-scm)

---

## 1. Overview

Stepwise Covariate Modeling (SCM) is a two-phase algorithm for identifying
which patient covariates (e.g., body weight, age, sex) significantly affect
pharmacokinetic parameters:

**Phase 1 -- Forward selection:**
- Start with the base model (no covariates).
- Test each candidate covariate one at a time.
- Add the covariate that produces the largest drop in OFV (objective function
  value), provided it exceeds the forward threshold (default: delta_OFV > 3.84,
  corresponding to p < 0.05 on a chi-squared test with 1 degree of freedom).
- Repeat until no remaining covariate meets the criterion.

**Phase 2 -- Backward elimination:**
- From the model with all forward-selected covariates, remove each covariate
  one at a time.
- Drop the covariate whose removal causes the smallest increase in OFV,
  provided that increase is below the backward threshold (default:
  delta_OFV < 6.63, corresponding to p > 0.01).
- Repeat until all remaining covariates are significant at the stricter
  backward threshold.

The asymmetric thresholds (0.05 forward, 0.01 backward) guard against
overfitting: it is easier to add a covariate than to keep it in the final
model.

---

## 2. Installation

```bash
pip install nextstat
```

Verify:

```python
import nextstat
print(nextstat.__version__)
```

---

## 3. Data preparation

`nextstat.scm()` operates on a 1-compartment oral PK model and expects the
following inputs:

### Observation data

| Parameter | Type | Description |
|-----------|------|-------------|
| `times` | `list[float]` | Observation times (length N_obs) |
| `y` | `list[float]` | Observed concentrations (length N_obs) |
| `subject_idx` | `list[int]` | Subject index for each observation (0-based, length N_obs) |
| `n_subjects` | `int` | Total number of subjects |

### Covariate data

| Parameter | Type | Description |
|-----------|------|-------------|
| `covariates` | `list[list[float]]` | Each inner list is one covariate vector (length N_obs). Per-subject values are extracted from the first observation of each subject. |
| `covariate_names` | `list[str]` | Names for each covariate (e.g., `["WT", "AGE"]`) |

The `covariates` parameter is a list of covariate vectors. Each vector has one
value per observation (length N_obs), but SCM uses only the first observation
per subject to extract a single per-subject covariate value. This matches the
NONMEM convention where time-varying covariates use the baseline value.

**Centering:** NextStat automatically centers each covariate at its median
across subjects. You do not need to pre-center.

### Dosing and error model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dose` | `float` | (required) | Dose amount |
| `bioavailability` | `float` | `1.0` | Bioavailability fraction |
| `error_model` | `str` | `"proportional"` | `"additive"`, `"proportional"`, or `"combined"` |
| `sigma` | `float` | `0.1` | Residual error magnitude |
| `sigma_add` | `float` | `None` | Additive sigma (required for `"combined"` model) |

### Initial estimates

| Parameter | Type | Description |
|-----------|------|-------------|
| `theta_init` | `list[float]` | Initial population parameters `[CL, V, Ka]` |
| `omega_init` | `list[float]` | Diagonal of the Omega matrix `[omega_CL, omega_V, omega_Ka]` |

---

## 4. Running SCM

```python
import nextstat

# Observation data (from NONMEM dataset or synthetic)
times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0] * 20  # 20 subjects, 6 timepoints each
y = [...]       # observed concentrations
subject_idx = [i for i in range(20) for _ in range(6)]

# Covariate vectors (one value per observation, length = len(times))
weights = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0,  # subject 0
           90.0, 90.0, 90.0, 90.0, 90.0, 90.0,  # subject 1
           ...]  # etc.
ages = [45.0, 45.0, ...]  # same pattern

result = nextstat.scm(
    times,
    y,
    subject_idx,
    n_subjects=20,
    covariates=[weights, ages],
    covariate_names=["WT", "AGE"],
    dose=100.0,
    bioavailability=1.0,
    error_model="proportional",
    sigma=0.1,
    theta_init=[1.0, 10.0, 1.5],       # [CL, V, Ka]
    omega_init=[0.30, 0.25, 0.30],      # diagonal Omega
    param_names=["CL", "V", "Ka"],
    relationships=["power", "exponential"],  # WT=power, AGE=exponential
    forward_alpha=0.05,
    backward_alpha=0.01,
)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `param_names` | `["CL", "V", "Ka"]` | Names for the 3 PK parameters |
| `relationships` | all `"power"` | Covariate relationship type per covariate |
| `forward_alpha` | `0.05` | Significance level for forward selection |
| `backward_alpha` | `0.01` | Significance level for backward elimination |
| `max_outer_iter` | `100` | Maximum FOCE outer iterations per refit |
| `max_inner_iter` | `20` | Maximum inner (eta) optimization iterations |
| `tol` | `1e-4` | FOCE convergence tolerance |

### How `relationships` maps to candidates

Each covariate is tested against each PK parameter. If you have 2 covariates
and 3 PK parameters, SCM evaluates 6 candidates:

| Candidate | Covariate | Parameter | Relationship |
|-----------|-----------|-----------|-------------|
| `WT_on_CL` | WT | CL | power |
| `WT_on_V` | WT | V | power |
| `WT_on_Ka` | WT | Ka | power |
| `AGE_on_CL` | AGE | CL | exponential |
| `AGE_on_V` | AGE | V | exponential |
| `AGE_on_Ka` | AGE | Ka | exponential |

---

## 5. Interpreting results

`nextstat.scm()` returns an `ScmResult` dictionary with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `selected` | `list[ScmStepResult]` | Final selected covariates (after backward elimination) |
| `forward_trace` | `list[ScmStepResult]` | All covariates added during forward selection |
| `backward_trace` | `list[ScmStepResult]` | Covariates tested during backward elimination |
| `base_ofv` | `float` | OFV of the base (no-covariate) model |
| `final_ofv` | `float` | OFV of the final model |
| `n_forward_steps` | `int` | Number of covariates added in forward phase |
| `n_backward_steps` | `int` | Number of covariates removed in backward phase |
| `theta` | `list[float]` | Final population parameters `[CL, V, Ka, coeff_1, ...]` |
| `omega` | `list[list[float]]` | Final Omega variance-covariance matrix (3x3) |

Each `ScmStepResult` entry contains:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Covariate-parameter name (e.g., `"WT_on_CL"`) |
| `param_index` | `int` | Parameter index (0=CL, 1=V, 2=Ka) |
| `relationship` | `str` | `"power"`, `"proportional"`, or `"exponential"` |
| `delta_ofv` | `float` | Change in OFV (negative = improvement when added) |
| `p_value` | `float` | p-value from chi-squared(1) test |
| `coefficient` | `float` | Estimated covariate coefficient |
| `included` | `bool` | Whether this covariate was retained |

### Printing results

```python
print(f"Base OFV:  {result['base_ofv']:.2f}")
print(f"Final OFV: {result['final_ofv']:.2f}")
print(f"OFV drop:  {result['base_ofv'] - result['final_ofv']:.2f}")
print(f"Forward steps:  {result['n_forward_steps']}")
print(f"Backward steps: {result['n_backward_steps']}")

print("\nSelected covariates:")
for step in result["selected"]:
    print(
        f"  {step['name']:15s} ({step['relationship']:14s}): "
        f"delta_OFV = {step['delta_ofv']:8.2f}, "
        f"p = {step['p_value']:.4f}, "
        f"coeff = {step['coefficient']:.4f}"
    )

print("\nFull forward trace:")
for step in result["forward_trace"]:
    status = "ADDED" if step["included"] else "rejected"
    print(
        f"  {step['name']:15s}: "
        f"delta_OFV = {step['delta_ofv']:8.2f}, "
        f"p = {step['p_value']:.4f} -> {status}"
    )

print(f"\nFinal theta: {result['theta']}")
```

---

## 6. Covariate relationship types

NextStat supports three covariate-parameter relationships:

### Power (allometric)

```
TV(P) = theta_P * (COV / COV_median) ^ beta
```

- The most common parameterization for body-size covariates.
- When beta = 0.75 for clearance, this is the standard allometric scaling.
- When beta = 1.0 for volume, this is isometric (linear) scaling.

### Proportional (linear)

```
TV(P) = theta_P * (1 + beta * (COV - COV_median))
```

- Linear modification around the centering value.
- Appropriate for covariates with a roughly linear effect.

### Exponential

```
TV(P) = theta_P * exp(beta * (COV - COV_median))
```

- Ensures the parameter stays positive regardless of covariate value.
- Common for age effects, laboratory values.

### Choosing a relationship

| Covariate | Recommended | Rationale |
|-----------|-------------|-----------|
| Body weight | `"power"` | Allometric scaling theory |
| BSA (body surface area) | `"power"` | Same biological basis |
| Age | `"exponential"` | Monotonic, physiologically bounded |
| Creatinine clearance | `"power"` | Organ function, proportional scaling |
| Sex (0/1) | `"proportional"` | Fractional shift |
| Genotype (0/1/2) | `"proportional"` | Ordinal coding |

---

## 7. Clinical example: allometric scaling

A common pharmacometric question: does body weight predict clearance according
to allometric scaling (CL proportional to WT^0.75)?

```python
import nextstat
import math

# Simulate 30 subjects with weight-dependent clearance
n_subjects = 30
n_obs_per = 6
sample_times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

# True parameters
cl_pop = 1.2      # L/h
v_pop = 15.0      # L
ka_pop = 2.0      # 1/h
dose = 100.0      # mg
wt_exponent = 0.75  # true allometric exponent

# Subject weights (40-100 kg range)
import random
random.seed(42)
weights_per_subject = [40.0 + 60.0 * i / (n_subjects - 1) for i in range(n_subjects)]
wt_median = sorted(weights_per_subject)[n_subjects // 2]

# Expand covariate to per-observation
weights_obs = []
for wt in weights_per_subject:
    weights_obs.extend([wt] * n_obs_per)

# ... (generate y from the PK model with true WT effect)

result = nextstat.scm(
    times,
    y,
    subject_idx,
    n_subjects=n_subjects,
    covariates=[weights_obs],
    covariate_names=["WT"],
    dose=dose,
    bioavailability=1.0,
    error_model="additive",
    sigma=0.05,
    theta_init=[1.0, 10.0, 1.5],
    omega_init=[0.30, 0.30, 0.30],
    relationships=["power"],
    forward_alpha=0.05,
    backward_alpha=0.01,
)

# Check if WT on CL was selected
wt_on_cl = [s for s in result["selected"] if s["name"] == "WT_on_CL"]
if wt_on_cl:
    coeff = wt_on_cl[0]["coefficient"]
    print(f"WT on CL selected: exponent = {coeff:.3f}")
    print(f"True exponent: {wt_exponent}")
    print(f"Estimation error: {abs(coeff - wt_exponent):.3f}")
else:
    print("WT on CL was NOT selected (insufficient power or effect)")
```

### Clinical interpretation

If the estimated exponent for weight on clearance is close to 0.75, this
confirms the standard allometric scaling relationship. This means:
- A 100 kg patient has CL = CL_pop * (100/70)^0.75 = 1.29 * CL_pop
- A 50 kg patient has CL = CL_pop * (50/70)^0.75 = 0.79 * CL_pop

This guides dose adjustment: heavier patients may need higher doses to achieve
the same target exposure, and vice versa.

---

## 8. Complete runnable example

This example generates synthetic data with a known weight effect on clearance
and runs SCM to recover it.

```python
import nextstat
import math
import random

random.seed(42)

# --- Configuration ---
n_subjects = 30
n_obs_per = 6
sample_times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
dose = 100.0
bioav = 1.0

# True population PK parameters
cl_pop = 1.2    # L/h
v_pop = 15.0    # L
ka_pop = 2.0    # 1/h
sigma = 0.05    # additive residual error SD

# True covariate effect: WT on CL with power = 0.75
true_wt_exponent = 0.75
wt_center = 70.0

# Inter-individual variability
omega_cl = 0.15
omega_v = 0.15
omega_ka = 0.15

# --- Generate subject-level data ---
weights = [40.0 + 60.0 * i / (n_subjects - 1) for i in range(n_subjects)]

times = []
y = []
subject_idx = []
weights_obs = []  # per-observation covariate
ages_obs = []     # a noise covariate (no true effect)

for sid in range(n_subjects):
    wt = weights[sid]
    age = 30.0 + 40.0 * random.random()  # random age, no true effect

    # Individual PK with weight effect on CL
    eta_cl = random.gauss(0, omega_cl)
    eta_v = random.gauss(0, omega_v)
    eta_ka = random.gauss(0, omega_ka)

    cl_i = cl_pop * (wt / wt_center) ** true_wt_exponent * math.exp(eta_cl)
    v_i = v_pop * math.exp(eta_v)
    ka_i = ka_pop * math.exp(eta_ka)

    for t in sample_times:
        # 1-compartment oral: C(t) = (F*D*Ka)/(V*(Ka-CL/V)) * (exp(-CL/V*t) - exp(-Ka*t))
        ke = cl_i / v_i
        if abs(ka_i - ke) < 1e-10:
            ka_i += 0.01  # avoid singularity
        c = (bioav * dose * ka_i) / (v_i * (ka_i - ke)) * (
            math.exp(-ke * t) - math.exp(-ka_i * t)
        )
        obs = max(0.0, c + random.gauss(0, sigma))

        times.append(t)
        y.append(obs)
        subject_idx.append(sid)
        weights_obs.append(wt)
        ages_obs.append(age)

# --- Run SCM ---
result = nextstat.scm(
    times,
    y,
    subject_idx,
    n_subjects=n_subjects,
    covariates=[weights_obs, ages_obs],
    covariate_names=["WT", "AGE"],
    dose=dose,
    bioavailability=bioav,
    error_model="additive",
    sigma=sigma,
    theta_init=[1.0, 10.0, 1.5],
    omega_init=[0.30, 0.30, 0.30],
    param_names=["CL", "V", "Ka"],
    relationships=["power", "exponential"],
    forward_alpha=0.05,
    backward_alpha=0.01,
)

# --- Print results ---
print("=" * 60)
print("SCM Results")
print("=" * 60)
print(f"Base OFV:    {result['base_ofv']:.2f}")
print(f"Final OFV:   {result['final_ofv']:.2f}")
print(f"OFV drop:    {result['base_ofv'] - result['final_ofv']:.2f}")
print(f"Forward steps:  {result['n_forward_steps']}")
print(f"Backward steps: {result['n_backward_steps']}")

print("\nSelected covariates:")
if not result["selected"]:
    print("  (none)")
for step in result["selected"]:
    print(
        f"  {step['name']:15s} ({step['relationship']:14s}): "
        f"delta_OFV = {step['delta_ofv']:8.2f}, "
        f"p = {step['p_value']:.6f}, "
        f"coeff = {step['coefficient']:.4f}"
    )

print("\nForward trace:")
for step in result["forward_trace"]:
    status = "ADDED" if step["included"] else "rejected"
    print(
        f"  {step['name']:15s}: "
        f"delta_OFV = {step['delta_ofv']:8.2f}, "
        f"p = {step['p_value']:.6f} -> {status}"
    )

if result["backward_trace"]:
    print("\nBackward trace:")
    for step in result["backward_trace"]:
        status = "KEPT" if step["included"] else "REMOVED"
        print(
            f"  {step['name']:15s}: "
            f"delta_OFV = {step['delta_ofv']:8.2f}, "
            f"p = {step['p_value']:.6f} -> {status}"
        )

print(f"\nFinal theta: {[f'{v:.4f}' for v in result['theta']]}")
```

### Expected output

With 30 subjects and a strong weight effect (exponent = 0.75), the SCM should:
1. Select `WT_on_CL` in forward selection (delta_OFV >> 3.84).
2. Reject `AGE` on all parameters (no true effect, delta_OFV < 3.84).
3. Retain `WT_on_CL` in backward elimination (delta_OFV >> 6.63).
4. Estimate a coefficient close to 0.75 for the weight-CL relationship.

---

## 9. Advanced configuration

### Custom FOCE settings

The FOCE inner/outer iteration limits and tolerance can be tuned:

```python
result = nextstat.scm(
    ...,
    max_outer_iter=200,    # more outer iterations for difficult models
    max_inner_iter=50,     # more inner iterations for complex eta landscapes
    tol=1e-6,              # tighter convergence
)
```

### Custom thresholds

Regulatory submissions may require different significance levels:

```python
# More conservative: harder to add AND harder to remove
result = nextstat.scm(
    ...,
    forward_alpha=0.01,    # delta_OFV > 6.63 to add
    backward_alpha=0.001,  # delta_OFV > 10.83 to keep
)

# More liberal: exploratory analysis
result = nextstat.scm(
    ...,
    forward_alpha=0.10,    # delta_OFV > 2.71 to add
    backward_alpha=0.05,   # delta_OFV > 3.84 to keep
)
```

### Single relationship type for all covariates

If `relationships` is not provided, all covariates default to the `"power"`
relationship:

```python
result = nextstat.scm(
    ...,
    covariates=[weights_obs, ages_obs, sexes_obs],
    covariate_names=["WT", "AGE", "SEX"],
    # relationships not specified => all use "power"
)
```

---

## 10. Comparison with NONMEM SCM

| Feature | NONMEM SCM | NextStat SCM |
|---------|-----------|--------------|
| Algorithm | Forward + backward | Forward + backward |
| Estimation | FOCE/FOCEI | FOCE with interaction |
| Thresholds | Configurable | Configurable (same defaults) |
| Centering | Manual | Automatic (median) |
| Relationship types | Power, linear, exponential | Power, proportional, exponential |
| Runtime | Minutes to hours (Fortran) | Seconds to minutes (compiled Rust) |
| License | Commercial | Open source |

### Migration from NONMEM

If you have a NONMEM SCM control stream, the mapping is:

| NONMEM | NextStat |
|--------|----------|
| `$COVARIATE TEST` | `covariates` + `covariate_names` |
| `FORWARD_CRITERIA` | `forward_alpha` |
| `BACKWARD_CRITERIA` | `backward_alpha` |
| `POWER` / `LINEAR` / `EXPONENTIAL` | `relationships=["power"]` / `"proportional"` / `"exponential"` |
| `THETA(x)` initial | `theta_init` |
| `OMEGA(x,x)` | `omega_init` |
