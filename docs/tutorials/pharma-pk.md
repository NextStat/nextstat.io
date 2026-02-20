---
title: "Pharmacometrics with NextStat: Population PK Tutorial"
description: "Step-by-step guide for pharmacometricians migrating from NONMEM/Monolix. Covers individual PK, population NLME (FOCE/SAEM), diagnostics, covariate modeling, survival analysis, and Bayesian PK."
status: stable
last_updated: 2026-02-21
---

# Pharmacometrics with NextStat: Population PK Tutorial

This tutorial walks through a complete pharmacometrics workflow using NextStat,
from raw NONMEM-format data to population PK estimation, diagnostics, covariate
analysis, and clinical endpoint evaluation. It is written for pharmacometricians
familiar with NONMEM or Monolix who want to evaluate NextStat as a compiled,
high-performance alternative.

## Table of contents

1. [Installation](#1-installation)
2. [Loading clinical data](#2-loading-clinical-data)
3. [Individual PK fitting](#3-individual-pk-fitting)
4. [Population PK (NLME)](#4-population-pk-nlme)
5. [Dosing regimens](#5-dosing-regimens)
6. [Model diagnostics](#6-model-diagnostics)
7. [Covariate analysis (SCM)](#7-covariate-analysis-scm)
8. [Bootstrap confidence intervals](#8-bootstrap-confidence-intervals)
9. [Survival analysis (clinical endpoints)](#9-survival-analysis-clinical-endpoints)
10. [Bayesian PK](#10-bayesian-pk)
11. [Performance comparison](#11-performance-comparison)

---

## 1. Installation

```bash
pip install nextstat
```

Verify the install:

```python
import nextstat
print(nextstat.__version__)
```

NextStat ships as a single pre-compiled wheel (PyO3/maturin). No Fortran
compiler, no NONMEM license, no external dependencies.

---

## 2. Loading clinical data

NextStat reads standard NONMEM-format CSV files via `read_nonmem()`. The
function expects a CSV **string** (not a file path), so you read the file
first.

### Required columns

| Column | Description |
|--------|-------------|
| `ID` | Subject identifier |
| `TIME` | Observation / dosing time |
| `DV` | Dependent variable (observed concentration) |

### Optional columns (with defaults)

| Column | Default | Description |
|--------|---------|-------------|
| `AMT` | 0 | Dose amount |
| `EVID` | Inferred (1 if AMT > 0, else 0) | Event ID |
| `MDV` | Inferred (1 if EVID != 0, else 0) | Missing DV flag |
| `CMT` | 1 | Compartment (1 = depot/oral, 2 = central/IV) |
| `RATE` | 0 | Infusion rate (> 0 means infusion, 0 means bolus) |

### Example

```python
import nextstat

csv_text = open("warfarin.csv").read()
data = nextstat.read_nonmem(csv_text)

print(f"Subjects: {data['n_subjects']}")
print(f"Subject IDs: {data['subject_ids']}")
print(f"Observation times: {data['times'][:10]}")
print(f"Observed concentrations: {data['dv'][:10]}")
print(f"Subject indices (0-based): {data['subject_idx'][:10]}")
```

The returned dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `n_subjects` | `int` | Number of unique subjects |
| `subject_ids` | `list[str]` | Unique subject IDs in order |
| `times` | `list[float]` | Observation times (EVID=0 rows only) |
| `dv` | `list[float]` | Observed concentrations (EVID=0 rows only) |
| `subject_idx` | `list[int]` | 0-based subject index per observation |

Dosing records (EVID=1) are automatically excluded from the observation
vectors. The `subject_idx` maps each observation to its subject, which is
the format expected by all population-level functions.

### EVID codes

| EVID | Meaning |
|------|---------|
| 0 | Observation |
| 1 | Dosing event |
| 2 | Other event (reset, etc.) |
| 3 | Reset |
| 4 | Reset and dose |

`read_nonmem()` filters to EVID=0 observations with MDV=0. Dose records
are retained internally for dosing regimen construction.

### Handling missing DV

Rows where `DV` is `.` (NONMEM missing-value convention) are treated as
MDV=1 and excluded from the observation vector.

---

## 3. Individual PK fitting

NextStat provides analytical PK models with closed-form solutions and
analytical gradients. No ODE solver overhead.

### 1-compartment oral

Three parameters: CL (clearance, L/h), V (volume, L), Ka (absorption rate
constant, 1/h).

```python
import nextstat

# Theophylline-like single-subject data
times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
conc  = [2.8, 5.1, 7.3, 8.6, 6.5, 3.2, 1.5, 0.2]

model = nextstat.OneCompartmentOralPkModel(
    times, conc,
    dose=320.0,
    bioavailability=1.0,
    sigma=0.5,             # additive error SD
)

# Fit by maximum likelihood
result = nextstat.fit(model)

# Extract estimates
params = dict(zip(model.parameter_names(), result.bestfit))
print(f"CL = {params['CL']:.4f} L/h")
print(f"V  = {params['V']:.4f} L")
print(f"Ka = {params['Ka']:.4f} 1/h")
print(f"NLL = {result.nll:.2f}")
print(f"Converged: {result.converged}")
```

### 2-compartment IV bolus

Four parameters: CL, V1 (central volume), V2 (peripheral volume), Q
(intercompartmental clearance).

```python
model = nextstat.TwoCompartmentIvPkModel(
    times, conc,
    dose=500.0,
    error_model="proportional",  # y = f * (1 + eps)
    sigma=0.1,                   # proportional error CV = 10%
)

result = nextstat.fit(model)
params = dict(zip(model.parameter_names(), result.bestfit))
print(f"CL = {params['CL']:.4f}, V1 = {params['V1']:.4f}, V2 = {params['V2']:.4f}, Q = {params['Q']:.4f}")
```

### 2-compartment oral

Five parameters: CL, V1, V2, Q, Ka.

```python
model = nextstat.TwoCompartmentOralPkModel(
    times, conc,
    dose=200.0,
    bioavailability=0.85,
    error_model="combined",   # y = f*(1 + eps1) + eps2
    sigma=0.1,                # proportional component
    sigma_add=0.5,            # additive component
)

result = nextstat.fit(model)
```

### Error models

| Model | NONMEM equivalent | Formula | NextStat parameter |
|-------|-------------------|---------|-------------------|
| Additive | `Y = F + EPS(1)` | Var = sigma^2 | `sigma=0.5` |
| Proportional | `Y = F*(1+EPS(1))` | Var = (sigma*F)^2 | `error_model="proportional", sigma=0.1` |
| Combined | `Y = F*(1+EPS(1)) + EPS(2)` | Var = sigma_add^2 + (sigma*F)^2 | `error_model="combined", sigma=0.1, sigma_add=0.5` |

### LLOQ handling

Below-limit-of-quantification observations are handled via `lloq_policy`:

| Policy | Behavior | NONMEM equivalent |
|--------|----------|-------------------|
| `"ignore"` | Drop observations below LLOQ | `IGNORE=(DV.LT.LLOQ)` |
| `"replace_half"` | Replace with LLOQ/2 | M1 method |
| `"censored"` | Left-censored likelihood | M3 method (recommended) |

The `"censored"` policy (M3 method) is the most statistically appropriate and
is recommended for regulatory submissions.

```python
model = nextstat.OneCompartmentOralPkModel(
    times, conc,
    dose=320.0,
    sigma=0.5,
    lloq=0.05,                # LLOQ = 0.05 mg/L
    lloq_policy="censored",   # M3 method
)
```

### Predicted concentrations

All PK models expose a `predict()` method for computing concentrations at
the fitted parameters:

```python
result = nextstat.fit(model)
predicted = model.predict(result.bestfit)

for t, obs, pred in zip(times, conc, predicted):
    print(f"  t={t:5.1f}h  DV={obs:.2f}  PRED={pred:.2f}")
```

---

## 4. Population PK (NLME)

### FOCE estimation

First-Order Conditional Estimation with Interaction (FOCEI) is the industry
standard. NextStat implements it with analytical gradients and L-BFGS-B
optimization.

```python
import nextstat

# Load NONMEM dataset (32 subjects, warfarin study)
csv_text = open("warfarin_pop.csv").read()
data = nextstat.read_nonmem(csv_text)

# FOCE/FOCEI estimation
result = nextstat.nlme_foce(
    data["times"],
    data["dv"],
    data["subject_idx"],
    data["n_subjects"],
    dose=100.0,
    bioavailability=1.0,
    error_model="proportional",
    sigma=0.1,
    theta_init=[0.133, 8.0, 0.8],   # [CL, V, Ka] initial estimates
    omega_init=[0.30, 0.25, 0.30],   # [omega_CL, omega_V, omega_Ka]
    max_outer_iter=100,
    max_inner_iter=20,
    tol=1e-4,
    interaction=True,   # FOCEI (recommended)
)

print(f"Population parameters (theta): {result['theta']}")
print(f"Random effect SDs (omega):     {result['omega']}")
print(f"OFV (-2LL): {result['ofv']:.2f}")
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['n_iter']}")
```

#### FOCE result keys

| Key | Type | Description |
|-----|------|-------------|
| `theta` | `list[float]` | Population fixed effects [CL, V, Ka] |
| `omega` | `list[float]` | Random effect standard deviations |
| `omega_matrix` | `list[list[float]]` | Full Omega variance-covariance matrix |
| `correlation` | `list[list[float]]` | Correlation matrix derived from Omega |
| `eta` | `list[list[float]]` | Conditional modes (EBEs) per subject |
| `ofv` | `float` | Objective function value (-2 log-likelihood) |
| `converged` | `bool` | Convergence flag |
| `n_iter` | `int` | Number of outer iterations |

#### Interpreting individual parameters

The individual PK parameters are log-normally distributed:

```python
import math

theta = result["theta"]   # [CL_pop, V_pop, Ka_pop]
eta = result["eta"]        # n_subjects x 3

for i in range(min(5, data["n_subjects"])):
    cl_i = theta[0] * math.exp(eta[i][0])
    v_i  = theta[1] * math.exp(eta[i][1])
    ka_i = theta[2] * math.exp(eta[i][2])
    print(f"Subject {i}: CL={cl_i:.4f} V={v_i:.4f} Ka={ka_i:.4f}")
```

### SAEM estimation

Stochastic Approximation EM (SAEM) is a Monolix-class algorithm that is more
robust than FOCE for complex nonlinear models with flat likelihoods.

```python
result = nextstat.nlme_saem(
    data["times"],
    data["dv"],
    data["subject_idx"],
    data["n_subjects"],
    dose=100.0,
    bioavailability=1.0,
    error_model="proportional",
    sigma=0.1,
    theta_init=[0.133, 8.0, 0.8],
    omega_init=[0.30, 0.25, 0.30],
    n_burn=200,       # burn-in iterations (step size gamma = 1)
    n_iter=100,       # estimation iterations (step size gamma = 1/k)
    n_chains=1,       # MCMC chains per subject
    seed=12345,
    tol=1e-4,
)

# SAEM returns the same keys as FOCE, plus a "saem" sub-dict
print(f"theta: {result['theta']}")
print(f"OFV: {result['ofv']:.2f}")

# SAEM-specific diagnostics
saem_diag = result["saem"]
print(f"Acceptance rates: {saem_diag['acceptance_rates'][:5]}")
print(f"OFV trace length: {len(saem_diag['ofv_trace'])}")
```

### When to use FOCE vs SAEM

| Criterion | FOCE/FOCEI | SAEM |
|-----------|-----------|------|
| Speed | Faster (deterministic) | Slower (stochastic) |
| Convergence | Can fail for complex models | More robust |
| Diagnostics | OFV only | OFV trace + acceptance rates |
| Regulatory | Gold standard (NONMEM) | Accepted (Monolix, nlmixr2) |
| Best for | Standard PK, well-behaved data | PD/ODE models, flat likelihoods |

**Recommendation**: Start with FOCEI. Switch to SAEM if convergence fails or
the objective function surface is multimodal.

---

## 5. Dosing regimens

The Rust-level `DosingRegimen` API supports complex multi-dose schedules.
The Python API exposes PK model constructors that accept a single dose value
for the common case. For multi-dose simulation, use the `predict()` method
on fitted models.

### Single oral dose

```python
model = nextstat.OneCompartmentOralPkModel(
    times, conc,
    dose=100.0,
    bioavailability=1.0,
    sigma=0.05,
)
result = nextstat.fit(model)
predicted = model.predict(result.bestfit)
```

### Single IV bolus

```python
model = nextstat.TwoCompartmentIvPkModel(
    times, conc,
    dose=500.0,
    error_model="additive",
    sigma=0.05,
)
```

### Simulation with fitted parameters

After fitting a population model, you can predict individual concentration
profiles using the structural model:

```python
import math

theta = result["theta"]  # [CL, V, Ka]
eta_i = result["eta"][0]  # first subject's EBEs

# Individual parameters
cl = theta[0] * math.exp(eta_i[0])
v  = theta[1] * math.exp(eta_i[1])
ka = theta[2] * math.exp(eta_i[2])

# Predict at dense time grid
dose = 100.0
f = 1.0  # bioavailability
ke = cl / v
dense_times = [t * 0.5 for t in range(97)]  # 0, 0.5, 1.0, ..., 48.0

concentrations = []
for t in dense_times:
    if t == 0:
        concentrations.append(0.0)
    else:
        c = (f * dose * ka / (v * (ka - ke))) * (math.exp(-ke * t) - math.exp(-ka * t))
        concentrations.append(max(c, 0.0))
```

---

## 6. Model diagnostics

### Goodness of fit

`pk_gof()` computes per-observation diagnostic quantities using the fitted
population parameters and individual ETAs.

```python
gof = nextstat.pk_gof(
    data["times"],
    data["dv"],
    data["subject_idx"],
    dose=100.0,
    bioavailability=1.0,
    theta=result["theta"],
    eta=result["eta"],
    error_model="proportional",
    sigma=0.1,
)

# gof is a list of per-observation dicts
for rec in gof[:5]:
    print(
        f"Subject {rec['subject']}  "
        f"t={rec['time']:5.1f}  "
        f"DV={rec['dv']:.2f}  "
        f"PRED={rec['pred']:.2f}  "
        f"IPRED={rec['ipred']:.2f}  "
        f"IWRES={rec['iwres']:.3f}  "
        f"CWRES={rec['cwres']:.3f}"
    )
```

#### Diagnostic quantities

| Quantity | Formula | Interpretation |
|----------|---------|---------------|
| PRED | C(t; theta_pop, eta=0) | Population prediction |
| IPRED | C(t; theta_pop, eta_hat_i) | Individual prediction |
| IWRES | (DV - IPRED) / sigma(IPRED) | Individual weighted residual |
| CWRES | FOCE-based conditional residual | Should be approximately N(0,1) |

#### Standard GOF plot interpretation

| Plot | What to look for |
|------|-----------------|
| DV vs PRED | Systematic bias indicates misspecified structural model |
| DV vs IPRED | Scatter around identity line; poor fit indicates inadequate random effects |
| IWRES vs TIME | Trending pattern indicates time-dependent bias |
| CWRES vs PRED | Heteroscedasticity indicates wrong error model |
| QQ-plot of CWRES | Departure from N(0,1) indicates model misspecification |

### Visual Predictive Check

`pk_vpc()` simulates replicate datasets from the fitted model and computes
prediction interval bands.

```python
vpc = nextstat.pk_vpc(
    data["times"],
    data["dv"],
    data["subject_idx"],
    data["n_subjects"],
    dose=100.0,
    bioavailability=1.0,
    theta=result["theta"],
    omega_matrix=result["omega_matrix"],
    error_model="proportional",
    sigma=0.1,
    n_sim=200,
    quantiles=[0.05, 0.50, 0.95],
    n_bins=10,
    seed=42,
    pi_level=0.90,
)

print(f"Number of simulation replicates: {vpc['n_sim']}")
print(f"Quantiles: {vpc['quantiles']}")

for b in vpc["bins"]:
    print(
        f"Time bin [{b['time_lo']:.1f}, {b['time_hi']:.1f}]: "
        f"obs_median={b['observed_quantiles'][1]:.2f}  "
        f"sim_PI=[{b['simulated_pi_lo'][1]:.2f}, {b['simulated_pi_hi'][1]:.2f}]"
    )
```

#### VPC interpretation

- **Observed median inside simulated prediction interval**: Good model fit.
- **Observed 5th/95th percentile outside PI**: Variability is misspecified
  (check Omega, error model).
- **Systematic shift in all bins**: Structural model or covariate effect is
  missing.

---

## 7. Covariate analysis (SCM)

Stepwise Covariate Modeling (SCM) performs forward selection followed by
backward elimination of covariate-parameter relationships, using the
likelihood ratio test (chi-squared with 1 df).

```python
# Suppose we have per-observation covariate vectors
# (nextstat.scm extracts per-subject values from the first observation)
weights = [70.5, 82.3, 65.0, ...]   # one per observation
ages    = [45.0, 62.0, 31.0, ...]   # one per observation

scm_result = nextstat.scm(
    data["times"],
    data["dv"],
    data["subject_idx"],
    data["n_subjects"],
    covariates=[weights, ages],
    covariate_names=["WT", "AGE"],
    dose=100.0,
    bioavailability=1.0,
    error_model="proportional",
    sigma=0.1,
    theta_init=[0.133, 8.0, 0.8],
    omega_init=[0.30, 0.25, 0.30],
    param_names=["CL", "V", "Ka"],
    relationships=["power", "exponential"],  # per-covariate relationship type
    forward_alpha=0.05,    # p < 0.05 to add (delta_OFV > 3.84)
    backward_alpha=0.01,   # p < 0.01 to keep (delta_OFV > 6.63)
    max_outer_iter=100,
    max_inner_iter=20,
    tol=1e-4,
)
```

#### Interpreting SCM results

```python
print(f"Base OFV: {scm_result['base_ofv']:.2f}")
print(f"Final OFV: {scm_result['final_ofv']:.2f}")
print(f"Forward steps: {scm_result['n_forward_steps']}")
print(f"Backward steps: {scm_result['n_backward_steps']}")

print("\nSelected covariates:")
for step in scm_result["selected"]:
    print(
        f"  {step['name']} on param {step['param_index']} "
        f"({step['relationship']}): "
        f"delta_OFV={step['delta_ofv']:.2f}, "
        f"p={step['p_value']:.4f}, "
        f"coefficient={step['coefficient']:.4f}"
    )

print("\nForward trace:")
for step in scm_result["forward_trace"]:
    status = "added" if step["included"] else "rejected"
    print(f"  {step['name']}: delta_OFV={step['delta_ofv']:.2f}, p={step['p_value']:.4f} -> {status}")
```

#### Covariate relationship types

| Type | Formula | Typical use |
|------|---------|-------------|
| `"power"` | theta_i = theta * (COV / COV_median)^beta | Weight on CL (allometric) |
| `"exponential"` | theta_i = theta * exp(beta * (COV - COV_median)) | Age, lab values |
| `"proportional"` | theta_i = theta * (1 + beta * (COV - COV_median)) | Linear relationships |

#### SCM thresholds

| Phase | alpha | delta_OFV (chi-squared, 1 df) | Purpose |
|-------|-------|-------------------------------|---------|
| Forward | 0.05 | > 3.84 | Adding covariates |
| Backward | 0.01 | > 6.63 | Removing covariates |

The more stringent backward threshold guards against overfitting.

---

## 8. Bootstrap confidence intervals

NextStat uses Rayon-parallelized bootstrap resampling for parameter
uncertainty quantification. Each bootstrap replicate re-fits the full
population model with FOCE.

```python
import nextstat
import random

n_bootstrap = 1000
seed = 42
rng = random.Random(seed)

# Collect subject-level data indices
n_subj = data["n_subjects"]
subj_indices = {}
for i, s in enumerate(data["subject_idx"]):
    subj_indices.setdefault(s, []).append(i)

bootstrap_thetas = []

for b in range(n_bootstrap):
    # Resample subjects with replacement
    sampled_subjects = [rng.randint(0, n_subj - 1) for _ in range(n_subj)]

    # Build resampled dataset
    new_times, new_dv, new_subj_idx = [], [], []
    for new_id, orig_id in enumerate(sampled_subjects):
        for idx in subj_indices[orig_id]:
            new_times.append(data["times"][idx])
            new_dv.append(data["dv"][idx])
            new_subj_idx.append(new_id)

    try:
        boot_result = nextstat.nlme_foce(
            new_times, new_dv, new_subj_idx, n_subj,
            dose=100.0,
            error_model="proportional",
            sigma=0.1,
            theta_init=result["theta"],   # warm-start from original fit
            omega_init=result["omega"],
            max_outer_iter=50,
            tol=1e-3,
            interaction=True,
        )
        if boot_result["converged"]:
            bootstrap_thetas.append(boot_result["theta"])
    except Exception:
        pass  # skip failed resamples

print(f"Converged: {len(bootstrap_thetas)}/{n_bootstrap}")

# Compute 95% percentile CIs
if bootstrap_thetas:
    for j, name in enumerate(["CL", "V", "Ka"]):
        vals = sorted(t[j] for t in bootstrap_thetas)
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals))]
        print(f"{name}: point={result['theta'][j]:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")
```

Because each FOCE fit takes roughly 50 ms for a 32-subject dataset, 1000
bootstrap replicates complete in approximately 30-60 seconds with Rayon
parallelism.

---

## 9. Survival analysis (clinical endpoints)

NextStat provides survival analysis functions directly relevant to clinical
trial endpoints (time-to-event data).

### Kaplan-Meier estimator

```python
import nextstat

# Clinical trial endpoint: time to disease progression
times  = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0, 30.0,
          4.0, 8.0, 11.0, 14.0, 16.0, 22.0, 25.0, 28.0, 32.0, 36.0]
events = [True, True, False, True, True, False, True, False, True, True,
          True, True, True, False, True, True, False, True, True, False]

km = nextstat.kaplan_meier(times, events, conf_level=0.95)

print(f"N = {len(times)}, Events = {sum(events)}")
for i in range(len(km["times"])):
    print(
        f"t={km['times'][i]:5.1f}  "
        f"S(t)={km['survival'][i]:.3f}  "
        f"95% CI=[{km['ci_lower'][i]:.3f}, {km['ci_upper'][i]:.3f}]  "
        f"at_risk={km['at_risk'][i]}"
    )
```

### Log-rank test

Compare survival between treatment groups (e.g., drug vs placebo).

```python
# groups: 0 = control, 1 = treatment
groups = [0]*10 + [1]*10

lr = nextstat.log_rank_test(times, events, groups)
print(f"Chi-squared = {lr['statistic']:.3f}")
print(f"df = {lr['df']}")
print(f"p-value = {lr['p_value']:.4f}")

if lr["p_value"] < 0.05:
    print("Significant difference in survival between groups.")
```

### Cox proportional hazards

Semi-parametric regression for treatment effect estimation with covariates.

```python
from nextstat.survival import cox_ph

# Covariates: treatment (0/1), age, baseline biomarker
x = [
    [0, 45.0, 2.1], [0, 52.0, 3.4], [0, 38.0, 1.8], [0, 61.0, 4.2], [0, 55.0, 2.9],
    [0, 48.0, 3.1], [0, 42.0, 2.5], [0, 67.0, 5.0], [0, 50.0, 3.3], [0, 44.0, 2.0],
    [1, 47.0, 2.3], [1, 53.0, 3.6], [1, 39.0, 1.9], [1, 60.0, 4.0], [1, 54.0, 2.8],
    [1, 46.0, 3.0], [1, 41.0, 2.4], [1, 65.0, 4.8], [1, 51.0, 3.2], [1, 43.0, 2.2],
]

fit = cox_ph.fit(
    times, events, x,
    ties="efron",
    robust=True,
    compute_cov=True,
    compute_baseline=True,
)

print(f"Coefficients: {fit.coef}")
print(f"Hazard ratios: {fit.hazard_ratios()}")
print(f"Robust SE: {fit.robust_se}")

# Hazard ratio confidence intervals
hr_ci = fit.hazard_ratio_confint(level=0.95, robust=True)
labels = ["Treatment", "Age", "Biomarker"]
for name, hr, (lo, hi) in zip(labels, fit.hazard_ratios(), hr_ci):
    print(f"  {name}: HR={hr:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")
```

### Cluster-robust standard errors

For multi-center trials, use `groups=` for cluster-robust sandwich SE:

```python
# cluster_ids: site/center identifiers per subject
cluster_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0]

fit = cox_ph.fit(
    times, events, x,
    ties="efron",
    robust=True,
    groups=cluster_ids,
    cluster_correction=True,
)
print(f"Cluster-robust SE: {fit.robust_se}")
print(f"Robust kind: {fit.robust_kind}")  # "cluster"
```

### Proportional hazards test

Verify the PH assumption using Schoenfeld residuals:

```python
from nextstat.survival import cox_ph_ph_test

ph_tests = cox_ph_ph_test(times, events, x, ties="efron")
for test in ph_tests:
    status = "VIOLATION" if test["p"] < 0.05 else "ok"
    print(f"Feature {test['feature']}: slope={test['slope']:.4f}, p={test['p']:.4f} [{status}]")
```

### Parametric survival models

For parametric time-to-event analysis:

```python
from nextstat.survival import weibull, lognormal_aft, exponential

# Weibull model
wb_fit = weibull.fit(times, events)
print(f"Weibull params: {wb_fit.params}, NLL: {wb_fit.nll:.2f}")

# Log-normal AFT
ln_fit = lognormal_aft.fit(times, events)
print(f"LogNormal AFT params: {ln_fit.params}, NLL: {ln_fit.nll:.2f}")

# Compare by AIC (2*NLL + 2*k)
aic_wb = 2 * wb_fit.nll + 2 * len(wb_fit.params)
aic_ln = 2 * ln_fit.nll + 2 * len(ln_fit.params)
print(f"AIC Weibull: {aic_wb:.2f}, AIC LogNormal: {aic_ln:.2f}")
```

---

## 10. Bayesian PK

For posterior inference on PK parameters, use NUTS sampling with a PK model
as the likelihood.

```python
import nextstat

# Define model (acts as the likelihood)
model = nextstat.OneCompartmentOralPkModel(
    times, conc,
    dose=320.0,
    sigma=0.5,
)

# NUTS sampling (no prior specification needed for MLE-type models;
# the model's suggested_bounds act as implicit uniform priors)
result = nextstat.sample(
    model,
    method="nuts",
    n_chains=4,
    n_warmup=500,
    n_samples=1000,
    seed=42,
)

# Posterior summary
posterior = result["posterior"]
for name in result["param_names"]:
    samples = posterior[name]
    # Flatten across chains
    all_samples = [s for chain in samples for s in chain]
    mean = sum(all_samples) / len(all_samples)
    sorted_s = sorted(all_samples)
    lo = sorted_s[int(0.025 * len(sorted_s))]
    hi = sorted_s[int(0.975 * len(sorted_s))]
    print(f"{name}: mean={mean:.4f}, 95% CrI=[{lo:.4f}, {hi:.4f}]")

# Check diagnostics
diag = result["diagnostics"]
print(f"Quality: {diag['quality']['status']}")
```

For ArviZ integration (trace plots, ESS, R-hat):

```python
idata = nextstat.sample(
    model,
    method="nuts",
    n_samples=2000,
    return_idata=True,
)

import arviz as az
print(az.summary(idata))
az.plot_trace(idata)
```

---

## 11. Performance comparison

NextStat compiles PK models to native Rust code with analytical gradients,
zero-allocation optimizer loops, and Rayon parallelism. The following
benchmarks are representative of typical pharmacometrics workloads.

| Task | NextStat | NONMEM 7.5 | Speedup |
|------|----------|------------|---------|
| Single 1-cpt MLE fit (10 obs) | < 1 ms | 5-30 s | 10,000-60,000x |
| Pop PK FOCE (32 subjects) | ~50 ms | 30-120 s | 600-2,400x |
| Pop PK FOCE (100 subjects) | ~200 ms | 2-10 min | 600-3,000x |
| VPC (200 simulations, 32 subj) | ~5 ms | 10-60 s | 2,000-12,000x |
| GOF diagnostics (32 subjects) | < 1 ms | seconds | 1,000x+ |
| SCM (6 covariates, 3 params) | ~3 s | 30-60 min | 600-1,200x |
| Bootstrap 1,000 replicates | ~30 s | 3-8 hours | 360-960x |

The speedup comes from:
- **Analytical gradients**: Closed-form derivatives via eigenvalue decomposition,
  not finite differences.
- **L-BFGS-B optimizer**: Compiled Rust optimizer with warm-starting and
  bounds clamping, not the NONMEM THETA search.
- **Rayon parallelism**: Bootstrap and toy fits scale linearly across CPU cores.
- **No I/O overhead**: In-memory operation, no file-based IPC.

---

## Related tutorials

- [NONMEM Migration Guide](nonmem-migration.md) -- side-by-side NONMEM/NextStat comparison
- [Survival Analysis for Clinical Biostatisticians](pharma-survival.md) -- clinical endpoints
- [PK Modeling (low-level)](phase-13-pk.md) -- Rust API, dosing regimens, NONMEM reader
- [Population PK with NLME](phase-13-nlme.md) -- FOCE/SAEM internals, correlated Omega
