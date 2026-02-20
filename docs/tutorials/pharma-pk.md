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
12. [3-compartment PK models](#12-3-compartment-pk-models)
13. [Bioequivalence analysis](#13-bioequivalence-analysis)
14. [Clinical trial simulation](#14-clinical-trial-simulation)
15. [MAP estimation](#15-map-estimation)
16. [CDISC XPT I/O](#16-cdisc-xpt-io)

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

## 12. 3-compartment PK models

For drugs with complex distribution kinetics (e.g., vancomycin, aminoglycosides,
large-molecule biologics), a 2-compartment model may not capture the terminal
elimination phase. NextStat provides 3-compartment models with analytical
solutions and gradients for both IV and oral administration.

### 3-compartment IV bolus

Six parameters: CL (clearance, L/h), V1 (central volume, L), Q2 (intercompartmental
clearance to peripheral 1, L/h), V2 (peripheral volume 1, L), Q3
(intercompartmental clearance to peripheral 2, L/h), V3 (peripheral volume 2, L).

```python
import nextstat

# Vancomycin-like 3-compartment IV data (single subject)
times = [0.083, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0]
conc  = [42.0, 35.5, 28.1, 20.3, 14.8, 11.2, 9.6, 8.4, 6.5, 4.2, 2.8, 1.1, 0.4]

model = nextstat.ThreeCompartmentIvPkModel(
    times, conc,
    dose=1000.0,            # 1 g IV bolus
    error_model="proportional",
    sigma=0.10,             # 10% proportional error
)

result = nextstat.fit(model)
params = dict(zip(model.parameter_names(), result.bestfit))
print(f"CL  = {params['CL']:.4f} L/h")
print(f"V1  = {params['V1']:.4f} L")
print(f"Q2  = {params['Q2']:.4f} L/h")
print(f"V2  = {params['V2']:.4f} L")
print(f"Q3  = {params['Q3']:.4f} L/h")
print(f"V3  = {params['V3']:.4f} L")
print(f"NLL = {result.nll:.2f}")
print(f"Converged: {result.converged}")

# Predicted concentrations
predicted = model.predict(result.bestfit)
for t, obs, pred in zip(times, conc, predicted):
    print(f"  t={t:6.3f}h  DV={obs:6.2f}  PRED={pred:6.2f}")
```

### 3-compartment oral

Seven parameters: CL, V1, Q2, V2, Q3, V3, Ka (absorption rate constant, 1/h).

```python
# Aminoglycoside-like oral absorption with deep tissue distribution
times = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0]
conc  = [1.2, 4.5, 7.8, 9.1, 8.3, 7.0, 5.1, 3.9, 2.5, 1.3, 0.7, 0.2, 0.05]

model = nextstat.ThreeCompartmentOralPkModel(
    times, conc,
    dose=500.0,
    bioavailability=0.70,
    error_model="combined",
    sigma=0.08,             # proportional component
    sigma_add=0.1,          # additive component (mg/L)
    lloq=0.05,
    lloq_policy="censored", # M3 method for BLQ
)

result = nextstat.fit(model)
params = dict(zip(model.parameter_names(), result.bestfit))
print(f"CL  = {params['CL']:.4f} L/h")
print(f"V1  = {params['V1']:.4f} L")
print(f"Q2  = {params['Q2']:.4f} L/h")
print(f"V2  = {params['V2']:.4f} L")
print(f"Q3  = {params['Q3']:.4f} L/h")
print(f"V3  = {params['V3']:.4f} L")
print(f"Ka  = {params['Ka']:.4f} 1/h")
```

### 3-compartment parameter summary

| Parameter | Units | Typical range (vancomycin) | Description |
|-----------|-------|---------------------------|-------------|
| CL | L/h | 2-6 | Total body clearance |
| V1 | L | 15-30 | Central compartment volume |
| Q2 | L/h | 5-15 | Intercompartmental clearance (shallow) |
| V2 | L | 20-60 | Shallow peripheral volume |
| Q3 | L/h | 1-5 | Intercompartmental clearance (deep) |
| V3 | L | 30-80 | Deep peripheral volume |
| Ka | 1/h | 0.5-3.0 | Absorption rate (oral only) |

### When to use 3-compartment models

- **Early rapid distribution** followed by a secondary redistribution phase
  that a 2-compartment model cannot capture.
- **Aminoglycosides, vancomycin**: well-established 3-compartment PK in the
  literature.
- **Monoclonal antibodies**: often require 3-compartment models for target-mediated
  drug disposition (TMDD) approximation.
- **Model selection**: compare 2-cpt vs 3-cpt using AIC = 2*NLL + 2*k. The
  3-compartment model adds 2 extra parameters (Q3, V3), so it must reduce
  NLL by at least 2 to justify the complexity.

---

## 13. Bioequivalence analysis

NextStat provides functions for average bioequivalence (ABE) assessment, the
standard regulatory approach for generic drug approval. The functions implement
the two one-sided tests (TOST) procedure on log-transformed PK parameters
(AUC, Cmax).

### Average bioequivalence test

`average_be()` performs the TOST analysis. It expects **log-transformed** values
(i.e., `math.log(AUC)` or `math.log(Cmax)`, not the raw values).

```python
import nextstat
import math

# Example 2x2 crossover bioequivalence study
# AUC values (ng*h/mL) from 24 subjects
test_auc = [485.2, 512.7, 398.1, 601.3, 445.8, 523.6,
            478.9, 550.1, 410.5, 492.3, 538.7, 467.2,
            505.4, 489.6, 430.1, 560.8, 475.3, 510.2,
            498.7, 542.1, 418.9, 530.5, 462.7, 515.3]

ref_auc  = [502.1, 498.3, 415.6, 588.9, 462.1, 535.2,
            495.7, 538.4, 425.3, 508.7, 520.9, 480.5,
            518.2, 476.8, 445.7, 572.3, 490.1, 522.8,
            510.4, 528.6, 432.1, 545.3, 478.9, 500.7]

# Log-transform the data (REQUIRED)
test_log = [math.log(x) for x in test_auc]
ref_log  = [math.log(x) for x in ref_auc]

be = nextstat.average_be(
    test_log,
    ref_log,
    alpha=0.05,
    limits=(0.80, 1.25),   # standard FDA/EMA limits
    design="2x2",          # 2-sequence, 2-period crossover
)

print(f"Geometric Mean Ratio: {be['geometric_mean_ratio']:.4f}")
print(f"90% CI: [{be['ci_lower']:.4f}, {be['ci_upper']:.4f}]")
print(f"Point estimate (log): {be['pe_log']:.6f}")
print(f"Standard error (log): {be['se_log']:.6f}")
print(f"Degrees of freedom: {be['df']}")
print(f"t-statistics: lower={be['t_lower']:.4f}, upper={be['t_upper']:.4f}")
print(f"p-values: lower={be['p_lower']:.6f}, upper={be['p_upper']:.6f}")
print(f"Conclusion: {be['conclusion']}")
```

#### Interpreting the result

| Field | Description |
|-------|-------------|
| `geometric_mean_ratio` | Exp(mean(log(test)) - mean(log(ref))); target is near 1.0 |
| `ci_lower`, `ci_upper` | 90% CI for the GMR; must fall within `limits` |
| `pe_log` | Point estimate on the log scale |
| `se_log` | Standard error on the log scale |
| `df` | Degrees of freedom (design-dependent) |
| `conclusion` | `"bioequivalent"` or `"not bioequivalent"` |

Bioequivalence is concluded when the 90% CI for the geometric mean ratio falls
entirely within [0.80, 1.25].

### Power analysis

Before running a bioequivalence study, compute the statistical power for a
given sample size and assumed intra-subject CV.

```python
# What power do we have with 24 total subjects?
power = nextstat.be_power(
    n_total=24,
    cv=0.30,          # 30% intra-subject coefficient of variation
    gmr=0.95,         # assumed true geometric mean ratio
    alpha=0.05,
    design="2x2",
)
print(f"Power with n=24: {power:.4f}")  # e.g., 0.82
```

### Sample size calculation

Determine the minimum sample size needed to achieve a target power.

```python
ss = nextstat.be_sample_size(
    cv=0.30,
    gmr=0.95,
    target_power=0.80,
    alpha=0.05,
    design="2x2",
)
print(f"Required per sequence: {ss['n_per_sequence']}")
print(f"Total subjects: {ss['n_total']}")
print(f"Achieved power: {ss['achieved_power']:.4f}")
```

#### Typical CV values by drug class

| Drug class | Typical intra-subject CV | Notes |
|------------|--------------------------|-------|
| Immediate-release oral | 15-30% | Standard BE studies |
| Modified-release oral | 20-40% | May need larger N |
| Highly variable drugs | > 30% | Consider reference-scaled ABE |
| Narrow therapeutic index | 10-20% | Tighter limits may apply (0.90-1.11) |

---

## 14. Clinical trial simulation

`simulate_trial()` generates virtual clinical trial data with between-subject
variability (BSV) and residual error, useful for trial design, dose finding,
and power analysis.

```python
import nextstat

# Simulate a 1-compartment oral PK trial
sim = nextstat.simulate_trial(
    n_subjects=50,
    dose=100.0,
    obs_times=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
    pk_model="1cpt_oral",
    theta=[2.5, 25.0, 1.2],           # [CL, V, Ka] population means
    omega=[0.30, 0.25, 0.40],         # BSV (SD on log scale) for [CL, V, Ka]
    sigma=0.10,                        # residual error
    error_model="proportional",
    bioavailability=1.0,
    seed=42,
)

# Simulated concentrations: n_subjects x len(obs_times)
print(f"Subjects simulated: {len(sim['concentrations'])}")
print(f"Observations per subject: {len(sim['concentrations'][0])}")

# NCA-like summary statistics from the simulation
print(f"\nPopulation PK summary (first 10 subjects):")
for i in range(10):
    print(
        f"  Subject {i:2d}: "
        f"Cmax={sim['cmax'][i]:6.2f} mg/L  "
        f"Tmax={sim['tmax'][i]:4.1f} h  "
        f"AUC={sim['auc'][i]:7.1f} mg*h/L  "
        f"Ctrough={sim['ctrough'][i]:5.2f} mg/L"
    )

# Individual PK parameters (with BSV applied)
print(f"\nIndividual parameters (first 5 subjects):")
for i in range(5):
    p = sim['individual_params'][i]
    print(f"  Subject {i}: CL={p[0]:.3f} L/h, V={p[1]:.2f} L, Ka={p[2]:.3f} 1/h")
```

### Simulation result fields

| Field | Type | Description |
|-------|------|-------------|
| `concentrations` | `list[list[float]]` | n_subjects x n_times concentration matrix |
| `individual_params` | `list[list[float]]` | n_subjects x n_params individual parameters |
| `auc` | `list[float]` | AUC(0-tlast) per subject (trapezoidal) |
| `cmax` | `list[float]` | Maximum concentration per subject |
| `tmax` | `list[float]` | Time of Cmax per subject |
| `ctrough` | `list[float]` | Trough concentration (last time point) per subject |

### Dose-finding simulation

Use `simulate_trial()` to compare exposure across dose levels:

```python
import nextstat

doses = [50.0, 100.0, 200.0, 400.0]
target_auc = 500.0  # target AUC for therapeutic window

for dose in doses:
    sim = nextstat.simulate_trial(
        n_subjects=100,
        dose=dose,
        obs_times=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
        pk_model="1cpt_oral",
        theta=[2.5, 25.0, 1.2],
        omega=[0.30, 0.25, 0.40],
        sigma=0.10,
        error_model="proportional",
        seed=42,
    )

    aucs = sim["auc"]
    mean_auc = sum(aucs) / len(aucs)
    sorted_auc = sorted(aucs)
    auc_5  = sorted_auc[int(0.05 * len(sorted_auc))]
    auc_95 = sorted_auc[int(0.95 * len(sorted_auc))]
    pct_in_target = sum(1 for a in aucs if 0.5 * target_auc <= a <= 2.0 * target_auc) / len(aucs)

    print(
        f"Dose {dose:5.0f} mg: "
        f"mean AUC={mean_auc:7.1f}, "
        f"90% PI=[{auc_5:.1f}, {auc_95:.1f}], "
        f"% in target={pct_in_target*100:.0f}%"
    )
```

---

## 15. MAP estimation

Maximum A Posteriori (MAP) estimation combines individual patient data with
population priors to obtain Bayesian point estimates. This is the standard
approach for therapeutic drug monitoring (TDM) and dose individualization.

### Basic MAP estimation

```python
import nextstat

# Patient data: 2 vancomycin trough measurements
times = [12.0, 36.0]
conc  = [15.2, 12.8]

model = nextstat.TwoCompartmentIvPkModel(
    times, conc,
    dose=1000.0,
    error_model="proportional",
    sigma=0.15,
)

# Define population priors (log-normal, from published PopPK model)
# Format: list of (mean, sd) tuples for each parameter on log scale
priors = [
    {"mean": 4.5, "sd": 0.30},   # CL ~ LogNormal(log(4.5), 0.30)
    {"mean": 50.0, "sd": 0.25},  # V1 ~ LogNormal(log(50.0), 0.25)
    {"mean": 15.0, "sd": 0.40},  # V2 ~ LogNormal(log(15.0), 0.40)
    {"mean": 10.0, "sd": 0.35},  # Q  ~ LogNormal(log(10.0), 0.35)
]

map_result = nextstat.map_estimate(
    model,
    priors,
    max_iter=1000,
    tol=1e-8,
    compute_se=True,
)

print(f"Converged: {map_result['converged']} (iterations: {map_result['n_iter']})")
print(f"NLL (posterior): {map_result['nll_posterior']:.4f}")
print(f"NLL (likelihood): {map_result['nll']:.4f}")
print(f"Log-prior: {map_result['log_prior']:.4f}")

# Parameter estimates with standard errors and shrinkage
for name, val, se, shrink in zip(
    map_result['param_names'],
    map_result['params'],
    map_result['se'],
    map_result['shrinkage'],
):
    print(f"  {name}: {val:.4f} (SE={se:.4f}, shrinkage={shrink:.1%})")
```

### MAP result fields

| Field | Type | Description |
|-------|------|-------------|
| `params` | `list[float]` | MAP parameter estimates |
| `se` | `list[float]` | Standard errors (from Hessian of posterior) |
| `nll_posterior` | `float` | Negative log-posterior at MAP |
| `nll` | `float` | Negative log-likelihood component |
| `log_prior` | `float` | Log-prior component |
| `n_iter` | `int` | Optimizer iterations |
| `converged` | `bool` | Convergence flag |
| `param_names` | `list[str]` | Parameter names from the model |
| `shrinkage` | `list[float]` | Bayesian shrinkage towards prior (0 = data-driven, 1 = prior-driven) |

### Therapeutic drug monitoring workflow

A typical TDM workflow using MAP estimation:

```python
import nextstat
import math

# Step 1: Define the structural model for the patient
times = [1.0, 8.0, 12.0]
conc  = [28.5, 18.2, 14.1]

model = nextstat.TwoCompartmentIvPkModel(
    times, conc,
    dose=1500.0,        # actual administered dose
    error_model="proportional",
    sigma=0.12,
)

# Step 2: Use published population priors
priors = [
    {"mean": 4.5, "sd": 0.30},
    {"mean": 50.0, "sd": 0.25},
    {"mean": 15.0, "sd": 0.40},
    {"mean": 10.0, "sd": 0.35},
]

# Step 3: Obtain MAP estimates
map_result = nextstat.map_estimate(model, priors)

# Step 4: Predict trough at next dosing interval
cl, v1, v2, q = map_result['params']
ke = cl / v1
t_next_dose = 24.0  # next trough time
predicted_trough = model.predict(map_result['params'])

print(f"Individual CL = {cl:.2f} L/h")
print(f"Recommended action based on predicted exposure:")

# Step 5: Check shrinkage -- high shrinkage means data is not informative
for name, shrink in zip(map_result['param_names'], map_result['shrinkage']):
    flag = " (data-limited)" if shrink > 0.50 else ""
    print(f"  {name}: shrinkage = {shrink:.1%}{flag}")
```

### Interpreting shrinkage

| Shrinkage | Interpretation | Action |
|-----------|---------------|--------|
| < 20% | Data-driven estimate | Reliable individual estimate |
| 20-50% | Moderate shrinkage | Consider more sampling |
| > 50% | Prior-driven estimate | Individual data not informative for this parameter |
| > 80% | Nearly prior-only | Additional measurements needed |

---

## 16. CDISC XPT I/O

NextStat reads and writes SAS Transport v5 (XPT) files, the standard format
for regulatory submissions (FDA, EMA). This allows direct integration with
CDISC SDTM/ADaM datasets without requiring a SAS license.

### Reading XPT files

```python
import nextstat

# Read one or more datasets from an XPT file
datasets = nextstat.read_xpt("sdtm_pc.xpt")

print(f"Number of datasets: {len(datasets)}")
for ds in datasets:
    print(f"  Dataset: {ds.name}, Variables: {len(ds.columns)}, Rows: {ds.n_rows}")

# Access the first dataset
pc = datasets[0]
print(f"\nColumn names: {pc.columns}")
print(f"First 5 rows of USUBJID: {pc['USUBJID'][:5]}")
print(f"First 5 rows of PCTPTNUM: {pc['PCTPTNUM'][:5]}")
print(f"First 5 rows of PCSTRESN: {pc['PCSTRESN'][:5]}")
```

### Writing XPT files

```python
import nextstat

# Create an XPT dataset for submission
datasets = [
    nextstat.XptDataset(
        name="ADPC",
        label="PK Analysis Dataset",
        columns=["USUBJID", "AVISIT", "ATPT", "ATPTN", "AVAL", "AVALC",
                 "TRTA", "TRTAN", "PARAM", "PARAMCD", "ANL01FL"],
        data={
            "USUBJID": ["SUBJ-001", "SUBJ-001", "SUBJ-001", "SUBJ-002", "SUBJ-002"],
            "AVISIT":  ["DAY 1",    "DAY 1",    "DAY 1",    "DAY 1",    "DAY 1"],
            "ATPT":    ["0.5H",     "1H",       "2H",       "0.5H",     "1H"],
            "ATPTN":   [0.5,        1.0,        2.0,        0.5,        1.0],
            "AVAL":    [3.45,       8.12,       6.89,       2.98,       7.54],
            "AVALC":   ["3.45",     "8.12",     "6.89",     "2.98",     "7.54"],
            "TRTA":    ["DRUG A",   "DRUG A",   "DRUG A",   "DRUG A",   "DRUG A"],
            "TRTAN":   [1.0,        1.0,        1.0,        1.0,        1.0],
            "PARAM":   ["Conc",     "Conc",     "Conc",     "Conc",     "Conc"],
            "PARAMCD": ["CONC",     "CONC",     "CONC",     "CONC",     "CONC"],
            "ANL01FL": ["Y",        "Y",        "Y",        "Y",        "Y"],
        },
    )
]

nextstat.write_xpt("adpc.xpt", datasets)
print("ADPC dataset written to adpc.xpt")
```

### Converting XPT to NONMEM format

`xpt_to_nonmem()` converts a CDISC SDTM or ADaM PK dataset into the
column-mapped dictionary expected by `read_nonmem()`.

```python
import nextstat

# Read the SDTM PC domain
datasets = nextstat.read_xpt("sdtm_pc.xpt")
pc = datasets[0]

# Convert to NONMEM-style structure
nonmem_data = nextstat.xpt_to_nonmem(pc)

print(f"Subjects: {nonmem_data['n_subjects']}")
print(f"Subject IDs: {nonmem_data['subject_ids'][:5]}")
print(f"Times: {nonmem_data['times'][:10]}")
print(f"DV: {nonmem_data['dv'][:10]}")

# Use directly with population PK estimation
result = nextstat.nlme_foce(
    nonmem_data["times"],
    nonmem_data["dv"],
    nonmem_data["subject_idx"],
    nonmem_data["n_subjects"],
    dose=100.0,
    error_model="proportional",
    sigma=0.1,
    theta_init=[0.133, 8.0, 0.8],
    omega_init=[0.30, 0.25, 0.30],
    interaction=True,
)
```

### End-to-end regulatory workflow

A typical workflow for processing clinical trial data from CDISC format through
to population PK analysis:

```python
import nextstat

# 1. Read SDTM PC domain from sponsor submission
datasets = nextstat.read_xpt("study_1234_pc.xpt")
pc = datasets[0]

# 2. Convert to NONMEM format
data = nextstat.xpt_to_nonmem(pc)
print(f"Loaded {data['n_subjects']} subjects, {len(data['times'])} observations")

# 3. Fit population PK model
result = nextstat.nlme_foce(
    data["times"], data["dv"], data["subject_idx"], data["n_subjects"],
    dose=100.0,
    error_model="proportional",
    sigma=0.1,
    theta_init=[0.133, 8.0, 0.8],
    omega_init=[0.30, 0.25, 0.30],
    interaction=True,
)

# 4. Generate diagnostics
gof = nextstat.pk_gof(
    data["times"], data["dv"], data["subject_idx"],
    dose=100.0, theta=result["theta"], eta=result["eta"],
    error_model="proportional", sigma=0.1,
)

# 5. Write results back to XPT for submission
result_dataset = nextstat.XptDataset(
    name="ADPPK",
    label="Population PK Results",
    columns=["USUBJID", "PARAM", "ESTIMATE", "SE"],
    data={
        "USUBJID": ["ALL"] * 3,
        "PARAM":   ["CL", "V", "Ka"],
        "ESTIMATE": result["theta"],
        "SE":       [0.0] * 3,  # placeholder
    },
)
nextstat.write_xpt("adppk_results.xpt", [result_dataset])
print("Results written to adppk_results.xpt")
```

---

## Related tutorials

- [NONMEM Migration Guide](nonmem-migration.md) -- side-by-side NONMEM/NextStat comparison
- [Survival Analysis for Clinical Biostatisticians](pharma-survival.md) -- clinical endpoints
- [PK Modeling (low-level)](phase-13-pk.md) -- Rust API, dosing regimens, NONMEM reader
- [Population PK with NLME](phase-13-nlme.md) -- FOCE/SAEM internals, correlated Omega
