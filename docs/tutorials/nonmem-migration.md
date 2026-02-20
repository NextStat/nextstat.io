---
title: "NONMEM to NextStat Migration Guide"
description: "Side-by-side comparison of NONMEM control streams and NextStat Python API for pharmacometricians transitioning to compiled PK/PD modeling."
status: stable
last_updated: 2026-02-21
---

# NONMEM to NextStat Migration Guide

This guide provides a direct mapping between NONMEM control stream syntax and
the NextStat Python API. It is intended for pharmacometricians who know NONMEM
and want to understand the NextStat equivalent for each workflow step.

## Table of contents

1. [Data input](#1-data-input)
2. [Model specification](#2-model-specification)
3. [Estimation methods](#3-estimation-methods)
4. [Output interpretation](#4-output-interpretation)
5. [Diagnostics](#5-diagnostics)
6. [Covariate modeling](#6-covariate-modeling)
7. [Common control streams](#7-common-control-streams)
8. [FAQ](#8-faq)

---

## 1. Data input

### NONMEM

```
$DATA warfarin.csv IGNORE=@
$INPUT ID TIME DV AMT EVID MDV WT AGE
```

NONMEM reads a fixed-format CSV with `$INPUT` declaring column names. Rows
starting with `@` (or other ignore characters) are skipped.

### NextStat

```python
import nextstat

csv_text = open("warfarin.csv").read()
data = nextstat.read_nonmem(csv_text)

# data is a dict:
#   n_subjects:  int
#   subject_ids: list[str]
#   times:       list[float]   (EVID=0 observations only)
#   dv:          list[float]   (EVID=0 observations only)
#   subject_idx: list[int]     (0-based subject index)
```

**Key differences:**

| Aspect | NONMEM | NextStat |
|--------|--------|----------|
| Input | File path in `$DATA` | CSV string passed to `read_nonmem()` |
| Column names | `$INPUT` declaration | Auto-detected from header row |
| Missing DV | `.` in DV column | `.` is recognized and excluded |
| Output format | Observation arrays for FOCE/SAEM | Dict with times, dv, subject_idx |
| Covariates | Available in `$PK` block | Passed separately to `scm()` |

**Covariates** like WT and AGE are not extracted by `read_nonmem()`. You
need to read them from the original CSV separately:

```python
import csv

# Read covariates from the raw CSV
with open("warfarin.csv") as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r.get("EVID", "0") == "0" and r.get("MDV", "0") == "0"]

weights = [float(r["WT"]) for r in rows]
ages = [float(r["AGE"]) for r in rows]
```

---

## 2. Model specification

### 1-compartment oral PK

#### NONMEM

```
$SUBROUTINES ADVAN2 TRANS2

$PK
  TVCL = THETA(1)
  TVV  = THETA(2)
  TVKA = THETA(3)
  CL = TVCL * EXP(ETA(1))
  V  = TVV  * EXP(ETA(2))
  KA = TVKA * EXP(ETA(3))
  S2 = V

$ERROR
  IPRED = F
  Y = IPRED * (1 + EPS(1))

$THETA
  (0.001, 0.133)   ; CL (L/h)
  (0.001, 8.0)     ; V (L)
  (0.001, 0.8)     ; KA (1/h)

$OMEGA
  0.09    ; BSV CL (omega^2 = 0.09 => omega = 0.3)
  0.0625  ; BSV V  (omega^2 = 0.0625 => omega = 0.25)
  0.09    ; BSV KA (omega^2 = 0.09 => omega = 0.3)

$SIGMA
  0.01    ; proportional error (sigma^2 = 0.01 => sigma = 0.1)

$ESTIMATION METHOD=COND INTERACTION MAXEVAL=9999
```

#### NextStat

```python
import nextstat

csv_text = open("warfarin.csv").read()
data = nextstat.read_nonmem(csv_text)

result = nextstat.nlme_foce(
    data["times"],
    data["dv"],
    data["subject_idx"],
    data["n_subjects"],
    dose=100.0,
    bioavailability=1.0,
    error_model="proportional",
    sigma=0.1,                        # sqrt($SIGMA) = 0.1
    theta_init=[0.133, 8.0, 0.8],    # $THETA initial estimates
    omega_init=[0.30, 0.25, 0.30],   # sqrt(diag($OMEGA))
    max_outer_iter=100,
    max_inner_iter=20,
    tol=1e-4,
    interaction=True,                 # METHOD=COND INTERACTION
)
```

**Parameter mapping:**

| NONMEM | NextStat | Notes |
|--------|----------|-------|
| `THETA(1)` | `theta_init[0]` | CL initial estimate |
| `THETA(2)` | `theta_init[1]` | V initial estimate |
| `THETA(3)` | `theta_init[2]` | Ka initial estimate |
| `$OMEGA 0.09` | `omega_init[0] = 0.30` | NextStat uses SD, not variance |
| `$SIGMA 0.01` | `sigma=0.1` | NextStat uses SD, not variance |
| `METHOD=COND INTERACTION` | `interaction=True` | FOCEI |
| `METHOD=COND` | `interaction=False` | FOCE (no interaction) |
| `MAXEVAL=9999` | `max_outer_iter=100` | NextStat converges faster |

**Critical difference**: NONMEM specifies `$OMEGA` and `$SIGMA` as
**variances** (omega^2, sigma^2). NextStat uses **standard deviations**
(omega, sigma). Always take the square root when converting.

### 2-compartment IV bolus

#### NONMEM

```
$SUBROUTINES ADVAN3 TRANS4

$PK
  CL = THETA(1) * EXP(ETA(1))
  V1 = THETA(2) * EXP(ETA(2))
  V2 = THETA(3)
  Q  = THETA(4)
  S1 = V1

$ERROR
  Y = F + EPS(1)

$THETA
  (0, 5.0)    ; CL
  (0, 20.0)   ; V1
  (0, 40.0)   ; V2
  (0, 2.0)    ; Q

$OMEGA 0.04 0.04

$SIGMA 0.5
```

#### NextStat

```python
# Individual fit (no random effects)
model = nextstat.TwoCompartmentIvPkModel(
    times, conc,
    dose=500.0,
    error_model="additive",
    sigma=0.5,
)
result = nextstat.fit(model)
# result.bestfit = [CL, V1, V2, Q]
```

### 2-compartment oral with combined error

#### NONMEM

```
$SUBROUTINES ADVAN4 TRANS4

$PK
  CL = THETA(1) * EXP(ETA(1))
  V2 = THETA(2) * EXP(ETA(2))
  V3 = THETA(3)
  Q  = THETA(4)
  KA = THETA(5) * EXP(ETA(3))
  S2 = V2

$ERROR
  IPRED = F
  W = SQRT(THETA(6)**2 + (THETA(7)*IPRED)**2)
  Y = IPRED + W*EPS(1)
```

#### NextStat

```python
model = nextstat.TwoCompartmentOralPkModel(
    times, conc,
    dose=200.0,
    bioavailability=0.85,
    error_model="combined",
    sigma=0.1,       # proportional component
    sigma_add=0.5,   # additive component
)
result = nextstat.fit(model)
# result.bestfit = [CL, V1, V2, Q, Ka]
```

---

## 3. Estimation methods

| NONMEM | NextStat | Notes |
|--------|----------|-------|
| `METHOD=COND` | `nextstat.nlme_foce(..., interaction=False)` | FOCE |
| `METHOD=COND INTERACTION` | `nextstat.nlme_foce(..., interaction=True)` | FOCEI (recommended) |
| `METHOD=SAEM` | `nextstat.nlme_saem(...)` | Stochastic Approximation EM |
| `METHOD=BAYES` | `nextstat.sample(model, method="nuts")` | NUTS (Bayesian) |
| `METHOD=0` (FO) | Not directly available | Use FOCE without interaction |

### SAEM side-by-side

#### NONMEM

```
$ESTIMATION METHOD=SAEM INTERACTION
  NBURN=200 NITER=100 ISAMPLE=1 SEED=12345
```

#### NextStat

```python
result = nextstat.nlme_saem(
    data["times"],
    data["dv"],
    data["subject_idx"],
    data["n_subjects"],
    dose=100.0,
    error_model="proportional",
    sigma=0.1,
    theta_init=[0.133, 8.0, 0.8],
    omega_init=[0.30, 0.25, 0.30],
    n_burn=200,
    n_iter=100,
    n_chains=1,
    seed=12345,
    tol=1e-4,
)
```

---

## 4. Output interpretation

### NONMEM output to NextStat result mapping

| NONMEM output | NextStat equivalent | Notes |
|---------------|---------------------|-------|
| `THETA(1)` in `.ext` file | `result["theta"][0]` | Direct mapping |
| `OMEGA(1,1)` in `.ext` file | `result["omega_matrix"][0][0]` | Omega variance |
| `OMEGA SD` | `result["omega"][0]` | Standard deviation |
| `SIGMA(1,1)` | `sigma` parameter | Fixed in current API |
| `ETA(1)` per subject | `result["eta"][subject_idx][0]` | Conditional modes |
| `OBJ` (OFV) | `result["ofv"]` | -2 * log-likelihood |
| `MINIMIZATION SUCCESSFUL` | `result["converged"]` | Bool |
| Correlation matrix | `result["correlation"]` | Omega correlation |
| Number of iterations | `result["n_iter"]` | Outer iterations |

### Parameter recovery example

```python
# NONMEM-style output table
print("=" * 60)
print("MINIMUM VALUE OF OBJECTIVE FUNCTION")
print(f"  {result['ofv']:.6f}")
print()
print("FINAL PARAMETER ESTIMATES")
print()
print("  THETA  -  VECTOR OF FIXED EFFECTS:")
for i, (name, val) in enumerate(zip(["CL", "V", "Ka"], result["theta"])):
    print(f"    THETA({i+1}) = {val:.6E}  ; {name}")
print()
print("  OMEGA  -  COV MATRIX FOR RANDOM EFFECTS:")
for i, row in enumerate(result["omega_matrix"]):
    print(f"    {'  '.join(f'{v:.6E}' for v in row)}")
print()
print("  RANDOM EFFECT SDs:")
for i, (name, sd) in enumerate(zip(["CL", "V", "Ka"], result["omega"])):
    print(f"    omega_{name} = {sd:.4f} (variance = {sd**2:.6f})")
```

---

## 5. Diagnostics

### GOF ($TABLE equivalent)

#### NONMEM

```
$TABLE ID TIME DV PRED IPRED IWRES CWRES NOPRINT ONEHEADER FILE=sdtab
```

#### NextStat

```python
gof = nextstat.pk_gof(
    data["times"],
    data["dv"],
    data["subject_idx"],
    dose=100.0,
    theta=result["theta"],
    eta=result["eta"],
    error_model="proportional",
    sigma=0.1,
)

# Print NONMEM-style table
print(f"{'SUBJECT':>8} {'TIME':>8} {'DV':>10} {'PRED':>10} {'IPRED':>10} {'IWRES':>10} {'CWRES':>10}")
for rec in gof:
    print(
        f"{rec['subject']:8d} {rec['time']:8.2f} {rec['dv']:10.4f} "
        f"{rec['pred']:10.4f} {rec['ipred']:10.4f} "
        f"{rec['iwres']:10.4f} {rec['cwres']:10.4f}"
    )
```

### VPC ($SIMULATION equivalent)

#### NONMEM

```
$SIMULATION (12345) ONLYSIM SUBPROBLEMS=200
$TABLE ID TIME DV FILE=simtab
; ... then post-process with PsN vpc command
```

#### NextStat

```python
vpc = nextstat.pk_vpc(
    data["times"],
    data["dv"],
    data["subject_idx"],
    data["n_subjects"],
    dose=100.0,
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

# Per-bin quantile comparison
for b in vpc["bins"]:
    print(
        f"[{b['time_lo']:5.1f}, {b['time_hi']:5.1f}]  "
        f"obs_p50={b['observed_quantiles'][1]:.2f}  "
        f"sim_PI=[{b['simulated_pi_lo'][1]:.2f}, {b['simulated_pi_hi'][1]:.2f}]"
    )
```

---

## 6. Covariate modeling

### NONMEM manual covariate

```
$PK
  TVCL = THETA(1) * (WT/70)**THETA(4)
  TVV  = THETA(2)
  TVKA = THETA(3)
  CL = TVCL * EXP(ETA(1))
  V  = TVV  * EXP(ETA(2))
  KA = TVKA * EXP(ETA(3))

$THETA
  ...
  (0, 0.75)  ; THETA(4): allometric exponent for WT on CL
```

### NextStat automated SCM

```python
scm_result = nextstat.scm(
    data["times"],
    data["dv"],
    data["subject_idx"],
    data["n_subjects"],
    covariates=[weights, ages],
    covariate_names=["WT", "AGE"],
    dose=100.0,
    error_model="proportional",
    sigma=0.1,
    theta_init=[0.133, 8.0, 0.8],
    omega_init=[0.30, 0.25, 0.30],
    param_names=["CL", "V", "Ka"],
    relationships=["power", "exponential"],
    forward_alpha=0.05,
    backward_alpha=0.01,
)

print(f"Base OFV: {scm_result['base_ofv']:.2f}")
print(f"Final OFV: {scm_result['final_ofv']:.2f}")

for step in scm_result["selected"]:
    print(f"  {step['name']}: coeff={step['coefficient']:.4f}, p={step['p_value']:.4f}")
```

### SCM mapping

| PsN/NONMEM SCM | NextStat `scm()` |
|-----------------|------------------|
| `scm.config` file | Function arguments |
| `-forward_alpha=0.05` | `forward_alpha=0.05` |
| `-backward_alpha=0.01` | `backward_alpha=0.01` |
| `relation CL WT=power` | `relationships=["power"]` |
| Forward inclusion output | `scm_result["forward_trace"]` |
| Backward elimination output | `scm_result["backward_trace"]` |
| Final model | `scm_result["theta"]`, `scm_result["omega"]` |

---

## 7. Common control streams

### Pattern 1: Simple 1-cpt oral with proportional error

**NONMEM:**

```
$PROB WARFARIN 1-CPT ORAL
$DATA warfarin.csv IGNORE=@
$INPUT ID TIME DV AMT EVID MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL=THETA(1)*EXP(ETA(1))
  V=THETA(2)*EXP(ETA(2))
  KA=THETA(3)*EXP(ETA(3))
  S2=V
$ERROR
  Y=F*(1+EPS(1))
$THETA (0,0.133) (0,8) (0,0.8)
$OMEGA 0.09 0.0625 0.09
$SIGMA 0.01
$EST METHOD=COND INTER MAXEVAL=9999
```

**NextStat:**

```python
data = nextstat.read_nonmem(open("warfarin.csv").read())
result = nextstat.nlme_foce(
    data["times"], data["dv"], data["subject_idx"], data["n_subjects"],
    dose=100.0, error_model="proportional", sigma=0.1,
    theta_init=[0.133, 8.0, 0.8], omega_init=[0.30, 0.25, 0.30],
    interaction=True,
)
```

### Pattern 2: Individual fit (no random effects)

**NONMEM:**

```
$PROB THEOPHYLLINE 1-CPT INDIVIDUAL
$DATA theo_subj1.csv IGNORE=@
$INPUT TIME DV AMT EVID
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL=THETA(1)
  V=THETA(2)
  KA=THETA(3)
  S2=V
$ERROR
  Y=F+EPS(1)
$THETA (0,2) (0,30) (0,1.5)
$SIGMA 1
$EST METHOD=ZERO MAXEVAL=9999
```

**NextStat:**

```python
model = nextstat.OneCompartmentOralPkModel(
    times, conc, dose=320.0, sigma=1.0,
)
result = nextstat.fit(model)
```

### Pattern 3: 2-cpt IV with LLOQ censoring

**NONMEM:**

```
$PROB 2-CPT IV WITH M3 LLOQ
$SUBROUTINES ADVAN3 TRANS4
$PK
  CL=THETA(1)*EXP(ETA(1))
  V1=THETA(2)*EXP(ETA(2))
  V2=THETA(3)
  Q=THETA(4)
  S1=V1
$ERROR
  LLOQ=0.05
  IF (DV.GE.LLOQ) THEN
    F_FLAG=0
    Y=F+EPS(1)
  ELSE
    F_FLAG=1
    Y=PHI((LLOQ-F)/SQRT(SIGMA(1,1)))
  ENDIF
```

**NextStat:**

```python
model = nextstat.TwoCompartmentIvPkModel(
    times, conc, dose=500.0,
    error_model="additive", sigma=0.5,
    lloq=0.05, lloq_policy="censored",  # M3 method
)
result = nextstat.fit(model)
```

### Pattern 4: SAEM estimation

**NONMEM:**

```
$EST METHOD=SAEM INTERACTION NBURN=300 NITER=200 ISAMPLE=2 SEED=42
$EST METHOD=IMP INTERACTION EONLY=1 NITER=5 ISAMPLE=3000 SEED=42
```

**NextStat:**

```python
result = nextstat.nlme_saem(
    data["times"], data["dv"], data["subject_idx"], data["n_subjects"],
    dose=100.0, error_model="proportional", sigma=0.1,
    theta_init=[0.133, 8.0, 0.8], omega_init=[0.30, 0.25, 0.30],
    n_burn=300, n_iter=200, n_chains=1, seed=42,
)
```

### Pattern 5: Bootstrap (PsN)

**PsN:**

```bash
bootstrap run1.mod -samples=1000 -threads=8 -seed=42
```

**NextStat:**

```python
import random

bootstrap_thetas = []
rng = random.Random(42)
subj_map = {}
for i, s in enumerate(data["subject_idx"]):
    subj_map.setdefault(s, []).append(i)

for _ in range(1000):
    sampled = [rng.randint(0, data["n_subjects"] - 1) for _ in range(data["n_subjects"])]
    t, y, si = [], [], []
    for new_id, orig in enumerate(sampled):
        for idx in subj_map[orig]:
            t.append(data["times"][idx])
            y.append(data["dv"][idx])
            si.append(new_id)
    try:
        r = nextstat.nlme_foce(
            t, y, si, data["n_subjects"],
            dose=100.0, error_model="proportional", sigma=0.1,
            theta_init=result["theta"], omega_init=result["omega"],
            max_outer_iter=50, tol=1e-3, interaction=True,
        )
        if r["converged"]:
            bootstrap_thetas.append(r["theta"])
    except Exception:
        pass
```

### Pattern 6: Individual fit with predictions

**NONMEM:**

```
$TABLE ID TIME DV IPRED FILE=patab
```

**NextStat:**

```python
result = nextstat.fit(model)
predicted = model.predict(result.bestfit)

for t, obs, pred in zip(times, conc, predicted):
    print(f"t={t:.1f}  DV={obs:.3f}  IPRED={pred:.3f}")
```

### Pattern 7: GOF diagnostics

**NONMEM + PsN:**

```
$TABLE ID TIME DV PRED IPRED IWRES CWRES NOPRINT FILE=sdtab
; then: execute run1.mod && vpc -samples=200 run1.mod
```

**NextStat (both in one script):**

```python
# GOF
gof = nextstat.pk_gof(
    data["times"], data["dv"], data["subject_idx"],
    dose=100.0, theta=result["theta"], eta=result["eta"],
    error_model="proportional", sigma=0.1,
)

# VPC
vpc = nextstat.pk_vpc(
    data["times"], data["dv"], data["subject_idx"], data["n_subjects"],
    dose=100.0, theta=result["theta"], omega_matrix=result["omega_matrix"],
    error_model="proportional", sigma=0.1,
    n_sim=200, n_bins=10, seed=42,
)
```

### Pattern 8: SCM covariate search (PsN)

**PsN:**

```bash
scm run1.mod -config=scm.config
```

Where `scm.config` contains:

```
search_direction=both
p_forward=0.05
p_backward=0.01

[test_relations]
CL=WT,AGE
V=WT
```

**NextStat:**

```python
scm_result = nextstat.scm(
    data["times"], data["dv"], data["subject_idx"], data["n_subjects"],
    covariates=[weights, ages],
    covariate_names=["WT", "AGE"],
    dose=100.0, error_model="proportional", sigma=0.1,
    theta_init=result["theta"], omega_init=result["omega"],
    param_names=["CL", "V", "Ka"],
    forward_alpha=0.05, backward_alpha=0.01,
)
```

### Pattern 9: Survival endpoint (Kaplan-Meier + log-rank)

**R (survival package):**

```r
library(survival)
km <- survfit(Surv(time, event) ~ group, data=dat)
survdiff(Surv(time, event) ~ group, data=dat)
```

**NextStat:**

```python
km = nextstat.kaplan_meier(times, events, conf_level=0.95)
lr = nextstat.log_rank_test(times, events, groups)
```

### Pattern 10: Cox PH with robust SE

**R:**

```r
coxph(Surv(time, event) ~ treatment + age + biomarker,
      data=dat, robust=TRUE)
```

**NextStat:**

```python
from nextstat.survival import cox_ph

fit = cox_ph.fit(times, events, x, ties="efron", robust=True)
```

---

## 8. FAQ

### Can NextStat handle NONMEM ADVAN5/6/13 (ODE-based models)?

NextStat currently provides **analytical** PK models (1-cpt oral, 2-cpt IV,
2-cpt oral) with closed-form solutions. General ODE models (ADVAN6/13) are
not yet supported. For most standard PK models (ADVAN1-4), NextStat is a
drop-in replacement with substantially better performance.

### Where is `$COVARIANCE`?

NextStat computes parameter uncertainties (standard errors) automatically
during fitting via the L-BFGS-B inverse Hessian approximation. Access them
from the `FitResult`:

```python
result = nextstat.fit(model)
se = result.uncertainties   # 1-sigma standard errors
```

For population models (FOCE/SAEM), the omega variance-covariance matrix is
returned directly in `result["omega_matrix"]`.

For bootstrap-based CIs, see [Section 8 of the PK tutorial](pharma-pk.md#8-bootstrap-confidence-intervals).

### How do I do VPC/GOF?

Use `nextstat.pk_vpc()` and `nextstat.pk_gof()` directly. No PsN post-processing
step is required. See [Section 5 of the PK tutorial](pharma-pk.md#6-model-diagnostics).

### Is NextStat validated for regulatory submissions?

NextStat provides deterministic, reproducible results with documented
numerical parity against reference implementations. For GxP-regulated
environments, an IQ/OQ/PQ validation protocol is available. Contact
the developers for qualification documentation.

The M3 LLOQ method (`lloq_policy="censored"`) is implemented per the
FDA guidance on handling BQL data.

### How do I convert NONMEM OMEGA/SIGMA to NextStat?

NONMEM uses **variances** (omega^2, sigma^2). NextStat uses **standard
deviations** (omega, sigma). Take the square root:

```python
# NONMEM: $OMEGA 0.09 0.0625 0.09
# NextStat:
omega_init = [0.30, 0.25, 0.30]   # sqrt(0.09), sqrt(0.0625), sqrt(0.09)

# NONMEM: $SIGMA 0.01
# NextStat:
sigma = 0.1                        # sqrt(0.01)
```

### Can I use NextStat with nlmixr2 or Monolix datasets?

Yes. `read_nonmem()` accepts any CSV with ID/TIME/DV columns. Most nlmixr2
and Monolix datasets use NONMEM-compatible format.

### What about allometric scaling?

Allometric scaling is handled through the SCM (Stepwise Covariate Modeling)
interface using `relationships=["power"]`. The power relationship
`CL_i = CL * (WT/WT_median)^beta` is equivalent to the NONMEM formulation
`TVCL = THETA(1) * (WT/70)**THETA(4)`.

### How does performance compare?

See [Section 11 of the PK tutorial](pharma-pk.md#11-performance-comparison)
for detailed benchmarks. In summary: 600-60,000x faster than NONMEM depending
on the workflow.

### Can I run NextStat on a cluster?

NextStat uses Rayon for automatic multi-core parallelism. For toy studies and
bootstrap, you can shard across nodes:

```python
# Each node runs a subset of bootstrap replicates
# Then merge results
nextstat.set_threads(16)  # use 16 cores per node
```

### What NONMEM features are NOT available?

| Feature | Status |
|---------|--------|
| ADVAN5/6/13 (general ODE) | Not yet available |
| `$MIXTURE` (mixture models) | Not yet available |
| `$PRIOR` (informative priors) | Use NUTS sampling instead |
| `$SIMULATION` (standalone) | Use `pk_vpc()` or write a loop |
| `$TABLE FILE=` (file output) | Results are in-memory dicts |
| `$NONPARAMETRIC` | Not available |
| `$SUPERPOSITION` | Rust API only (DosingRegimen) |
| Time-varying covariates | Not yet in NLME estimators |
| IOV (inter-occasion variability) | Not yet available |

---

## Related

- [Population PK Tutorial](pharma-pk.md) -- complete pharmacometrics workflow
- [Survival Analysis](pharma-survival.md) -- clinical endpoint analysis
- [PK Modeling (internal)](phase-13-pk.md) -- Rust API details
- [NLME Tutorial (internal)](phase-13-nlme.md) -- FOCE/SAEM internals
