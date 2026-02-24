---
title: "Survival Analysis for Clinical Biostatisticians"
description: "Tutorial covering Kaplan-Meier, log-rank test, Cox PH, parametric models, interval censoring, and proportional hazards diagnostics for clinical trial endpoints."
status: stable
last_updated: 2026-02-21
---

# Survival Analysis for Clinical Biostatisticians

This tutorial covers survival analysis methods available in NextStat for
clinical trial endpoints. It is written for biostatisticians working on
time-to-event analyses in Phase II/III trials.

## Table of contents

1. [Kaplan-Meier estimation](#1-kaplan-meier-estimation)
2. [Log-rank test for treatment comparison](#2-log-rank-test-for-treatment-comparison)
3. [Cox proportional hazards](#3-cox-proportional-hazards)
4. [Parametric survival models](#4-parametric-survival-models)
5. [Interval censoring](#5-interval-censoring)
6. [Proportional hazards diagnostics](#6-proportional-hazards-diagnostics)
7. [Clinical trial workflow example](#7-clinical-trial-workflow-example)

---

## 1. Kaplan-Meier estimation

The Kaplan-Meier estimator is the standard non-parametric method for
estimating the survival function from right-censored data. NextStat provides
Greenwood variance estimates and log-log confidence intervals.

```python
import nextstat

# Time-to-event data: progression-free survival
times  = [2.1, 3.5, 5.0, 6.2, 7.8, 9.1, 10.3, 12.0, 14.5, 16.0,
          1.5, 4.2, 6.8, 8.5, 11.0, 13.2, 15.0, 18.0, 20.5, 24.0]
# True = event (progression), False = right-censored
events = [True, True, False, True, True, True, False, True, True, False,
          True, True, True, False, True, True, False, True, True, False]

km = nextstat.kaplan_meier(times, events, conf_level=0.95)
```

### Result structure

The returned `KaplanMeierResult` dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `times` | `list[float]` | Distinct event times (sorted ascending) |
| `survival` | `list[float]` | S(t) at each event time |
| `ci_lower` | `list[float]` | Lower 95% CI (log-log transform) |
| `ci_upper` | `list[float]` | Upper 95% CI (log-log transform) |
| `at_risk` | `list[int]` | Number at risk at each event time |
| `events` | `list[int]` | Number of events at each time |

### Printing a survival table

```python
print(f"{'Time':>8} {'At Risk':>8} {'Events':>8} {'S(t)':>8} {'95% CI':>18}")
print("-" * 55)
for i in range(len(km["times"])):
    print(
        f"{km['times'][i]:8.1f} "
        f"{km['at_risk'][i]:8d} "
        f"{km['events'][i]:8d} "
        f"{km['survival'][i]:8.3f} "
        f"[{km['ci_lower'][i]:.3f}, {km['ci_upper'][i]:.3f}]"
    )
```

### Median survival

The median survival time is the smallest time t where S(t) <= 0.5:

```python
median = None
for t, s in zip(km["times"], km["survival"]):
    if s <= 0.5:
        median = t
        break

if median is not None:
    print(f"Median survival: {median:.1f} months")
else:
    print("Median survival not reached")
```

---

## 2. Log-rank test for treatment comparison

The log-rank (Mantel-Cox) test compares survival distributions between two
or more groups. It is the standard primary analysis for randomized clinical
trials with a time-to-event endpoint.

```python
# Group labels: 0 = placebo, 1 = treatment
groups = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

lr = nextstat.log_rank_test(times, events, groups)

print(f"Log-rank test:")
print(f"  Chi-squared = {lr['statistic']:.4f}")
print(f"  df = {lr['df']}")
print(f"  p-value = {lr['p_value']:.6f}")

alpha = 0.05
if lr["p_value"] < alpha:
    print(f"  Result: Reject H0 at alpha={alpha} (significant difference in survival)")
else:
    print(f"  Result: Fail to reject H0 at alpha={alpha}")
```

### Stratified analysis

For stratified log-rank tests (e.g., stratified by region or prior therapy),
run separate tests per stratum and combine the chi-squared statistics:

```python
# Example: three strata
strata_results = []
for stratum_mask in [stratum_0_mask, stratum_1_mask, stratum_2_mask]:
    t_s = [times[i] for i in stratum_mask]
    e_s = [events[i] for i in stratum_mask]
    g_s = [groups[i] for i in stratum_mask]
    strata_results.append(nextstat.log_rank_test(t_s, e_s, g_s))

# Pooled stratified chi-squared
pooled_chi2 = sum(r["statistic"] for r in strata_results)
pooled_df = sum(r["df"] for r in strata_results)

from statistics import NormalDist
# Approximate p-value from chi-squared (using normal approximation for large df)
# For exact: use scipy.stats.chi2.sf(pooled_chi2, pooled_df)
print(f"Stratified log-rank: chi2={pooled_chi2:.4f}, df={pooled_df}")
```

---

## 3. Cox proportional hazards

The Cox PH model is the workhorse of clinical trial biostatistics for
estimating treatment effects while adjusting for covariates.

### Basic Cox PH fit

```python
from nextstat.survival import cox_ph

# Covariates matrix: each row is [treatment, age, ecog_ps, prior_therapy]
# treatment: 0 = control, 1 = experimental
x = [
    [0, 55.0, 1.0, 0],
    [0, 62.0, 0.0, 1],
    [0, 48.0, 1.0, 0],
    [0, 71.0, 2.0, 1],
    [0, 59.0, 0.0, 0],
    [0, 45.0, 1.0, 1],
    [0, 67.0, 1.0, 0],
    [0, 53.0, 0.0, 1],
    [0, 61.0, 2.0, 0],
    [0, 50.0, 1.0, 1],
    [1, 57.0, 1.0, 0],
    [1, 64.0, 0.0, 1],
    [1, 46.0, 1.0, 0],
    [1, 69.0, 2.0, 1],
    [1, 58.0, 0.0, 0],
    [1, 44.0, 1.0, 1],
    [1, 66.0, 1.0, 0],
    [1, 52.0, 0.0, 1],
    [1, 63.0, 2.0, 0],
    [1, 49.0, 1.0, 1],
]

fit = cox_ph.fit(
    times, events, x,
    ties="efron",         # Efron approximation (recommended)
    robust=True,          # sandwich SE
    compute_cov=True,     # compute covariance matrix
    compute_baseline=True,  # estimate baseline hazard
)
```

### Result: `CoxPhFit`

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef` | `list[float]` | Log-hazard ratio coefficients |
| `nll` | `float` | Negative partial log-likelihood |
| `converged` | `bool` | Convergence flag |
| `ties` | `str` | Tie-handling method used |
| `se` | `list[float]` | Model-based standard errors |
| `robust_se` | `list[float]` | Sandwich (robust) standard errors |
| `robust_kind` | `str` | `"hc0"` or `"cluster"` |
| `cov` | `list[list[float]]` | Model-based covariance matrix |
| `robust_cov` | `list[list[float]]` | Sandwich covariance matrix |
| `baseline_times` | `list[float]` | Baseline hazard time points |
| `baseline_cumhaz` | `list[float]` | Baseline cumulative hazard |

### Hazard ratios and confidence intervals

```python
covariate_names = ["Treatment", "Age", "ECOG PS", "Prior Therapy"]

print(f"{'Covariate':>15} {'HR':>8} {'95% CI':>18} {'p-value':>10}")
print("-" * 55)

import math
from statistics import NormalDist

nd = NormalDist()
hr_cis = fit.hazard_ratio_confint(level=0.95, robust=True)

for name, beta, se, hr, (hr_lo, hr_hi) in zip(
    covariate_names, fit.coef, fit.robust_se, fit.hazard_ratios(), hr_cis
):
    z = beta / se
    p = 2.0 * (1.0 - nd.cdf(abs(z)))
    sig = "*" if p < 0.05 else ""
    print(f"{name:>15} {hr:8.3f} [{hr_lo:.3f}, {hr_hi:.3f}] {p:10.4f} {sig}")
```

### Cluster-robust SE for multi-center trials

When subjects are clustered within centers (sites), the independence
assumption of the standard sandwich estimator is violated. Use `groups=`
to specify cluster membership:

```python
# center_ids: site ID for each subject
center_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3,
              1, 1, 2, 2, 2, 3, 3, 3, 3, 1]

fit_clustered = cox_ph.fit(
    times, events, x,
    ties="efron",
    robust=True,
    groups=center_ids,
    cluster_correction=True,  # G/(G-1) small-sample correction
)

print(f"Robust kind: {fit_clustered.robust_kind}")  # "cluster"
print(f"Cluster-robust SE: {fit_clustered.robust_se}")
```

### Survival predictions

Predict individual survival curves for new subjects using the Breslow
baseline cumulative hazard:

```python
# Predict survival for two hypothetical patients
new_patients = [
    [1, 55.0, 0.0, 0],  # treatment, age 55, ECOG 0, no prior therapy
    [0, 65.0, 1.0, 1],  # control, age 65, ECOG 1, prior therapy
]

# Time grid for evaluation
eval_times = [1.0, 3.0, 6.0, 12.0, 18.0, 24.0]

surv = fit.predict_survival(new_patients, times=eval_times)

print(f"{'Time':>6}", end="")
for i in range(len(new_patients)):
    print(f"  {'Patient ' + str(i+1):>12}", end="")
print()

for j, t in enumerate(eval_times):
    print(f"{t:6.1f}", end="")
    for i in range(len(new_patients)):
        print(f"  {surv[i][j]:12.4f}", end="")
    print()
```

---

## 4. Parametric survival models

When the proportional hazards assumption does not hold or when you need
to extrapolate beyond the observed data range, parametric models provide
a principled alternative.

### Weibull model

The Weibull distribution is the most common parametric choice for survival
data. It includes the exponential distribution as a special case (k=1).

Parameters: `log_lambda` (log-scale), `log_k` (log-shape).

```python
from nextstat.survival import weibull

wb = weibull.fit(times, events)

print(f"Weibull parameters: {wb.params}")
print(f"Standard errors: {wb.se}")
print(f"NLL: {wb.nll:.4f}")
print(f"Converged: {wb.converged}")

# Confidence intervals for parameters
cis = wb.confint(level=0.95)
for param, (lo, hi) in zip(["log_lambda", "log_k"], cis):
    print(f"  {param}: [{lo:.4f}, {hi:.4f}]")
```

### Log-normal AFT (Accelerated Failure Time)

The log-normal AFT model assumes that log(T) follows a normal distribution.
It models the acceleration factor directly, which can be easier to interpret
than hazard ratios.

Parameters: `mu` (location), `log_sigma` (log-scale).

```python
from nextstat.survival import lognormal_aft

ln = lognormal_aft.fit(times, events)

print(f"LogNormal AFT parameters: {ln.params}")
print(f"NLL: {ln.nll:.4f}")
```

### Exponential model

The simplest parametric model, assuming constant hazard.

Parameter: `log_lambda` (log-rate).

```python
from nextstat.survival import exponential

exp_fit = exponential.fit(times, events)

import math
lam = math.exp(exp_fit.params[0])
print(f"Hazard rate (lambda): {lam:.4f}")
print(f"Median survival (theoretical): {math.log(2) / lam:.2f}")
```

### Model comparison by AIC

```python
models = {
    "Exponential": exponential.fit(times, events),
    "Weibull": weibull.fit(times, events),
    "LogNormal AFT": lognormal_aft.fit(times, events),
}

print(f"{'Model':>15} {'NLL':>10} {'k':>4} {'AIC':>10}")
print("-" * 42)
for name, fit_result in models.items():
    k = len(fit_result.params)
    aic = 2 * fit_result.nll + 2 * k
    print(f"{name:>15} {fit_result.nll:10.2f} {k:4d} {aic:10.2f}")
```

---

## 5. Interval censoring

When event times are only known to lie within an interval (e.g., between
clinic visits), standard right-censoring methods are inappropriate. NextStat
provides interval-censored survival models.

### Censoring types

| `censor_type` | Meaning | Time specification |
|---------------|---------|-------------------|
| `"exact"` | Event observed at exact time | `time_lower == time_upper` |
| `"right"` | Right-censored (event after last follow-up) | `time_lower = last observation, time_upper = inf` |
| `"left"` | Left-censored (event before first observation) | `time_lower = 0, time_upper = first observation` |
| `"interval"` | Event occurred between two visits | `time_lower < time_upper` |

### Interval-censored Weibull

```python
import nextstat

# Dental study: time to caries onset between visits
time_lower  = [6.0, 12.0,  0.0, 18.0, 6.0,  12.0, 24.0, 0.0,  12.0, 18.0]
time_upper  = [12.0, 18.0, 6.0, 24.0, 12.0, 18.0, 1e10, 6.0,  18.0, 24.0]
censor_type = [
    "interval", "interval", "left", "interval", "interval",
    "interval", "right", "left", "interval", "interval",
]

model = nextstat.IntervalCensoredWeibullModel(time_lower, time_upper, censor_type)
result = nextstat.fit(model)

print(f"Parameters: {dict(zip(model.parameter_names(), result.bestfit))}")
print(f"NLL: {result.nll:.4f}")
```

### Interval-censored Weibull AFT with covariates

For regression analysis with interval-censored data:

```python
import nextstat

# Covariates: treatment (0/1), age
covariates = [
    [1, 45.0], [0, 52.0], [1, 38.0], [0, 61.0], [1, 55.0],
    [0, 48.0], [1, 42.0], [0, 67.0], [1, 50.0], [0, 44.0],
]

model = nextstat.IntervalCensoredWeibullAftModel(
    time_lower, time_upper, censor_type, covariates,
)
result = nextstat.fit(model)

param_names = model.parameter_names()
for name, val, se in zip(param_names, result.bestfit, result.uncertainties):
    print(f"  {name}: {val:.4f} (SE: {se:.4f})")
```

### Interval-censored exponential and log-normal

```python
# Exponential (constant hazard)
exp_model = nextstat.IntervalCensoredExponentialModel(
    time_lower, time_upper, censor_type,
)
exp_result = nextstat.fit(exp_model)

# Log-normal
ln_model = nextstat.IntervalCensoredLogNormalModel(
    time_lower, time_upper, censor_type,
)
ln_result = nextstat.fit(ln_model)
```

---

## 6. Proportional hazards diagnostics

The proportional hazards assumption is critical for valid inference from
the Cox model. NextStat provides Schoenfeld residual-based diagnostics.

### Schoenfeld residuals

Schoenfeld residuals are defined at each event time and should show no
systematic trend over time if the PH assumption holds.

```python
from nextstat.survival import cox_ph_schoenfeld

sr = cox_ph_schoenfeld(times, events, x, ties="efron")

# sr.event_times: list of event times
# sr.residuals: list of residual vectors (n_events x p)
# sr.coef: fitted coefficients used

print(f"Number of event times: {len(sr.event_times)}")
print(f"Number of covariates: {len(sr.residuals[0]) if sr.residuals else 0}")
```

### Correlation with log(time)

A non-zero correlation between Schoenfeld residuals and log(time) suggests
a violation of the PH assumption for that covariate.

```python
corrs = sr.corr_log_time()
covariate_names = ["Treatment", "Age", "ECOG PS", "Prior Therapy"]

for name, r in zip(covariate_names, corrs):
    print(f"  {name}: correlation with log(time) = {r:.4f}")
```

### Formal PH test

The `ph_test_log_time()` method performs a regression of each covariate's
Schoenfeld residuals on log(time) and tests H0: slope = 0.

```python
from nextstat.survival import cox_ph_ph_test

ph_tests = cox_ph_ph_test(times, events, x, ties="efron")

print(f"{'Covariate':>15} {'Slope':>10} {'SE':>10} {'z':>8} {'p-value':>10} {'Result':>10}")
print("-" * 68)

for test in ph_tests:
    name = covariate_names[test["feature"]]
    result_str = "VIOLATION" if test["p"] < 0.05 else "OK"
    print(
        f"{name:>15} "
        f"{test['slope']:10.4f} "
        f"{test['se_slope']:10.4f} "
        f"{test['z']:8.3f} "
        f"{test['p']:10.4f} "
        f"{result_str:>10}"
    )
```

### Interpreting PH test results

| Result | Interpretation | Action |
|--------|---------------|--------|
| p > 0.05 for all covariates | PH assumption holds | Proceed with Cox PH |
| p < 0.05 for treatment | Treatment effect changes over time | Consider time-dependent covariate or stratification |
| p < 0.05 for a prognostic factor | Time-varying effect | Stratify on that factor or use AFT model |

When the PH assumption is violated for the treatment variable, consider:

1. **Stratified Cox model** -- stratify by the violating covariate.
2. **Time-varying coefficients** -- extend the model (not yet in NextStat).
3. **AFT model** -- use `nextstat.LogNormalAftModel` or Weibull AFT for
   acceleration factor interpretation.
4. **RMST** -- restricted mean survival time is a valid alternative that
   does not require PH (available via `nextstat.churn_uplift_survival()`).

---

## 7. Clinical trial workflow example

This section demonstrates a complete primary analysis workflow for a
two-arm Phase III trial with progression-free survival (PFS) as the
primary endpoint.

### Step 1: Load and inspect data

```python
import nextstat

# Simulated Phase III PFS data
# 200 patients: 100 control (group=0), 100 experimental (group=1)
import random
rng = random.Random(42)

n_per_arm = 100
times_ctrl = [rng.expovariate(1.0 / 12.0) for _ in range(n_per_arm)]
times_expt = [rng.expovariate(1.0 / 18.0) for _ in range(n_per_arm)]

# Apply administrative censoring at 36 months
censor_time = 36.0
events_ctrl = [t < censor_time for t in times_ctrl]
events_expt = [t < censor_time for t in times_expt]
times_ctrl = [min(t, censor_time) for t in times_ctrl]
times_expt = [min(t, censor_time) for t in times_expt]

all_times = times_ctrl + times_expt
all_events = events_ctrl + events_expt
all_groups = [0] * n_per_arm + [1] * n_per_arm

print(f"Total N = {len(all_times)}")
print(f"Events = {sum(all_events)} ({100*sum(all_events)/len(all_events):.1f}%)")
print(f"Control events = {sum(events_ctrl)}")
print(f"Experimental events = {sum(events_expt)}")
```

### Step 2: Kaplan-Meier curves by arm

```python
# Compute KM for each arm separately
km_ctrl = nextstat.kaplan_meier(times_ctrl, events_ctrl, conf_level=0.95)
km_expt = nextstat.kaplan_meier(times_expt, events_expt, conf_level=0.95)

# Find median PFS per arm
def find_median(km_result):
    for t, s in zip(km_result["times"], km_result["survival"]):
        if s <= 0.5:
            return t
    return None

med_ctrl = find_median(km_ctrl)
med_expt = find_median(km_expt)
print(f"Median PFS control: {med_ctrl:.1f} months" if med_ctrl else "Median PFS control: not reached")
print(f"Median PFS experimental: {med_expt:.1f} months" if med_expt else "Median PFS experimental: not reached")
```

### Step 3: Primary analysis -- log-rank test

```python
lr = nextstat.log_rank_test(all_times, all_events, all_groups)

print(f"\nPrimary analysis: Log-rank test")
print(f"  Chi-squared = {lr['statistic']:.4f}")
print(f"  p-value = {lr['p_value']:.6f}")
print(f"  Conclusion: {'Significant' if lr['p_value'] < 0.05 else 'Not significant'}")
```

### Step 4: Cox PH for adjusted treatment effect

```python
from nextstat.survival import cox_ph
import math
from statistics import NormalDist

# Build covariate matrix: treatment indicator + baseline covariates
# In a real trial, include stratification factors and key prognostic variables
ages = [rng.gauss(60, 10) for _ in range(2 * n_per_arm)]
ecog = [rng.choice([0, 1, 2]) for _ in range(2 * n_per_arm)]
x = [[float(g), float(a), float(e)] for g, a, e in zip(all_groups, ages, ecog)]

fit = cox_ph.fit(
    all_times, all_events, x,
    ties="efron",
    robust=True,
    compute_baseline=True,
)

nd = NormalDist()
hr_cis = fit.hazard_ratio_confint(level=0.95, robust=True)
labels = ["Treatment", "Age", "ECOG PS"]

print(f"\nCox PH regression (adjusted)")
print(f"{'Covariate':>12} {'HR':>8} {'95% CI':>18} {'p':>10}")
for name, beta, se, hr, (lo, hi) in zip(
    labels, fit.coef, fit.robust_se, fit.hazard_ratios(), hr_cis
):
    z = beta / se
    p = 2.0 * (1.0 - nd.cdf(abs(z)))
    print(f"{name:>12} {hr:8.3f} [{lo:.3f}, {hi:.3f}] {p:10.4f}")
```

### Step 5: PH assumption check

```python
from nextstat.survival import cox_ph_ph_test

ph_tests = cox_ph_ph_test(all_times, all_events, x, ties="efron")

print(f"\nProportional hazards test (Schoenfeld residuals)")
any_violation = False
for test in ph_tests:
    name = labels[test["feature"]]
    status = "VIOLATION" if test["p"] < 0.05 else "OK"
    if test["p"] < 0.05:
        any_violation = True
    print(f"  {name}: slope={test['slope']:.4f}, p={test['p']:.4f} [{status}]")

if not any_violation:
    print("  All PH assumptions hold.")
```

### Step 6: Sensitivity analyses

```python
# 6a. Parametric models for extrapolation
from nextstat.survival import weibull, lognormal_aft

wb = weibull.fit(all_times, all_events)
ln = lognormal_aft.fit(all_times, all_events)

print(f"\nParametric model comparison:")
aic_wb = 2 * wb.nll + 2 * len(wb.params)
aic_ln = 2 * ln.nll + 2 * len(ln.params)
print(f"  Weibull AIC: {aic_wb:.2f}")
print(f"  LogNormal AFT AIC: {aic_ln:.2f}")
print(f"  Preferred: {'Weibull' if aic_wb < aic_ln else 'LogNormal AFT'}")

# 6b. Landmark survival rates
print(f"\n12-month PFS rates (Kaplan-Meier):")
for arm_name, km_result in [("Control", km_ctrl), ("Experimental", km_expt)]:
    s12 = None
    for t, s in zip(km_result["times"], km_result["survival"]):
        if t <= 12.0:
            s12 = s
    if s12 is not None:
        print(f"  {arm_name}: S(12) = {s12:.3f}")
    else:
        print(f"  {arm_name}: no events before 12 months")
```

### Step 7: Report summary

```python
print("\n" + "=" * 60)
print("PRIMARY ANALYSIS SUMMARY")
print("=" * 60)
print(f"Endpoint: Progression-Free Survival")
print(f"Analysis population: ITT (N={len(all_times)})")
print(f"Events: {sum(all_events)} ({100*sum(all_events)/len(all_events):.0f}%)")
print(f"Median PFS: Control {med_ctrl:.1f}m vs Experimental {med_expt:.1f}m"
      if med_ctrl and med_expt else "")
print(f"Log-rank p-value: {lr['p_value']:.6f}")
print(f"Adjusted HR (treatment): {fit.hazard_ratios()[0]:.3f} "
      f"[{hr_cis[0][0]:.3f}, {hr_cis[0][1]:.3f}]")
print(f"PH assumption: {'Upheld' if not any_violation else 'Violated (see details)'}")
print("=" * 60)
```

---

## API reference summary

### Non-parametric

| Function | Description |
|----------|-------------|
| `nextstat.kaplan_meier(times, events, conf_level=0.95)` | KM estimator with Greenwood CI |
| `nextstat.log_rank_test(times, events, groups)` | Mantel-Cox log-rank test |

### Semi-parametric

| Function | Description |
|----------|-------------|
| `nextstat.survival.cox_ph.fit(times, events, x, ...)` | Cox PH with robust/cluster SE |
| `nextstat.survival.cox_ph_schoenfeld(times, events, x, ...)` | Schoenfeld residuals |
| `nextstat.survival.cox_ph_ph_test(times, events, x, ...)` | PH assumption test |

### Parametric (right-censored)

| Function | Description |
|----------|-------------|
| `nextstat.survival.exponential.fit(times, events)` | Exponential MLE |
| `nextstat.survival.weibull.fit(times, events)` | Weibull MLE |
| `nextstat.survival.lognormal_aft.fit(times, events)` | LogNormal AFT MLE |

### Parametric (interval-censored)

| Model | Description |
|-------|-------------|
| `nextstat.IntervalCensoredWeibullModel(tl, tu, ct)` | Weibull, interval censoring |
| `nextstat.IntervalCensoredExponentialModel(tl, tu, ct)` | Exponential, interval censoring |
| `nextstat.IntervalCensoredLogNormalModel(tl, tu, ct)` | LogNormal, interval censoring |
| `nextstat.IntervalCensoredWeibullAftModel(tl, tu, ct, x)` | Weibull AFT with covariates |

---

## Related

- [Population PK Tutorial](pharma-pk.md) -- full pharmacometrics workflow
- [NONMEM Migration Guide](nonmem-migration.md) -- NONMEM/NextStat comparison
- [Survival Analysis (internal)](phase-9-survival.md) -- implementation notes
