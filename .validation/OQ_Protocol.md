# Operational Qualification (OQ) Protocol

## Document Control

| Field | Value |
|-------|-------|
| Document ID | NS-VAL-OQ-001 |
| Version | 1.0 |
| Effective Date | ____/____/________ |
| Supersedes | N/A |
| Classification | GxP Validation Deliverable |
| Author | ________________________________ |
| Reviewer | ________________________________ |
| Approver | ________________________________ |

---

## 1. Purpose

This document defines the Operational Qualification (OQ) protocol for verifying that
NextStat produces computationally accurate results across all statistical and pharmacometric
functions used in biostatistical and pharmacometric workflows. Each test case compares
NextStat output against published reference values, analytical solutions, or validated
reference implementations with quantitative acceptance criteria.

Successful completion of OQ provides documented evidence that the software operates
correctly within its specified parameters, as required by:
- 21 CFR Part 11 Section 11.10(a) -- validation of computerised systems
- EU Annex 11 Section 4.7 -- data accuracy checks
- ICH E6(R3) Section 4.9 -- computerised systems used in clinical trials
- GAMP 5 Section 7.6 -- operational qualification requirements

## 2. Scope

This protocol covers 93 test cases across 11 functional domains. Each test compares
NextStat output against published reference values or validated reference implementations
with documented numerical tolerances.

**Note:** For detailed pharma-specific OQ test cases (FOCE, SAEM, SCM, VPC, GOF, NONMEM I/O,
Artifact export), refer to the companion document `docs/validation/iq-oq-pq-protocol.md`
which contains 85 dedicated test cases for the pharmacometrics module.

## 3. Reference Implementations

| Domain | Reference Implementation | Validation Basis |
|--------|------------------------|-----------------|
| PK Analytical | Closed-form solution / Maple CAS | Exact analytical formula, eigenvalue decomposition |
| MLE Optimization | NONMEM 7.5 | Parameter estimates, OFV at convergence |
| FOCE/FOCEI | NONMEM 7.5, nlmixr2 | Warfarin and Theophylline reference datasets |
| SAEM | Monolix 2024R1 | Convergence diagnostics, parameter recovery |
| Survival (Cox PH) | R `survival` 3.7-1 | Coefficients, robust SE, likelihood ratio test |
| Kaplan-Meier | R `survival::survfit()` | Survival probabilities, Greenwood CI |
| GLM | R `glm()`, SAS PROC LOGISTIC | Coefficients, deviance, AIC |
| LMM | R `lme4::lmer()` 1.1-35 | REML estimates, variance components, BLUPs |
| Bootstrap CI | R `boot::boot.ci()` | BCa intervals, acceleration, z0 |
| Emax/Hill | Manual analytical calculation | Dose-response curve predictions |
| ODE Solver | Analytical PK solutions | RK45 vs closed-form at tol=1e-8 |
| HEP/HistFactory | pyhf 0.7.x | NLL, q(mu), CLs, ranking, profile scan |

## 4. Test Environment

All OQ tests shall be executed on the IQ-qualified platform. Document the test environment:

| Item | Value |
|------|-------|
| NextStat Version | ________________________________ |
| Python Version | ________________________________ |
| OS / Architecture | ________________________________ |
| CPU Model | ________________________________ |
| RAM | ________________________________ |
| Date of Execution | ________________________________ |
| Executed By | ________________________________ |

## 5. Test Cases

### 5.1 PK Analytical Models (8 tests)

| ID | Test Description | Input Parameters | Expected Result | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------------|-----------------|-----------|--------|-----------|
| OQ-PK-001 | 1-compartment oral C(t) at 10 time points | D=320 mg, F=1.0, CL=2.0 L/h, V=30.0 L, Ka=1.5 h^-1, t={0.5,1,2,4,6,8,12,24,36,48} h | Each C(t) matches formula: C = (F*D/V) * Ka/(Ka-Ke) * (exp(-Ke*t) - exp(-Ka*t)) | Relative error < 1e-10 | | |
| OQ-PK-002 | 1-compartment oral boundary: C(0) = 0 | Same parameters, t=0 | C(0) = 0.0 | Exact (< 1e-15) | | |
| OQ-PK-003 | 2-compartment IV C(0) = D/V1 | D=1000 mg, CL=4.0, V1=10.0, V2=20.0, Q=2.0 L/h | C(0) = 100.0 mg/L | Relative error < 1e-12 | | |
| OQ-PK-004 | 2-compartment IV C(inf) approaches 0 | Same parameters, t=1000 h | C(1000) < 1e-10 | Absolute < 1e-10 | | |
| OQ-PK-005 | 2-compartment oral peak exists | D=500, F=0.9, CL=3.0, V1=15.0, V2=25.0, Q=1.5, Ka=1.2 | tmax > 0 and C(tmax) > C(tmax-0.1) and C(tmax) > C(tmax+0.1) | Boolean True | | |
| OQ-PK-006 | Analytical gradient vs finite difference | All PK models, 5 parameter tuples each | |grad_analytical - grad_fd| where grad_fd = (C(p+h) - C(p-h)) / (2h), h=1e-6 | Relative error < 1e-6 per component | | |
| OQ-PK-007 | Degenerate case: Ka approximately equals Ke | CL=1.5, V=1.0, Ka=1.5 (Ka = Ke = CL/V) | C(1.0) is finite and positive; no NaN or Inf | isfinite(C) and C > 0 | | |
| OQ-PK-008 | LLOQ censored likelihood computation | DV < LLOQ=0.5, additive error sigma=0.1 | Censored NLL = -log(Phi((LLOQ - f) / sigma)) | Absolute error < 1e-8 | | |

### 5.2 FOCE/FOCEI Population PK (6 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-FOCE-001 | Warfarin FOCE theta recovery (CL, V, Ka) | NONMEM 7.5 | \|theta_NS - theta_NM\| / theta_NM < 0.05 | | |
| OQ-FOCE-002 | Warfarin OFV agreement with NONMEM | NONMEM 7.5 | \|OFV_NS - OFV_NM\| < 1.0 | | |
| OQ-FOCE-003 | Theophylline FOCE theta and OFV | NONMEM 7.5 | Same as OQ-FOCE-001/002 | | |
| OQ-FOCE-004 | FOCEI interaction terms computed correctly | NONMEM 7.5 | Variance depends on IPRED | | |
| OQ-FOCE-005 | Correlated Omega recovery (rho=0.5) | Simulated data, 30 subjects | \|rho_hat - 0.5\| < 0.3 | | |
| OQ-FOCE-006 | Convergence on near-boundary pathological data | Simulated near-boundary | Converges without crash; converged == True | | |

### 5.3 SAEM (5 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-SAEM-001 | SAEM converges on 30-subject pop PK | Simulated 1-cpt oral | burn_in_only == False; converged | | |
| OQ-SAEM-002 | SAEM theta recovery | True theta=[CL=2.5, V=35, Ka=1.2] | Relative error < 15% per parameter | | |
| OQ-SAEM-003 | SAEM omega SD recovery | True omega_SD=[0.3, 0.2, 0.4] | Relative error < 25% per component | | |
| OQ-SAEM-004 | SAEM MCMC acceptance rate | Internal diagnostics | Mean acceptance rate in [0.15, 0.60] | | |
| OQ-SAEM-005 | SAEM reproducibility (fixed seed) | Two runs, seed=42 | theta_run1 == theta_run2 (bit-exact) | | |

### 5.4 Goodness of Fit / VPC (5 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-GOF-001 | PRED = C(t; theta_pop, eta=0) | Manual computation | Relative error < 1e-10 | | |
| OQ-GOF-002 | IPRED = C(t; theta_pop * exp(eta_hat)) | Manual computation | Relative error < 1e-10 | | |
| OQ-GOF-003 | IWRES approximately N(0,1) under correct model | Simulated correct model | abs(mean) < 0.3, abs(SD - 1.0) < 0.3 | | |
| OQ-GOF-004 | VPC 90% PI coverage under correct model | n_sim=500, correct model | Coverage in [0.80, 1.00] | | |
| OQ-GOF-005 | VPC detects model misspecification | 2-cpt data, 1-cpt fit | Coverage < 0.70 for >= 1 quantile | | |

### 5.5 Stepwise Covariate Modeling (4 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-SCM-001 | Forward selection identifies true covariate | Weight on CL (power=0.75) | delta_OFV > 3.84 and WT selected | | |
| OQ-SCM-002 | Forward selection rejects noise covariate | Random noise variable | Noise not in selected set | | |
| OQ-SCM-003 | Backward elimination removes non-significant | Forced noise covariate | Noise removed at alpha=0.01 | | |
| OQ-SCM-004 | Covariate coefficient recovery | True exponent 0.75 | \|theta_cov_hat - 0.75\| < 0.30 | | |

### 5.6 Error Models (4 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-ERR-001 | Additive error: variance = sigma^2 (constant) | Manual: var=0.01 for sigma=0.1 | Exact to 1e-15 | | |
| OQ-ERR-002 | Proportional error: variance = (sigma*f)^2 | Manual calculation | Relative error < 1e-14 | | |
| OQ-ERR-003 | Combined error: variance = sigma_add^2 + (sigma_prop*f)^2 | Manual calculation | Relative error < 1e-14 | | |
| OQ-ERR-004 | Error model rejects invalid input | sigma < 0, sigma = NaN | Error raised for all invalid inputs | | |

### 5.7 NONMEM Dataset I/O (4 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-IO-001 | Parse Theophylline NONMEM CSV | 12 subjects, 132 observations | n_subjects == 12, n_obs == 132 | | |
| OQ-IO-002 | EVID/MDV column handling | EVID=0 obs, EVID=1 dose, MDV=1 missing | Correct classification | | |
| OQ-IO-003 | Missing required column detection | CSV without DV | Error raised with descriptive message | | |
| OQ-IO-004 | Case-insensitive column names | Lowercase headers | Parse succeeds | | |

### 5.8 Survival Analysis (10 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-SURV-001 | Cox PH coefficients on veteran lung data | R `survival::coxph()` | Relative error < 1e-6 per coefficient | | |
| OQ-SURV-002 | Cox PH robust SE (clustered) | R `survival::coxph(cluster=...)` | Relative error < 1e-4 per SE | | |
| OQ-SURV-003 | Cox PH likelihood ratio test | R `survival::coxph()` | \|LRT_NS - LRT_R\| < 1e-4 | | |
| OQ-SURV-004 | Schoenfeld residuals computation | R `survival::cox.zph()` | Relative error < 1e-4 | | |
| OQ-SURV-005 | Kaplan-Meier survival probabilities | R `survival::survfit()` | Relative error < 1e-8 at each event time | | |
| OQ-SURV-006 | Kaplan-Meier Greenwood CI | R `survival::survfit()` | Relative error < 1e-6 for CI bounds | | |
| OQ-SURV-007 | Kaplan-Meier right-censoring | Simulated censored data | Correct handling of censored observations | | |
| OQ-SURV-008 | Log-rank test statistic | R `survival::survdiff()` | \|chi2_NS - chi2_R\| < 1e-4 | | |
| OQ-SURV-009 | Breslow baseline hazard | R `survival::basehaz()` | Relative error < 1e-6 at each time | | |
| OQ-SURV-010 | Stratified Cox PH | R `survival::coxph(strata=...)` | Relative error < 1e-4 per coefficient | | |

### 5.9 Generalized Linear Models (8 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-GLM-001 | Logistic regression coefficients | R `glm(family=binomial)` | Relative error < 1e-6 per coefficient | | |
| OQ-GLM-002 | Logistic regression deviance | R `glm()` | \|dev_NS - dev_R\| < 1e-6 | | |
| OQ-GLM-003 | Logistic regression SE | R `summary.glm()` | Relative error < 1e-4 per SE | | |
| OQ-GLM-004 | Poisson regression coefficients | R `glm(family=poisson)` | Relative error < 1e-6 per coefficient | | |
| OQ-GLM-005 | Poisson regression deviance and AIC | R `glm()` | Relative error < 1e-6 | | |
| OQ-GLM-006 | GLM convergence on separable data | Near-separable logistic data | Converges or reports separation warning | | |
| OQ-GLM-007 | GLM prediction (link function) | Manual calculation | Predicted probabilities match to 1e-8 | | |
| OQ-GLM-008 | GLM with offset term | R `glm(offset=...)` | Relative error < 1e-6 per coefficient | | |

### 5.10 Linear Mixed Models (5 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-LMM-001 | REML fixed-effect estimates | R `lme4::lmer()` | Relative error < 1e-6 per coefficient | | |
| OQ-LMM-002 | REML variance components | R `lme4::lmer()` | Relative error < 1e-4 per variance | | |
| OQ-LMM-003 | Random effects BLUPs | R `lme4::ranef()` | Relative error < 1e-4 per BLUP | | |
| OQ-LMM-004 | LMM log-likelihood | R `logLik(lmer(...))` | \|logLik_NS - logLik_R\| < 1e-4 | | |
| OQ-LMM-005 | LMM with crossed random effects | R `lme4::lmer()` | Variance components < 1e-3 relative error | | |

### 5.11 Bootstrap Confidence Intervals (4 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-BOOT-001 | Percentile CI (95%) | R `boot::boot.ci(type="perc")` | Relative error < 1e-4 for CI bounds | | |
| OQ-BOOT-002 | BCa CI (95%) | R `boot::boot.ci(type="bca")` | Relative error < 1e-4 for CI bounds | | |
| OQ-BOOT-003 | BCa acceleration and z0 diagnostics | R `boot::boot.ci()` | \|a_NS - a_R\| < 1e-3 and \|z0_NS - z0_R\| < 1e-3 | | |
| OQ-BOOT-004 | Bootstrap reproducibility (fixed seed) | Two runs, seed=42 | CI bounds identical between runs | | |

### 5.12 Emax/Hill Dose-Response (3 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-EMAX-001 | Emax model predictions at 5 doses | E(d) = E0 + Emax*d/(ED50+d) | Relative error < 1e-12 | | |
| OQ-EMAX-002 | Hill model (sigmoid) predictions | E(d) = E0 + Emax*d^n/(ED50^n+d^n) | Relative error < 1e-12 | | |
| OQ-EMAX-003 | Emax MLE parameter recovery | Simulated dose-response data | Relative error < 10% per parameter | | |

### 5.13 ODE Solver Verification (3 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-ODE-001 | RK45 vs analytical 1-cpt oral at tol=1e-8 | Analytical C(t) | Relative error < 1e-6 at all time points | | |
| OQ-ODE-002 | RK45 vs analytical 2-cpt IV at tol=1e-8 | Analytical C(t) | Relative error < 1e-6 at all time points | | |
| OQ-ODE-003 | Stiff system detection and handling | Stiff 2-cpt with extreme Q/V ratio | Solver returns finite results or reports stiffness | | |

### 5.14 HEP / HistFactory (10 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-HEP-001 | NLL evaluation on reference workspace | pyhf 0.7.x | \|NLL_NS - NLL_pyhf\| < 1e-8 | | |
| OQ-HEP-002 | MLE fit: parameter estimates | pyhf best-fit | Relative error < 1e-6 per parameter | | |
| OQ-HEP-003 | Profile likelihood q(mu) | pyhf q(mu) | \|q_NS - q_pyhf\| < 1e-5 | | |
| OQ-HEP-004 | CLs exclusion limit | pyhf CLs | \|CLs_NS - CLs_pyhf\| < 1e-4 | | |
| OQ-HEP-005 | Pulls (post-fit NP shifts) | pyhf pulls | Absolute error < 1e-4 per NP | | |
| OQ-HEP-006 | Ranking (impact ordering) | pyhf ranking | Same top-5 NP ordering | | |
| OQ-HEP-007 | Profile scan curve | pyhf profile | \|delta_NLL\| < 1e-4 at each scan point | | |
| OQ-HEP-008 | Toy-based hypothesis test | pyhf toys | \|p-value_NS - p-value_pyhf\| < 0.05 | | |
| OQ-HEP-009 | Multi-channel workspace | pyhf multi-channel | NLL and fit parity with single-channel | | |
| OQ-HEP-010 | Systematic interpolation (Code1/Code4) | pyhf interpolation | \|modifier_NS - modifier_pyhf\| < 1e-8 | | |

### 5.15 Bayesian Sampling -- NUTS/MAMS/LAPS (8 tests)

| ID | Test Description | Reference | Tolerance | Result | Pass/Fail |
|----|-----------------|-----------|-----------|--------|-----------|
| OQ-MCMC-001 | NUTS StdNormal 10D: posterior mean | Analytical: mean=0 | \|mean\| < 0.1 per dimension | | |
| OQ-MCMC-002 | NUTS StdNormal 10D: posterior variance | Analytical: var=1 | \|var - 1.0\| < 0.1 per dimension | | |
| OQ-MCMC-003 | NUTS R-hat < 1.05 | Stan diagnostics convention | R-hat < 1.05 for all parameters | | |
| OQ-MCMC-004 | NUTS no divergent transitions (StdNormal) | Stan diagnostics convention | n_divergent == 0 | | |
| OQ-MCMC-005 | NUTS Eight Schools: posterior mean of mu | CmdStan reference | \|mu_NS - mu_Stan\| < 1.0 | | |
| OQ-MCMC-006 | LAPS GPU NealFunnel NCP: R-hat < 1.05 | Internal benchmark | R-hat < 1.05 for all parameters | | |
| OQ-MCMC-007 | LAPS GPU StdNormal: ESS_tail > 1000 | Internal benchmark | ESS_tail > 1000 | | |
| OQ-MCMC-008 | Sampling reproducibility (fixed seed) | Two runs, seed=42 | Posterior means identical to 1e-10 | | |

## 6. Automated Execution

The OQ test suite can be executed automatically using the built-in validation harness:

```bash
# Run all OQ tests and generate JSON report
python -m nextstat.validate.oq --output oq_results.json

# Run specific domain
python -m nextstat.validate.oq --domain survival --output oq_survival.json

# Run with verbose output (shows each test case)
python -m nextstat.validate.oq --verbose --output oq_results.json
```

The automated harness:
1. Executes each test case with documented input parameters
2. Compares output against stored reference values
3. Evaluates acceptance criteria
4. Generates a timestamped JSON report with pass/fail status per test
5. Records NextStat version, platform, and execution environment

## 7. Acceptance Criteria

OQ is considered **PASSED** when:

1. All 93 test cases record a **PASS** result.
2. Zero tolerance violations across all domains.
3. Execution log is saved with timestamps, platform details, and NextStat version.
4. Any deviations are documented per the deviation handling procedure in VSR_Template.md.

OQ is considered **FAILED** if any test case cannot meet its acceptance criteria after
investigation and corrective action. Critical failures block PQ execution.

## 8. Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Executed By | ________________________________ | ________________________________ | ____/____/________ |
| Reviewed By | ________________________________ | ________________________________ | ____/____/________ |
| Approved By (QA) | ________________________________ | ________________________________ | ____/____/________ |

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-21 | NextStat Team | Initial release |
