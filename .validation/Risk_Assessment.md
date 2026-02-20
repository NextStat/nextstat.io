# Risk Assessment -- Failure Mode and Effects Analysis (FMEA)

## Document Control

| Field | Value |
|-------|-------|
| Document ID | NS-VAL-RA-001 |
| Version | 1.0 |
| Effective Date | ____/____/________ |
| Classification | GxP Validation Deliverable |
| Author | ________________________________ |
| Reviewer | ________________________________ |
| Approver | ________________________________ |

---

## 1. Purpose

This document presents a Failure Mode and Effects Analysis (FMEA) for the NextStat
statistical computing platform, identifying potential failure modes in computational
functions, assessing their impact on GxP-regulated activities, and documenting
mitigating controls.

This risk assessment follows the GAMP 5 risk-based approach (ISPE GAMP 5 Second Edition,
2022) and satisfies:
- 21 CFR Part 11 Section 11.10(a) -- risk-based validation
- EU Annex 11 Section 1 -- risk management throughout the lifecycle
- ICH Q9 -- Quality Risk Management principles

## 2. Methodology

### 2.1 Risk Scoring

Each failure mode is scored on three dimensions:

**Severity (S)** -- Impact on patient safety, data integrity, or regulatory compliance

| Score | Level | Description |
|-------|-------|-------------|
| 1 | Negligible | No impact on regulatory decisions or patient safety |
| 2 | Low | Minor documentation impact; no effect on conclusions |
| 3 | Medium | Incorrect intermediate result; may affect model selection |
| 4 | High | Incorrect parameter estimate; could affect dosing recommendation |
| 5 | Critical | Directly impacts patient safety through incorrect dose selection |

**Probability (P)** -- Likelihood of the failure mode occurring

| Score | Level | Description |
|-------|-------|-------------|
| 1 | Very Low | Failure requires multiple independent faults; never observed |
| 2 | Low | Theoretically possible but not observed in testing |
| 3 | Medium | Observed in edge cases or extreme parameter ranges |
| 4 | High | Observed under normal operating conditions |
| 5 | Very High | Expected to occur during routine use |

**Detection (D)** -- Ability to detect the failure before it causes harm (inverse scale)

| Score | Level | Description |
|-------|-------|-------------|
| 1 | Very High | Automatic detection: software raises error or warning |
| 2 | High | Detected by standard diagnostics (convergence flag, GOF, VPC) |
| 3 | Medium | Detected by cross-validation with reference implementation |
| 4 | Low | Requires expert review of detailed numerical output |
| 5 | Very Low | Undetectable without independent re-analysis |

**Risk Priority Number (RPN)** = Severity x Probability x Detection

| RPN Range | Risk Level | Action Required |
|-----------|------------|-----------------|
| 1-12 | Low | Accept risk; standard validation sufficient |
| 13-36 | Medium | Additional controls recommended; documented justification |
| 37-75 | High | Mandatory mitigation; enhanced testing required |
| 76-125 | Critical | Unacceptable; must implement controls before deployment |

### 2.2 Scope

This FMEA covers all computational functions in NextStat that produce results used in
regulated analyses: PK modeling, population estimation, diagnostics, survival analysis,
GLM, LMM, bootstrap, dose-response, Bayesian sampling, and data I/O.

## 3. FMEA Table

### 3.1 MLE Optimization

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-001 | MLE optimization (L-BFGS-B) | Non-convergence: optimizer reaches max iterations without meeting gradient tolerance | Parameter estimates may be inaccurate; OFV not at true minimum | 4 | 2 | 1 | 8 | Convergence flag (`converged`) reported in output. Max iterations configurable. OQ-MLE-004 verifies convergence detection. User SOP: always check convergence status. |
| RA-002 | MLE optimization (L-BFGS-B) | Convergence to local minimum instead of global minimum | Wrong parameter estimates; suboptimal model fit | 4 | 2 | 3 | 24 | Multi-start optimization available. OQ-MLE-005 verifies NLL is minimal vs perturbations. VPC and GOF diagnostics detect poor fit. |
| RA-003 | MLE optimization (L-BFGS-B) | Optimizer returns parameters at boundary constraints | Parameter estimate clamped; may not reflect true population value | 3 | 2 | 2 | 12 | Active bounds count reported in FitResult (`n_active_bounds`). Termination reason documented. User can inspect and widen bounds. |

### 3.2 Gradient Computation

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-004 | Analytical PK gradients | Numerical error in gradient (e.g., near-degenerate eigenvalues) | Slow convergence or convergence to wrong parameters | 4 | 1 | 2 | 8 | Finite-difference fallback for degenerate cases (Ka approx Ke). OQ-PK-006 verifies gradient vs FD parity. OQ-PK-007/008 test degenerate cases. |
| RA-005 | Reverse-mode AD (tape) | Gradient computation overflow/underflow for extreme parameter values | NaN/Inf in gradient; optimizer stalls | 3 | 1 | 1 | 3 | Automatic NaN detection in optimizer. Error reported to user. Parameter bounds prevent extreme values. |

### 3.3 PK Analytical Models

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-006 | conc_oral() | Mathematical error in concentration formula | Incorrect PK predictions; wrong dosing decisions | 5 | 1 | 2 | 10 | Formula verified against closed-form derivation. OQ-PK-001 tests 10 time points against independent calculation to 1e-10 precision. |
| RA-007 | conc_iv_2cpt() | Eigenvalue computation error for extreme compartment parameters | NaN concentration or negative values | 4 | 1 | 1 | 4 | Degenerate case handling with FD fallback. OQ-PK-008 tests near-degenerate discriminant. isfinite() guard in output. |
| RA-008 | Multi-dose superposition | Incorrect accumulation across dosing intervals | Wrong steady-state predictions | 4 | 1 | 3 | 12 | OQ-PK-007 (planned) verifies multi-dose against manual calculation. GOF diagnostics detect systematic prediction bias. |

### 3.4 Population PK Estimation

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-009 | FOCE estimator | Non-convergence on complex models (many parameters, correlated omegas) | No result produced; analysis delayed | 3 | 2 | 1 | 6 | Convergence flag and iteration count reported. OQ-FOCE-006 tests convergence within max iterations. SAEM available as alternative estimator. |
| RA-010 | FOCE estimator | Omega matrix estimated as non-positive-definite | Invalid variance-covariance structure | 4 | 1 | 1 | 4 | Cholesky parameterization enforces positive-definiteness by construction. OQ-FOCE-011 (pharma protocol) verifies eigenvalues. |
| RA-011 | FOCE Laplace approximation | Poor Laplace approximation for non-normal random effects | Biased parameter estimates | 3 | 2 | 3 | 18 | SAEM (which does not rely on Laplace) available as cross-check. OQ-SAEM-006 compares FOCE vs SAEM. VPC detects model misspecification. |
| RA-012 | SAEM estimator | Insufficient burn-in leads to poor convergence | Biased theta/omega estimates | 3 | 2 | 2 | 12 | OFV trace monitored for stabilization (OQ-SAEM-005). Configurable burn-in (default 200). OQ-SAEM-008 (pharma protocol) tests burn-in length impact. |
| RA-013 | SAEM MCMC | Poor mixing of MCMC chains within SAEM | Slow convergence; potentially biased parameter estimates | 3 | 2 | 2 | 12 | Acceptance rate diagnostics reported (target 0.15-0.60). Adaptive Metropolis-Hastings step-size. OQ-SAEM-004 verifies acceptance rates. |

### 3.5 Error Models

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-014 | ErrorModel (all types) | Wrong error model selected (e.g., additive instead of proportional) | Biased parameter SEs; incorrect confidence intervals | 4 | 2 | 2 | 16 | User-specified error model type. IWRES diagnostics reveal misspecification (non-constant variance). OQ-ERR-008/009 test residual distributions. |
| RA-015 | ErrorModel (combined) | Sigma components estimated near zero (unidentifiable) | Numerical instability in variance computation | 3 | 2 | 2 | 12 | Bounds on sigma parameters prevent zero. OQ-ERR-003 verifies combined model correctness. |

### 3.6 LLOQ Handling

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-016 | LloqPolicy::Ignore | BLQ data silently excluded; bias in low-concentration parameters | Biased CL estimate (positive bias) in presence of high BLQ fraction | 3 | 3 | 3 | 27 | OQ-LLOQ-004 quantifies bias vs Censored method. User guidance recommends Censored for BLQ > 10%. BLQ count reported in output. |
| RA-017 | LloqPolicy::Censored | Incorrect Phi() computation for censored likelihood | Wrong likelihood contribution for BLQ observations | 4 | 1 | 2 | 8 | OQ-PK-008 verifies censored NLL against manual calculation. OQ-LLOQ-003 verifies observation count. |

### 3.7 Covariate Modeling (SCM)

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-018 | SCM forward selection | False positive: noise covariate selected | Overfitted model; spurious covariate in label | 4 | 2 | 2 | 16 | Backward elimination step removes false positives. OQ-SCM-002 tests noise rejection. Conservative alpha thresholds (0.05 forward, 0.01 backward). |
| RA-019 | SCM backward elimination | False negative: true covariate removed | Underfitted model; clinically relevant covariate missed | 4 | 2 | 3 | 24 | Clinical judgment required for final model selection. OQ-SCM-001 verifies true covariate detection. p-values and delta_OFV reported for audit. |

### 3.8 Data I/O

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-020 | read_nonmem() CSV parser | Wrong column mapping (e.g., DV and AMT swapped) | Completely incorrect model; all parameters wrong | 5 | 1 | 1 | 5 | Automatic header validation. OQ-IO-001..010 test parsing. Column names are explicit. Missing columns raise error (OQ-IO-003). |
| RA-021 | read_nonmem() CSV parser | Numeric parsing error (comma vs period decimal separator) | Silent data corruption; biased parameters | 5 | 1 | 2 | 10 | Standard NONMEM format uses period. Input data should be inspected before analysis. GOF diagnostics detect gross data errors. |
| RA-022 | read_nonmem() CSV parser | MDV/EVID flags ignored or misinterpreted | Dosing events treated as observations or vice versa | 5 | 1 | 1 | 5 | OQ-IO-002 explicitly tests EVID/MDV semantics. Record counts reported and can be verified against source. |

### 3.9 Diagnostics

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-023 | GOF (CWRES) | CWRES computation error | Misleading diagnostic; misspecified model accepted | 3 | 1 | 3 | 9 | OQ-GOF-003/004 verify CWRES distribution. Cross-check with VPC. Multiple diagnostic endpoints reduce single-metric reliance. |
| RA-024 | VPC | Too few simulations produce noisy prediction intervals | Model appears misspecified when it is not | 2 | 2 | 2 | 8 | Default n_sim=200. OQ-VPC-003 tests coverage with n_sim=500. User can increase n_sim. Seed-based reproducibility (OQ-VPC-005). |

### 3.10 Survival Analysis

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-025 | Cox PH | Numerical error in partial likelihood with tied event times | Incorrect hazard ratios | 3 | 2 | 2 | 12 | Breslow/Efron tie-handling methods implemented. OQ-SURV-001 verifies against R `survival` (which uses Efron by default). |
| RA-026 | Kaplan-Meier | Incorrect handling of right-censored observations | Wrong survival probabilities | 4 | 1 | 2 | 8 | OQ-SURV-005..007 verify survival probabilities and censoring against R reference. |

### 3.11 Bayesian Sampling

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-027 | NUTS sampler | Divergent transitions in funnel-like geometries | Biased posterior; incorrect credible intervals | 3 | 2 | 1 | 6 | Divergence count reported. R-hat and ESS diagnostics flag non-convergence. Non-centered parametrization available. OQ-MCMC-003/004 verify diagnostics. |
| RA-028 | LAPS GPU sampler | GPU memory exhaustion with large chain count | Process crash; no result | 2 | 2 | 1 | 4 | Memory requirement estimated before launch. Error message on allocation failure. Automatic fallback strategy documented. |

### 3.12 Platform and Environment

| ID | Function | Failure Mode | Effect | S | P | D | RPN | Mitigation |
|----|----------|-------------|--------|---|---|---|-----|------------|
| RA-029 | Cross-platform execution | f64 rounding differences between x86_64 and ARM64 | Results differ at last significant digit across platforms | 1 | 3 | 4 | 12 | Deterministic f64 computation on each platform. OQ tolerances account for platform ULP differences. Results are reproducible on the same platform. |
| RA-030 | Python version compatibility | API behavior change in new Python version | Import error or incorrect data marshalling | 3 | 1 | 1 | 3 | Wheels built for Python 3.9-3.13. IQ verifies import. CI tests all supported versions. |

## 4. Risk Summary

### 4.1 RPN Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| Low (RPN 1-12) | 20 | 67% |
| Medium (RPN 13-36) | 10 | 33% |
| High (RPN 37-75) | 0 | 0% |
| Critical (RPN 76-125) | 0 | 0% |

### 4.2 Highest-RPN Failure Modes

| Rank | ID | Function | Failure Mode | RPN | Status |
|------|-----|----------|-------------|-----|--------|
| 1 | RA-016 | LLOQ Ignore | BLQ bias with high censoring rate | 27 | Mitigated: OQ test + user guidance |
| 2 | RA-002 | MLE optimization | Local minimum convergence | 24 | Mitigated: multi-start + diagnostics |
| 3 | RA-019 | SCM backward | False negative covariate removal | 24 | Mitigated: p-value reporting + clinical review |
| 4 | RA-011 | FOCE Laplace | Poor approximation bias | 18 | Mitigated: SAEM cross-check |
| 5 | RA-014 | Error model | Wrong model selection | 16 | Mitigated: IWRES diagnostics |
| 5 | RA-018 | SCM forward | False positive covariate | 16 | Mitigated: backward elimination step |

### 4.3 Residual Risk Statement

After implementation of the mitigating controls documented above, no failure modes
have an RPN exceeding 27 (Medium risk level). All Medium-risk items have documented
mitigations including OQ test coverage, diagnostic outputs, and user guidance. The
residual risk is acceptable for use in GxP-regulated environments, provided that:

1. Users follow the recommended SOPs for model development (convergence checking,
   diagnostic review, cross-validation).
2. The OQ test suite is executed and passes on each qualified installation.
3. Results from NextStat are reviewed by qualified pharmacometricians or biostatisticians.

## 5. Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Risk Assessment Author | ________________________________ | ________________________________ | ____/____/________ |
| Subject Matter Expert | ________________________________ | ________________________________ | ____/____/________ |
| Quality Assurance | ________________________________ | ________________________________ | ____/____/________ |

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-21 | NextStat Team | Initial release |
