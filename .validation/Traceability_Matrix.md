# Traceability Matrix

## Document Control

| Field | Value |
|-------|-------|
| Document ID | NS-VAL-TM-001 |
| Version | 1.0 |
| Effective Date | ____/____/________ |
| Classification | GxP Validation Deliverable |

---

## 1. Purpose

This document provides a bidirectional traceability matrix mapping each functional
requirement to its corresponding IQ, OQ, and PQ test cases. Traceability ensures
that every requirement is verified by at least one test and that every test traces
back to a documented requirement.

This matrix satisfies:
- 21 CFR Part 11 Section 11.10(a) -- validation evidence for computerised systems
- EU Annex 11 Section 4.4 -- documented traceability from requirements to tests
- GAMP 5 Section D.5 -- traceability matrix as a validation deliverable

## 2. Requirement-to-Test Mapping

### 2.1 PK Analytical Models

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-PK-001 | 1-compartment oral PK concentration is mathematically correct | OQ-PK-001, OQ-PK-002 | PQ-REF-001, PQ-REF-002 | High |
| REQ-PK-002 | 2-compartment IV PK concentration is mathematically correct | OQ-PK-003, OQ-PK-004 | -- | High |
| REQ-PK-003 | 2-compartment oral PK concentration is mathematically correct | OQ-PK-005 | -- | High |
| REQ-PK-004 | Analytical gradients are correct for all PK models | OQ-PK-006 | -- | High |
| REQ-PK-005 | Degenerate parameter cases handled without numerical failure | OQ-PK-007 | -- | Medium |
| REQ-PK-006 | LLOQ censored likelihood is computed correctly | OQ-PK-008 | -- | Medium |

### 2.2 Population PK Estimation (FOCE/FOCEI)

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-NLME-001 | FOCE converges on synthetic and reference population data | OQ-FOCE-001, OQ-FOCE-006 | PQ-WARF-002 | High |
| REQ-NLME-002 | FOCE fixed-effect (theta) estimates recover true values | OQ-FOCE-001, OQ-FOCE-003 | PQ-WARF-004..006, PQ-XVAL-002 | High |
| REQ-NLME-003 | FOCE random-effect (omega) estimates recover true values | OQ-FOCE-005 | PQ-WARF-007..008 | High |
| REQ-NLME-004 | FOCE OFV is numerically correct | OQ-FOCE-002, OQ-FOCE-003 | PQ-WARF-003, PQ-XVAL-001 | High |
| REQ-NLME-005 | FOCEI interaction terms computed correctly | OQ-FOCE-004 | -- | High |
| REQ-NLME-006 | FOCE handles correlated Omega (off-diagonal elements) | OQ-FOCE-005 | -- | High |

### 2.3 Population PK Estimation (SAEM)

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-SAEM-001 | SAEM converges on synthetic population data | OQ-SAEM-001 | PQ-WARF-009 | High |
| REQ-SAEM-002 | SAEM theta estimates recover true values | OQ-SAEM-002 | PQ-WARF-010 | High |
| REQ-SAEM-003 | SAEM omega estimates recover true values | OQ-SAEM-003 | -- | High |
| REQ-SAEM-004 | SAEM MCMC diagnostics are within expected ranges | OQ-SAEM-004 | -- | Medium |
| REQ-SAEM-005 | SAEM is reproducible with fixed seed | OQ-SAEM-005 | PQ-WARF-017, PQ-REPR-002 | High |

### 2.4 Error Models

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-ERR-001 | Additive error model variance is correct | OQ-ERR-001 | -- | High |
| REQ-ERR-002 | Proportional error model variance is correct | OQ-ERR-002 | PQ-WARF-002 | High |
| REQ-ERR-003 | Combined error model variance is correct | OQ-ERR-003 | -- | High |
| REQ-ERR-004 | Error model rejects invalid input parameters | OQ-ERR-004 | -- | Medium |

### 2.5 Stepwise Covariate Modeling (SCM)

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-SCM-001 | Forward selection identifies statistically significant covariates | OQ-SCM-001 | -- | High |
| REQ-SCM-002 | Forward selection rejects non-significant covariates | OQ-SCM-002 | -- | High |
| REQ-SCM-003 | Backward elimination removes non-significant covariates | OQ-SCM-003 | -- | High |
| REQ-SCM-004 | Covariate coefficients are recovered accurately | OQ-SCM-004 | -- | High |

### 2.6 Diagnostics (GOF and VPC)

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-DIAG-001 | PRED computed correctly (population prediction, eta=0) | OQ-GOF-001 | PQ-WARF-015 | Medium |
| REQ-DIAG-002 | IPRED computed correctly (individual prediction) | OQ-GOF-002 | PQ-WARF-014, PQ-XVAL-004 | Medium |
| REQ-DIAG-003 | IWRES/CWRES approximately N(0,1) under correct model | OQ-GOF-003 | PQ-WARF-012, PQ-XVAL-005 | Medium |
| REQ-DIAG-004 | VPC prediction intervals have correct coverage | OQ-GOF-004, OQ-GOF-005 | PQ-WARF-013 | Medium |

### 2.7 NONMEM Dataset I/O

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-IO-001 | NONMEM-format CSV files parsed correctly | OQ-IO-001, OQ-IO-004 | PQ-WARF-001 | High |
| REQ-IO-002 | EVID/MDV column semantics correct | OQ-IO-002 | -- | High |
| REQ-IO-003 | Missing required columns detected and reported | OQ-IO-003 | -- | High |

### 2.8 Survival Analysis

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-SURV-001 | Cox PH coefficients match R `survival` reference | OQ-SURV-001, OQ-SURV-002, OQ-SURV-003 | PQ-PERF-006 | High |
| REQ-SURV-002 | Schoenfeld residuals computed correctly | OQ-SURV-004 | -- | Medium |
| REQ-SURV-003 | Kaplan-Meier survival probabilities match R reference | OQ-SURV-005, OQ-SURV-006, OQ-SURV-007 | -- | High |
| REQ-SURV-004 | Log-rank test statistic matches R reference | OQ-SURV-008 | -- | High |
| REQ-SURV-005 | Breslow baseline hazard matches R reference | OQ-SURV-009 | -- | Medium |
| REQ-SURV-006 | Stratified Cox PH matches R reference | OQ-SURV-010 | -- | Medium |

### 2.9 Generalized Linear Models

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-GLM-001 | Logistic regression coefficients match R `glm()` | OQ-GLM-001, OQ-GLM-002, OQ-GLM-003 | PQ-PERF-007 | High |
| REQ-GLM-002 | Poisson regression coefficients match R `glm()` | OQ-GLM-004, OQ-GLM-005 | -- | High |
| REQ-GLM-003 | GLM handles near-separable data gracefully | OQ-GLM-006 | -- | Medium |
| REQ-GLM-004 | GLM predictions use correct link function | OQ-GLM-007 | -- | High |
| REQ-GLM-005 | GLM with offset term matches R reference | OQ-GLM-008 | -- | Medium |

### 2.10 Linear Mixed Models

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-LMM-001 | REML fixed-effect estimates match R `lme4` | OQ-LMM-001 | -- | High |
| REQ-LMM-002 | REML variance components match R `lme4` | OQ-LMM-002 | -- | High |
| REQ-LMM-003 | Random effects BLUPs match R `lme4::ranef()` | OQ-LMM-003 | -- | Medium |
| REQ-LMM-004 | LMM log-likelihood matches R reference | OQ-LMM-004 | -- | Medium |
| REQ-LMM-005 | Crossed random effects handled correctly | OQ-LMM-005 | -- | Medium |

### 2.11 Bootstrap Confidence Intervals

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-BOOT-001 | Percentile CI matches R `boot` reference | OQ-BOOT-001 | -- | High |
| REQ-BOOT-002 | BCa CI matches R `boot` reference | OQ-BOOT-002, OQ-BOOT-003 | -- | High |
| REQ-BOOT-003 | Bootstrap is reproducible with fixed seed | OQ-BOOT-004 | -- | High |

### 2.12 Dose-Response (Emax/Hill)

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-EMAX-001 | Emax model predictions are mathematically correct | OQ-EMAX-001 | -- | High |
| REQ-EMAX-002 | Hill (sigmoid) model predictions are correct | OQ-EMAX-002 | -- | High |
| REQ-EMAX-003 | Emax MLE recovers true parameters | OQ-EMAX-003 | -- | Medium |

### 2.13 ODE Solver

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-ODE-001 | RK45 solver matches analytical PK solutions | OQ-ODE-001, OQ-ODE-002 | -- | High |
| REQ-ODE-002 | Stiff systems handled without failure | OQ-ODE-003 | -- | Medium |

### 2.14 HEP / HistFactory

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-HEP-001 | NLL evaluation matches pyhf reference | OQ-HEP-001 | -- | High |
| REQ-HEP-002 | MLE fit parameters match pyhf reference | OQ-HEP-002 | -- | High |
| REQ-HEP-003 | Profile likelihood q(mu) matches pyhf | OQ-HEP-003 | -- | High |
| REQ-HEP-004 | CLs exclusion limit matches pyhf | OQ-HEP-004 | -- | High |
| REQ-HEP-005 | Pulls and ranking match pyhf ordering | OQ-HEP-005, OQ-HEP-006 | -- | Medium |
| REQ-HEP-006 | Systematic interpolation codes match pyhf | OQ-HEP-010 | -- | High |

### 2.15 Bayesian Sampling (NUTS/MAMS/LAPS)

| Requirement ID | Requirement Description | OQ Test Case(s) | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|-----------------|------------|
| REQ-MCMC-001 | NUTS posterior moments match analytical solutions | OQ-MCMC-001, OQ-MCMC-002 | -- | High |
| REQ-MCMC-002 | NUTS R-hat diagnostics within Stan conventions | OQ-MCMC-003, OQ-MCMC-004 | -- | High |
| REQ-MCMC-003 | NUTS matches CmdStan on Eight Schools | OQ-MCMC-005 | -- | Medium |
| REQ-MCMC-004 | LAPS GPU sampler converges (R-hat, ESS) | OQ-MCMC-006, OQ-MCMC-007 | -- | Medium |
| REQ-MCMC-005 | Sampling is reproducible with fixed seed | OQ-MCMC-008 | -- | High |

### 2.16 Installation and Platform

| Requirement ID | Requirement Description | IQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|------------|
| REQ-INST-001 | Package installs from PyPI without error | IQ-INST-001..004 | High |
| REQ-INST-002 | Binary integrity verified via SHA-256 | IQ-CHK-001..003 | High |
| REQ-INST-003 | Platform requirements met | IQ-PLT-001..005 | Medium |
| REQ-INST-004 | All API domains importable | IQ-API-001..005 | High |
| REQ-INST-005 | No dependency conflicts | IQ-DEP-001..003 | Medium |

### 2.17 Reproducibility and Provenance

| Requirement ID | Requirement Description | PQ Test Case(s) | Risk Level |
|---------------|------------------------|-----------------|------------|
| REQ-REPR-001 | Deterministic execution for fixed inputs | PQ-REPR-001, PQ-REPR-002 | High |
| REQ-REPR-002 | Artifact round-trip fidelity | PQ-REPR-003 | Medium |
| REQ-REPR-003 | RunBundle provenance is complete | PQ-REPR-004 | Medium |

## 3. Coverage Summary

| Domain | Requirements | OQ Tests | PQ Tests | Coverage |
|--------|-------------|----------|----------|----------|
| PK Analytical | 6 | 8 | 2 | Complete |
| FOCE/FOCEI | 6 | 6 | 8 | Complete |
| SAEM | 5 | 5 | 3 | Complete |
| Error Models | 4 | 4 | 1 | Complete |
| SCM | 4 | 4 | 0 | OQ only |
| Diagnostics (GOF/VPC) | 4 | 5 | 4 | Complete |
| NONMEM I/O | 3 | 4 | 1 | Complete |
| Survival | 6 | 10 | 1 | Complete |
| GLM | 5 | 8 | 1 | Complete |
| LMM | 5 | 5 | 0 | OQ only |
| Bootstrap CI | 3 | 4 | 0 | OQ only |
| Emax/Hill | 3 | 3 | 0 | OQ only |
| ODE Solver | 2 | 3 | 0 | OQ only |
| HEP/HistFactory | 6 | 10 | 0 | OQ only |
| Bayesian Sampling | 5 | 8 | 0 | OQ only |
| Installation | 5 | -- (IQ) | 0 | IQ only |
| Reproducibility | 3 | 0 | 4 | PQ only |
| **Total** | **75** | **93** | **25** | **Complete** |

## 4. Orphan Analysis

### 4.1 Requirements Without Tests
None. All 75 requirements are covered by at least one test case.

### 4.2 Tests Without Requirements
None. All 93 OQ tests and 25 PQ tests trace to at least one requirement.

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-21 | NextStat Team | Initial release |
