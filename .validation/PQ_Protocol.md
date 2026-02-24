# Performance Qualification (PQ) Protocol

## Document Control

| Field | Value |
|-------|-------|
| Document ID | NS-VAL-PQ-001 |
| Version | 1.0 |
| Effective Date | ____/____/________ |
| Supersedes | N/A |
| Classification | GxP Validation Deliverable |
| Author | ________________________________ |
| Reviewer | ________________________________ |
| Approver | ________________________________ |

---

## 1. Purpose

This document defines the Performance Qualification (PQ) protocol for verifying that
NextStat performs correctly under conditions representative of the organization's intended
use. PQ demonstrates fitness for purpose by executing site-specific workflows with
real or representative datasets and comparing results against established reference
implementations.

While IQ verifies installation and OQ verifies algorithmic correctness, PQ bridges the
gap to operational reality by confirming that the software produces acceptable results
in the user's specific computational environment and workflow context.

### 1.1 Regulatory Basis

| Regulation | PQ Requirement |
|-----------|---------------|
| 21 CFR Part 11 Section 11.10(a) | Validated systems must produce accurate results under conditions of use |
| EU Annex 11 Section 4.7 | Accuracy checks at appropriate intervals; data entered checked for correctness |
| ICH E6(R3) Section 4.9 | Systems used in clinical trials validated for their intended purpose |
| GAMP 5 Section 7.7 | PQ verifies fitness for purpose in the user's operational environment |

## 2. Scope

PQ tests are site-specific. This protocol provides:
1. A blank template for documenting site-specific test cases (Section 4)
2. A worked example using the Warfarin population PK dataset (Section 5)
3. Cross-validation procedures against NONMEM and other reference tools (Section 6)
4. Reproducibility verification (Section 7)
5. Performance baseline recording (Section 8)

Organizations should customize this protocol by adding test cases that reflect their
standard pharmacometric and biostatistical workflows.

## 3. Prerequisites

| Prerequisite | Status |
|-------------|--------|
| IQ Protocol (NS-VAL-IQ-001) completed and passed | ______ |
| OQ Protocol (NS-VAL-OQ-001) completed and passed | ______ |
| Reference datasets available in NONMEM format | ______ |
| Reference results from comparator software (NONMEM, R, SAS) available | ______ |
| Qualified test environment documented | ______ |

## 4. Site-Specific Test Case Template

Use this template to document each PQ test case. Copy and complete for each workflow
your organization validates.

---

### PQ Test Case: ________________________________________

| Field | Value |
|-------|-------|
| Test ID | PQ-SITE-___ |
| Workflow | ________________________________ |
| Dataset | ________________________________ |
| Model | ________________________________ |
| Reference Tool | ________________________________ |
| Executed By | ________________________________ |
| Date | ____/____/________ |

#### Input Specification

| Parameter | Value |
|-----------|-------|
| Dataset file | ________________________________ |
| Dataset SHA-256 | ________________________________ |
| Number of subjects | ________________________________ |
| Number of observations | ________________________________ |
| Structural model | ________________________________ |
| Error model | ________________________________ |
| Random effects specification | ________________________________ |
| Estimation method | ________________________________ |
| Covariates (if applicable) | ________________________________ |
| NextStat function call | ________________________________ |

#### Acceptance Criteria

| Criterion | Threshold | Observed | Pass/Fail |
|-----------|-----------|----------|-----------|
| Convergence | converged == True | | |
| Fixed effects (theta) | Relative error vs reference < ___% | | |
| Random effects (omega) | Relative error vs reference < ___% | | |
| Residual error (sigma) | Relative error vs reference < ___% | | |
| OFV | \|OFV_NS - OFV_ref\| < ___ | | |
| Execution time | < ___ seconds | | |

#### Results

| Parameter | NextStat Estimate | Reference Estimate | Relative Error | Pass/Fail |
|-----------|------------------|--------------------|----------------|-----------|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

#### Deviation Report (if applicable)

| Field | Value |
|-------|-------|
| Deviation ID | DEV-PQ-___ |
| Classification | Critical / Major / Minor |
| Description | ________________________________ |
| Root Cause | ________________________________ |
| Corrective Action | ________________________________ |
| Re-test Date | ____/____/________ |
| Re-test Result | Pass / Fail |

---

## 5. Worked Example: Warfarin Population PK

This section provides a complete PQ example using the Warfarin dataset, a standard
pharmacometric reference. Organizations may use this as-is or as a template for their
own datasets.

### 5.1 Dataset Description

| Property | Value |
|----------|-------|
| Drug | Warfarin |
| Source | O'Reilly RA (1968). Clin Pharmacol Ther; 11:378-384. |
| Subjects | 32 patients |
| Observations | ~251 |
| Model | 1-compartment oral PK |
| Parameters | CL (clearance), V (volume), Ka (absorption rate) |
| Random effects | Log-normal on CL, V |
| Error model | Proportional |
| Dataset file | `data/warfarin.csv` |

### 5.2 FOCE Workflow

| Step | Action | Expected | Observed | Pass/Fail |
|------|--------|----------|----------|-----------|
| PQ-WARF-001 | Load dataset: `data = nextstat.read_nonmem("data/warfarin.csv")` | No error. n_subjects == 32. | | |
| PQ-WARF-002 | Fit FOCE: `result = nextstat.nlme_foce(...)` with proportional error | converged == True | | |
| PQ-WARF-003 | Compare OFV to NONMEM reference | \|OFV_NS - OFV_NM\| < 5.0 | | |
| PQ-WARF-004 | Compare CL estimate to NONMEM | Relative error < 20% | | |
| PQ-WARF-005 | Compare V estimate to NONMEM | Relative error < 20% | | |
| PQ-WARF-006 | Compare Ka estimate to NONMEM | Relative error < 20% | | |
| PQ-WARF-007 | Compare omega_CL to NONMEM | Relative error < 30% | | |
| PQ-WARF-008 | Compare omega_V to NONMEM | Relative error < 30% | | |

### 5.3 SAEM Workflow

| Step | Action | Expected | Observed | Pass/Fail |
|------|--------|----------|----------|-----------|
| PQ-WARF-009 | Fit SAEM: `result = nextstat.nlme_saem(...)` with seed=12345 | Completes burn-in and estimation phases | | |
| PQ-WARF-010 | SAEM theta vs FOCE theta | Relative difference < 25% per parameter | | |
| PQ-WARF-011 | SAEM OFV vs FOCE OFV | \|OFV_SAEM - OFV_FOCE\| < 10.0 | | |

### 5.4 Diagnostics Workflow

| Step | Action | Expected | Observed | Pass/Fail |
|------|--------|----------|----------|-----------|
| PQ-WARF-012 | Compute GOF: `gof = nextstat.pk_gof(...)` | CWRES approximately N(0,1) | | |
| PQ-WARF-013 | Run VPC: `vpc = nextstat.pk_vpc(n_sim=500)` | >= 80% observed medians within 90% PI | | |
| PQ-WARF-014 | IPRED vs DV correlation | Pearson r^2 > 0.90 | | |
| PQ-WARF-015 | PRED vs DV correlation | Pearson r^2 > 0.50 | | |

### 5.5 Reproducibility

| Step | Action | Expected | Observed | Pass/Fail |
|------|--------|----------|----------|-----------|
| PQ-WARF-016 | Re-run FOCE (identical input) | Bit-for-bit identical OFV, theta, omega | | |
| PQ-WARF-017 | Re-run SAEM (same seed=12345) | Bit-for-bit identical OFV, theta, omega | | |
| PQ-WARF-018 | Export artifact, reimport, re-export | JSON byte-for-byte identical | | |

## 6. Cross-Validation with Reference Implementations

When NONMEM, Monolix, or nlmixr2 results are available for the same dataset and model,
perform the following cross-validation:

| ID | Cross-Validation Test | Acceptance Criteria | Pass/Fail |
|----|----------------------|---------------------|-----------|
| PQ-XVAL-001 | OFV comparison (FOCE) | \|OFV_NS - OFV_ref\| <= 2.0 | |
| PQ-XVAL-002 | Fixed effects comparison | Relative error < 10% per theta | |
| PQ-XVAL-003 | Individual ETA correlation | Pearson r > 0.90 per ETA | |
| PQ-XVAL-004 | IPRED correlation | Pearson r > 0.99 | |
| PQ-XVAL-005 | CWRES distribution agreement | KS test p-value > 0.05 | |

## 7. Reproducibility Verification

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|---------------------|-----------|
| PQ-REPR-001 | Deterministic FOCE | Same input produces identical output across runs | |
| PQ-REPR-002 | Deterministic SAEM (fixed seed) | Same seed produces identical output across runs | |
| PQ-REPR-003 | Artifact round-trip fidelity | JSON export-import-export produces identical output | |
| PQ-REPR-004 | RunBundle provenance completeness | Version, git rev, timestamp, seed, data hash all present | |

## 8. Performance Baseline

Record timing benchmarks for the qualified configuration. These are informational (not
pass/fail) but establish a baseline for future re-qualification and performance monitoring.

| ID | Operation | Dataset | Hardware | Observed Time | Notes |
|----|-----------|---------|----------|---------------|-------|
| PQ-PERF-001 | FOCE fit | Site dataset | ________________ | _________ s | |
| PQ-PERF-002 | SAEM fit | Site dataset | ________________ | _________ s | |
| PQ-PERF-003 | VPC (n_sim=500) | Site dataset | ________________ | _________ s | |
| PQ-PERF-004 | SCM (all candidates) | Site dataset | ________________ | _________ s | |
| PQ-PERF-005 | Bootstrap (n=1000) | Site dataset | ________________ | _________ s | |
| PQ-PERF-006 | Cox PH fit | Site dataset | ________________ | _________ s | |
| PQ-PERF-007 | GLM fit | Site dataset | ________________ | _________ s | |

## 9. Acceptance Criteria

PQ is considered **PASSED** when:

1. All site-specific test cases record a **PASS** result.
2. At least one complete workflow (e.g., Warfarin FOCE + diagnostics) is fully validated.
3. Cross-validation with at least one reference implementation is completed.
4. Reproducibility tests (PQ-REPR-001 through PQ-REPR-004) all pass.
5. Performance baselines are recorded.
6. All deviations are documented and resolved.

## 10. Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Executed By | ________________________________ | ________________________________ | ____/____/________ |
| Reviewed By (SME) | ________________________________ | ________________________________ | ____/____/________ |
| Reviewed By (QA) | ________________________________ | ________________________________ | ____/____/________ |
| Approved By | ________________________________ | ________________________________ | ____/____/________ |

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-21 | NextStat Team | Initial release |
