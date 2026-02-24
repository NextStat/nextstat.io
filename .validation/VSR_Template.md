# Validation Summary Report (VSR)

## Document Control

| Field | Value |
|-------|-------|
| Document ID | NS-VAL-VSR-001 |
| Version | 1.0 |
| Effective Date | ____/____/________ |
| Supersedes | N/A |
| Classification | GxP Validation Deliverable |
| Author | ________________________________ |
| Reviewer | ________________________________ |
| Approver | ________________________________ |

---

## 1. Purpose

This Validation Summary Report (VSR) documents the completed validation of the NextStat
statistical computing platform for use in the organization's GxP-regulated environment.
It summarizes the results of Installation Qualification (IQ), Operational Qualification (OQ),
and Performance Qualification (PQ), and provides a final determination of the system's
validated status.

This report satisfies the documentation requirements of:
- 21 CFR Part 11 Section 11.10(a) -- documented validation of computerised systems
- EU Annex 11 Section 4.1 -- validation documentation maintained throughout lifecycle
- GAMP 5 Section 7.9 -- validation summary report as a lifecycle deliverable
- ICH E6(R3) Section 4.9.3 -- documentation of validation activities

## 2. System Description

| Item | Value |
|------|-------|
| System Name | NextStat |
| Version Validated | ________________________________ |
| Vendor | NextStat, Inc. |
| GAMP Category | Category 4 (Configured) / Category 5 (Custom) |
| Intended Use | ________________________________ |
| Deployment Environment | ________________________________ |
| Date of Validation | ________________________________ |

### 2.1 System Components

| Component | Version | Description |
|-----------|---------|-------------|
| NextStat Python package | ________ | Statistical computing engine (native Rust, Python API) |
| Python runtime | ________ | CPython interpreter |
| Operating system | ________ | Host operating system |
| Hardware | ________ | CPU, RAM, GPU (if applicable) |

### 2.2 Intended Use Statement

Describe the specific regulated workflows for which NextStat is being validated:

________________________________________________________________________

________________________________________________________________________

________________________________________________________________________

________________________________________________________________________

## 3. Validation Activities Summary

### 3.1 Protocols Executed

| Protocol | Document ID | Version | Execution Date | Executed By |
|----------|------------|---------|----------------|-------------|
| Installation Qualification | NS-VAL-IQ-001 | ________ | ____/____/________ | ________________________________ |
| Operational Qualification | NS-VAL-OQ-001 | ________ | ____/____/________ | ________________________________ |
| Performance Qualification | NS-VAL-PQ-001 | ________ | ____/____/________ | ________________________________ |
| Risk Assessment | NS-VAL-RA-001 | ________ | ____/____/________ | ________________________________ |

### 3.2 Supporting Documents Reviewed

| Document | Version | Date Reviewed | Reviewer |
|----------|---------|---------------|----------|
| Traceability Matrix (NS-VAL-TM-001) | ________ | ____/____/________ | ________________ |
| Risk Assessment (NS-VAL-RA-001) | ________ | ____/____/________ | ________________ |
| CHANGELOG.md | ________ | ____/____/________ | ________________ |
| docs/validation/iq-oq-pq-protocol.md | ________ | ____/____/________ | ________________ |

## 4. IQ Results Summary

| Category | Total Tests | Passed | Failed | N/A |
|----------|------------|--------|--------|-----|
| Package Installation (IQ-INST) | | | | |
| Binary Integrity (IQ-CHK) | | | | |
| Dependency Verification (IQ-DEP) | | | | |
| Platform Verification (IQ-PLT) | | | | |
| API Availability (IQ-API) | | | | |
| Source Build (IQ-SRC) -- optional | | | | |
| **Total** | | | | |

### 4.1 IQ Qualified Environment

| Item | Value |
|------|-------|
| NextStat Version | ________________________________ |
| Python Version | ________________________________ |
| Operating System | ________________________________ |
| CPU Architecture | ________________________________ |
| CPU Model | ________________________________ |
| RAM | ________________________________ |
| CUDA Available | Yes / No |
| Metal Available | Yes / No |

### 4.2 IQ Conclusion

[ ] PASS -- All mandatory IQ tests passed. The system is correctly installed.

[ ] PASS WITH DEVIATIONS -- IQ passed with documented deviations (see Section 7).

[ ] FAIL -- IQ did not pass. Corrective actions required before proceeding.

## 5. OQ Results Summary

| Domain | Total Tests | Passed | Failed | Deviations |
|--------|------------|--------|--------|------------|
| PK Analytical Models | 8 | | | |
| FOCE/FOCEI | 6 | | | |
| SAEM | 5 | | | |
| GOF / VPC | 5 | | | |
| SCM | 4 | | | |
| Error Models | 4 | | | |
| NONMEM I/O | 4 | | | |
| Survival Analysis | 10 | | | |
| GLM | 8 | | | |
| LMM | 5 | | | |
| Bootstrap CI | 4 | | | |
| Emax/Hill | 3 | | | |
| ODE Solver | 3 | | | |
| HEP/HistFactory | 10 | | | |
| Bayesian Sampling | 8 | | | |
| **Total** | **93** | | | |

### 5.1 OQ Automated Report Reference

| Item | Value |
|------|-------|
| Report file | ________________________________ |
| Report SHA-256 | ________________________________ |
| Execution timestamp | ________________________________ |
| Total execution time | ________________________________ |

### 5.2 OQ Conclusion

[ ] PASS -- All 93 OQ tests passed. The system operates correctly within specifications.

[ ] PASS WITH DEVIATIONS -- OQ passed with documented deviations (see Section 7).

[ ] FAIL -- OQ did not pass. Corrective actions required.

## 6. PQ Results Summary

| Category | Total Tests | Passed | Failed | Deviations |
|----------|------------|--------|--------|------------|
| Site-Specific Workflow Tests | | | | |
| Reference Dataset Tests | | | | |
| Cross-Validation Tests | | | | |
| Reproducibility Tests | 4 | | | |
| Performance Baselines | -- (informational) | | | |
| **Total** | | | | |

### 6.1 Reference Datasets Used

| Dataset | Source | Subjects | Observations | SHA-256 |
|---------|--------|----------|--------------|---------|
| | | | | |
| | | | | |

### 6.2 Cross-Validation Summary

| Comparator Tool | Version | Datasets Compared | Agreement Level |
|-----------------|---------|-------------------|-----------------|
| | | | |
| | | | |

### 6.3 PQ Conclusion

[ ] PASS -- All PQ tests passed. The system is fit for intended use.

[ ] PASS WITH DEVIATIONS -- PQ passed with documented deviations (see Section 7).

[ ] FAIL -- PQ did not pass. Corrective actions required.

## 7. Deviations

### 7.1 Deviation Log

| Deviation ID | Test ID | Classification | Description | Root Cause | Corrective Action | Resolution Date | Re-test Result |
|-------------|---------|----------------|-------------|------------|-------------------|-----------------|----------------|
| DEV-___-001 | | Critical / Major / Minor | | | | ____/____/________ | Pass / Fail |
| DEV-___-002 | | Critical / Major / Minor | | | | ____/____/________ | Pass / Fail |
| DEV-___-003 | | Critical / Major / Minor | | | | ____/____/________ | Pass / Fail |

### 7.2 Deviation Impact Assessment

For each deviation, document the impact on the system's validated status:

________________________________________________________________________

________________________________________________________________________

### 7.3 Deviation Summary

| Classification | Count | All Resolved? |
|----------------|-------|---------------|
| Critical | | Yes / No |
| Major | | Yes / No |
| Minor | | Yes / No |

## 8. Risk Assessment Summary

### 8.1 Residual Risk

| Risk Level | Count | Acceptable? |
|------------|-------|------------|
| Low (RPN 1-12) | 20 | Yes |
| Medium (RPN 13-36) | 10 | Yes, with mitigations |
| High (RPN 37-75) | 0 | N/A |
| Critical (RPN 76-125) | 0 | N/A |

### 8.2 Risk Acceptance Statement

Based on the FMEA documented in NS-VAL-RA-001, all identified risks are at or below
the Medium level after implementation of mitigating controls. The residual risk is
acceptable for the intended use described in Section 2.2.

[ ] Risk accepted by Quality Assurance: ________________ Date: ____/____/________

## 9. Traceability Confirmation

### 9.1 Completeness Check

| Check | Result |
|-------|--------|
| All requirements have at least one test case | [ ] Confirmed |
| All test cases trace to at least one requirement | [ ] Confirmed |
| All high-risk requirements have OQ AND PQ coverage | [ ] Confirmed |
| No orphan tests (tests without requirements) | [ ] Confirmed |

### 9.2 Traceability Matrix Reference

Refer to NS-VAL-TM-001 (Traceability_Matrix.md) for the complete bidirectional mapping
of 75 requirements to 93 OQ tests and 25 PQ tests.

## 10. Change Control

### 10.1 Re-Qualification Requirements

This validation is specific to the system version and environment documented in
Sections 2 and 4.1. Re-qualification is required upon:

| Trigger | Required Scope |
|---------|---------------|
| NextStat version upgrade (major or minor) | Full IQ + OQ + PQ |
| NextStat patch version (bug fix only) | IQ + targeted OQ |
| Operating system change | IQ + targeted PQ |
| Python version change | IQ + regression OQ |
| Hardware change (CPU architecture) | IQ + PQ performance baselines |
| Regulatory audit finding | Scope per finding |

### 10.2 Periodic Review

This validation shall be reviewed at least annually to confirm continued applicability.
The periodic review shall verify:

1. No unvalidated NextStat versions have been deployed.
2. No deviations have been identified since the last review.
3. The intended use has not changed.
4. Regulatory requirements have not changed.

| Review Date | Reviewer | Outcome | Next Review Due |
|-------------|----------|---------|-----------------|
| ____/____/________ | ________________ | Valid / Re-qualification needed | ____/____/________ |

## 11. Overall Validation Conclusion

Based on the results documented in this report:

[ ] **VALIDATED** -- NextStat version ________ is validated for use in the organization's
GxP-regulated environment for the intended use described in Section 2.2. All IQ, OQ, and
PQ tests have passed (or passed with acceptable documented deviations). Residual risks
are at an acceptable level.

[ ] **CONDITIONALLY VALIDATED** -- NextStat is validated with the restrictions documented
in Section 7. The following functions/workflows are excluded from the validated scope:
________________________________________________________________________

[ ] **NOT VALIDATED** -- Validation was not completed. The following critical deviations
remain unresolved: ________________________________________________________________________

## 12. Approval Signatures

This Validation Summary Report has been reviewed and approved by the following personnel.
By signing below, each individual confirms that the validation activities described herein
were conducted in accordance with the applicable protocols and that the conclusions
accurately reflect the results.

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Validation Lead | ________________________________ | ________________________________ | ____/____/________ |
| Subject Matter Expert (Pharmacometrician / Biostatistician) | ________________________________ | ________________________________ | ____/____/________ |
| IT / System Administrator | ________________________________ | ________________________________ | ____/____/________ |
| Quality Assurance | ________________________________ | ________________________________ | ____/____/________ |
| System Owner / Sponsor | ________________________________ | ________________________________ | ____/____/________ |

---

## Appendix A: Attached Evidence

| Attachment | Description | File Reference |
|------------|-------------|----------------|
| A-1 | IQ Protocol (executed) | NS-VAL-IQ-001 |
| A-2 | OQ Protocol (executed) | NS-VAL-OQ-001 |
| A-3 | OQ Automated Report (JSON) | ________________________________ |
| A-4 | PQ Protocol (executed) | NS-VAL-PQ-001 |
| A-5 | Traceability Matrix | NS-VAL-TM-001 |
| A-6 | Risk Assessment (FMEA) | NS-VAL-RA-001 |
| A-7 | CHANGELOG.md (validated version) | ________________________________ |
| A-8 | Reference dataset checksums | ________________________________ |

## Appendix B: Abbreviations

| Abbreviation | Definition |
|-------------|-----------|
| BCa | Bias-Corrected and Accelerated (bootstrap CI method) |
| BLUP | Best Linear Unbiased Predictor |
| CI | Confidence Interval |
| CLs | Modified frequentist confidence level (HEP) |
| CWRES | Conditional Weighted Residuals |
| FOCE | First-Order Conditional Estimation |
| FOCEI | FOCE with Interaction |
| FMEA | Failure Mode and Effects Analysis |
| GAMP | Good Automated Manufacturing Practice |
| GLM | Generalized Linear Model |
| GOF | Goodness of Fit |
| GxP | Good Practice (GCP, GLP, GMP collectively) |
| HEP | High Energy Physics |
| IQ | Installation Qualification |
| LLOQ | Lower Limit of Quantification |
| LMM | Linear Mixed Model |
| MLE | Maximum Likelihood Estimation |
| NLME | Nonlinear Mixed-Effects |
| NUTS | No-U-Turn Sampler |
| OFV | Objective Function Value |
| OQ | Operational Qualification |
| PK | Pharmacokinetics |
| PQ | Performance Qualification |
| REML | Restricted Maximum Likelihood |
| RPN | Risk Priority Number |
| SAEM | Stochastic Approximation Expectation-Maximization |
| SCM | Stepwise Covariate Modeling |
| SE | Standard Error |
| VPC | Visual Predictive Check |
| VSR | Validation Summary Report |

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-21 | NextStat Team | Initial release |
