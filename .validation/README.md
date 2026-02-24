# NextStat Validation Pack v1.0

## Purpose

This validation pack provides Installation Qualification (IQ), Operational Qualification (OQ),
and Performance Qualification (PQ) documentation for NextStat in GxP-regulated environments.
It is intended for use by Quality Assurance, Validation, and Biometrics teams deploying NextStat
as computational software supporting regulatory submissions, clinical trial analyses, and
pharmacometric modeling in pharmaceutical, biotechnology, and medical device organizations.

## Regulatory Framework

This validation pack has been prepared in accordance with the following regulatory and industry
guidance documents:

- **21 CFR Part 11** (FDA -- Electronic Records; Electronic Signatures)
- **EU Annex 11** (EMA -- Computerised Systems, EudraLex Volume 4)
- **ICH E6(R3)** (Good Clinical Practice -- Computerised Systems)
- **ICH E9(R1)** (Statistical Principles for Clinical Trials -- Estimands and Sensitivity Analysis)
- **ISPE GAMP 5 Second Edition (2022)** (Good Automated Manufacturing Practice)
- **FDA Guidance for Industry: Computer Systems Used in Clinical Investigations (2007)**
- **EMA Guideline on Computerised Systems and Electronic Data in Clinical Trials (2023)**

## GAMP 5 Classification

NextStat is classified as **GAMP Category 4** (Configured Software Product) for general
statistical and biostatistical workflows:

- Pre-built, validated statistical algorithms with documented behavior
- User configures models, data inputs, and analysis parameters
- No custom code execution is required for standard workflows
- Deterministic f64 computation: same input produces same output on any supported platform

**Note:** For pharmacometric workflows involving custom NLME models (FOCE, SAEM), NextStat may
be classified as **GAMP Category 5** (Custom Software) depending on the extent of user-defined
model specification. Refer to `docs/validation/iq-oq-pq-protocol.md` for the pharma-specific
Category 5 validation protocol with 85+ test cases across 11 OQ subsections.

## Document Inventory

| Document | File | Description |
|----------|------|-------------|
| Installation Qualification | [`IQ_Protocol.md`](IQ_Protocol.md) | Verify correct installation and platform requirements |
| Operational Qualification | [`OQ_Protocol.md`](OQ_Protocol.md) | Verify computational accuracy across all functional domains |
| Performance Qualification | [`PQ_Protocol.md`](PQ_Protocol.md) | Verify site-specific workflows with user data |
| Traceability Matrix | [`Traceability_Matrix.md`](Traceability_Matrix.md) | Requirement-to-test-case bidirectional mapping |
| Risk Assessment | [`Risk_Assessment.md`](Risk_Assessment.md) | FMEA for statistical computation functions |
| Validation Summary Report | [`VSR_Template.md`](VSR_Template.md) | Template for completed validation documentation |

## Companion Documents

| Document | Location | Description |
|----------|----------|-------------|
| Pharma IQ/OQ/PQ Protocol | `docs/validation/iq-oq-pq-protocol.md` | Category 5 protocol with 85 pharma-specific test cases |
| Validation Report Schema | `docs/schemas/validation/validation_report_v1.schema.json` | JSON schema for automated validation reports |
| CHANGELOG | `CHANGELOG.md` | Version-controlled change history for audit trail |

## How to Use This Validation Pack

### Initial Qualification

1. **IQ**: Complete Installation Qualification before first use. Verify package installation,
   binary integrity, platform compatibility, and dependency satisfaction.

2. **OQ**: Execute the Operational Qualification test suite. This can be run automatically:
   ```bash
   python -m nextstat.validate.oq --output oq_results.json
   ```
   All test cases compare NextStat output against published reference values or validated
   reference implementations with documented numerical tolerances.

3. **PQ**: Complete Performance Qualification using site-specific data and workflows.
   Use the PQ template to document results with your organization's datasets.

4. **VSR**: Document all results in the Validation Summary Report template. Obtain
   required signatures per your organization's Quality Management System (QMS).

### Re-Qualification

Re-validate when any of the following occur:

| Trigger | Required Scope |
|---------|---------------|
| NextStat major or minor version upgrade | Full IQ + OQ + PQ |
| NextStat patch version (bug fix only) | IQ + targeted OQ for affected functions |
| Operating system or hardware change | IQ + targeted PQ |
| Python version change | IQ + regression OQ |
| Regulatory audit finding | Scope determined by finding |

### Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-02-21 | NextStat Team | Initial release |

---

*This validation pack is provided as a template. Organizations must customize it to align with
their internal Quality Management System (QMS), Standard Operating Procedures (SOPs), and
regulatory requirements. NextStat, Inc. does not provide validation services; the responsibility
for GxP validation rests with the end-user organization.*
