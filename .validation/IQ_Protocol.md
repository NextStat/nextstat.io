# Installation Qualification (IQ) Protocol

## Document Control

| Field | Value |
|-------|-------|
| Document ID | NS-VAL-IQ-001 |
| Version | 1.0 |
| Effective Date | ____/____/________ |
| Supersedes | N/A |
| Classification | GxP Validation Deliverable |
| Author | ________________________________ |
| Reviewer | ________________________________ |
| Approver | ________________________________ |

---

## 1. Purpose

This document defines the Installation Qualification (IQ) protocol for verifying that
NextStat is correctly installed and operational in the target computing environment.
Successful completion of IQ is a prerequisite for Operational Qualification (OQ) and
Performance Qualification (PQ).

IQ ensures that:
- The correct version of NextStat is installed
- The installed binary matches the published cryptographic hash (integrity verification)
- All system prerequisites are satisfied
- The software imports and initializes without error

## 2. Scope

This protocol covers Python package installation via pip from PyPI, verification of
binary integrity through SHA-256 checksums, confirmation of system requirements,
and optional source-build verification for organizations that compile from source.

### 2.1 Applicable Regulations

| Regulation | Relevance to IQ |
|-----------|----------------|
| 21 CFR Part 11 Section 11.10(a) | System validation including installation verification |
| EU Annex 11 Section 4.3 | Configuration management and baseline verification |
| GAMP 5 Section 7.5 | Installation qualification requirements for configured software |

## 3. Prerequisites

| Requirement | Specification | Notes |
|-------------|---------------|-------|
| Operating System | Linux x86_64 (glibc >= 2.17), macOS ARM64 or x86_64 (>= 12.0), Windows x86_64 (>= 10) | |
| Python | 3.9, 3.10, 3.11, 3.12, or 3.13 | CPython only; PyPy not supported |
| Memory | >= 4 GB RAM | >= 8 GB recommended for population PK |
| Disk | >= 500 MB free | For package and temporary files |
| Network | PyPI access (pip install) | Not required for offline/air-gapped installation |

## 4. Installation Verification

### 4.1 Package Installation

| Step | ID | Action | Expected Result | Actual Result | Pass/Fail |
|------|-----|--------|-----------------|---------------|-----------|
| 1 | IQ-INST-001 | Execute: `pip install nextstat==X.Y.Z` | Installation completes without errors. Exit code 0. No compilation required (pre-built wheel). | | |
| 2 | IQ-INST-002 | Execute: `python -c "import nextstat; print(nextstat.__version__)"` | Prints the expected version string `X.Y.Z`. No ImportError. | | |
| 3 | IQ-INST-003 | Execute: `python -c "import nextstat; print(nextstat.has_cuda())"` | Prints `True` or `False` (platform-dependent). No error. | | |
| 4 | IQ-INST-004 | Execute: `python -c "import nextstat; print(nextstat.has_metal())"` | Prints `True` or `False` (platform-dependent). No error. | | |

### 4.2 Binary Integrity

| Step | ID | Action | Expected Result | Actual Result | Pass/Fail |
|------|-----|--------|-----------------|---------------|-----------|
| 5 | IQ-CHK-001 | Obtain SHA-256 hash from PyPI JSON API: `curl -s https://pypi.org/pypi/nextstat/X.Y.Z/json \| python -m json.tool \| grep sha256` | Returns SHA-256 hash string for the installed wheel. | | |
| 6 | IQ-CHK-002 | Compute SHA-256 of installed wheel file: `sha256sum $(pip show -f nextstat \| grep Location \| cut -d' ' -f2)/nextstat*.dist-info/RECORD` or verify via `pip hash`. | Hash matches the value obtained in IQ-CHK-001. | | |
| 7 | IQ-CHK-003 | Execute: `python -c "import nextstat._core"` | No ImportError. Native extension loads successfully. | | |

### 4.3 Dependency Verification

| Step | ID | Action | Expected Result | Actual Result | Pass/Fail |
|------|-----|--------|-----------------|---------------|-----------|
| 8 | IQ-DEP-001 | Execute: `pip check nextstat` | Reports no dependency conflicts. | | |
| 9 | IQ-DEP-002 | Execute: `pip show nextstat \| grep Requires` | Lists zero or minimal Python dependencies. NextStat ships as a self-contained native wheel. | | |
| 10 | IQ-DEP-003 | Verify numpy availability (if used for array I/O): `python -c "import numpy; print(numpy.__version__)"` | Prints numpy version >= 1.21 (if installed). | | |

### 4.4 Platform Verification

| Step | ID | Action | Expected Result | Actual Result | Pass/Fail |
|------|-----|--------|-----------------|---------------|-----------|
| 11 | IQ-PLT-001 | Record Python version: `python --version` | Version in range 3.9 -- 3.13 | | |
| 12 | IQ-PLT-002 | Record OS version: `uname -a` (Linux/macOS) or `ver` (Windows) | Supported operating system | | |
| 13 | IQ-PLT-003 | Record CPU architecture: `uname -m` or `python -c "import platform; print(platform.machine())"` | x86_64 or arm64 | | |
| 14 | IQ-PLT-004 | Record available RAM: `python -c "import os; print(f'{os.sysconf(\"SC_PAGE_SIZE\") * os.sysconf(\"SC_PHYS_PAGES\") / (1024**3):.1f} GB')"` | >= 4 GB | | |
| 15 | IQ-PLT-005 | Record available disk space: `df -h .` | >= 500 MB free | | |

### 4.5 API Availability Verification

| Step | ID | Action | Expected Result | Actual Result | Pass/Fail |
|------|-----|--------|-----------------|---------------|-----------|
| 16 | IQ-API-001 | Execute: `python -c "from nextstat import fit, ranking, hypotest, profile_scan; print('HEP OK')"` | Prints `HEP OK`. Core HEP functions importable. | | |
| 17 | IQ-API-002 | Execute: `python -c "from nextstat import cox_ph, kaplan_meier; print('Survival OK')"` | Prints `Survival OK`. Survival analysis functions importable. | | |
| 18 | IQ-API-003 | Execute: `python -c "from nextstat import sample; print('Bayes OK')"` | Prints `Bayes OK`. Bayesian sampling function importable. | | |
| 19 | IQ-API-004 | Execute: `python -c "from nextstat import nlme_foce, nlme_saem; print('Pharma OK')"` | Prints `Pharma OK`. Pharmacometric functions importable. | | |
| 20 | IQ-API-005 | Execute: `python -c "from nextstat import glm, lmm; print('Biostat OK')"` | Prints `Biostat OK`. Biostatistical functions importable. | | |

### 4.6 Source Build Verification (Optional)

For organizations that build NextStat from source (e.g., for regulatory audit of the
compilation process or for air-gapped environments):

| Step | ID | Action | Expected Result | Actual Result | Pass/Fail |
|------|-----|--------|-----------------|---------------|-----------|
| 21 | IQ-SRC-001 | Clone repository and checkout release tag: `git clone <repo> && git checkout vX.Y.Z` | Checkout succeeds. `git log -1` shows expected release tag. | | |
| 22 | IQ-SRC-002 | Verify Cargo.toml version: `grep '^version' Cargo.toml` | Shows `version = "X.Y.Z"`. | | |
| 23 | IQ-SRC-003 | Build from source: `cargo build --release -p ns-inference` | Exit code 0. No compilation errors. | | |
| 24 | IQ-SRC-004 | Run unit test suite: `cargo test --release -p ns-core -p ns-ad -p ns-compute -p ns-translate -p ns-inference` | All tests pass. Zero failures. | | |
| 25 | IQ-SRC-005 | Build Python wheel: `maturin build --release -m bindings/ns-py/Cargo.toml` | Wheel file produced in `target/wheels/`. | | |
| 26 | IQ-SRC-006 | Install and verify built wheel: `pip install target/wheels/nextstat-*.whl && python -c "import nextstat; print(nextstat.__version__)"` | Import succeeds. Version matches. | | |

## 5. Acceptance Criteria

IQ is considered **PASSED** when:

1. All mandatory steps (IQ-INST-001 through IQ-API-005) record a **PASS** result.
2. Any **FAIL** results are documented as deviations per the deviation handling procedure
   in the Validation Summary Report (VSR_Template.md).
3. Optional source build steps (IQ-SRC-*) are evaluated only when applicable.

IQ is considered **FAILED** if any mandatory step cannot be resolved. The failure must be
documented, root cause identified, and the step re-executed after corrective action.

## 6. Environmental Record

Complete this section to document the qualified environment:

| Item | Value |
|------|-------|
| NextStat Version | ________________________________ |
| Python Version | ________________________________ |
| Operating System | ________________________________ |
| CPU Architecture | ________________________________ |
| CPU Model | ________________________________ |
| RAM (GB) | ________________________________ |
| Disk Free (GB) | ________________________________ |
| CUDA Available | Yes / No |
| Metal Available | Yes / No |
| Date of Qualification | ________________________________ |

## 7. Sign-Off

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
