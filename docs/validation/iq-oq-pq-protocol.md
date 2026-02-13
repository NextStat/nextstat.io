# NextStat IQ/OQ/PQ Validation Protocol

**Document ID:** NS-VAL-001  
**Version:** 1.0.0  
**Status:** Template — requires site-specific customization  
**Applicable Regulations:** 21 CFR Part 11, ICH E6(R2), EU Annex 11  

---

## 1. Purpose

This document defines the Installation Qualification (IQ), Operational Qualification (OQ), and Performance Qualification (PQ) protocols for NextStat statistical and pharmacometric software. It provides a structured framework for validating that the software is correctly installed, operates according to specifications, and produces reliable results in a regulated GxP environment.

## 2. Scope

This protocol covers:

- **NextStat core** (`ns-core`, `ns-inference`, `ns-compute`) — statistical engine
- **Pharmacometrics modules** — FOCE/FOCEI, SAEM, PK models, PD models, ODE solvers, SCM, VPC/GOF
- **Python bindings** (`ns-py`) — user-facing API
- **R bindings** (`ns-r`) — NLME/PK/PD wrappers (when available)
- **CLI** (`ns-cli`) — command-line interface

### Out of Scope

- Operating system qualification (covered by IT infrastructure validation)
- Network and database infrastructure
- Third-party tools used alongside NextStat (R, Python, NONMEM)

## 3. Definitions

| Term | Definition |
|------|-----------|
| **IQ** | Documented verification that the software is installed correctly and all components are present and match expected checksums. |
| **OQ** | Documented verification that the software operates correctly within specified operating ranges using predefined test cases. |
| **PQ** | Documented verification that the software consistently produces correct results under conditions approximating real-world use. |
| **CSV** | Computer System Validation — systematic approach to ensuring computerized systems do what they are designed to do. |
| **GAMP 5** | Good Automated Manufacturing Practice — risk-based approach to CSV. NextStat is Category 4 (configured product). |
| **OFV** | Objective Function Value (−2·log L) — primary metric for NLME model fitting. |

## 4. Roles and Responsibilities

| Role | Responsibility |
|------|---------------|
| **Validation Lead** | Oversees protocol execution, reviews results, approves deviations |
| **QA Reviewer** | Independent review of validation deliverables |
| **System Administrator** | Performs IQ steps, manages installation |
| **Pharmacometrician** | Executes OQ/PQ test cases, interprets results |
| **IT Security** | Verifies access controls and audit trail configuration |

---

## 5. Installation Qualification (IQ)

### 5.1 Prerequisites

| ID | Check | Expected Result | Pass/Fail |
|----|-------|----------------|-----------|
| IQ-PRE-01 | Rust toolchain version | ≥ 1.82.0 (per `rust-toolchain.toml`) | ☐ |
| IQ-PRE-02 | Python version (if using bindings) | ≥ 3.9 | ☐ |
| IQ-PRE-03 | R version (if using bindings) | ≥ 4.1 | ☐ |
| IQ-PRE-04 | Operating system | Linux x86_64, macOS arm64/x86_64, or Windows x86_64 | ☐ |
| IQ-PRE-05 | Available disk space | ≥ 500 MB | ☐ |

### 5.2 Installation Verification

| ID | Step | Expected Result | Pass/Fail |
|----|------|----------------|-----------|
| IQ-INST-01 | Build from source: `cargo build --release` | Exit code 0, no errors | ☐ |
| IQ-INST-02 | Verify binary exists | `target/release/ns-cli --version` returns version string | ☐ |
| IQ-INST-03 | Python wheel install: `pip install ns-py` | Package installs without errors | ☐ |
| IQ-INST-04 | Python import: `import nextstat` | No ImportError | ☐ |
| IQ-INST-05 | Run unit test suite: `cargo test --release` | All tests pass | ☐ |
| IQ-INST-06 | Verify SHA-256 checksum of release binary | Matches published checksum | ☐ |

### 5.3 Configuration Verification

| ID | Check | Expected Result | Pass/Fail |
|----|-------|----------------|-----------|
| IQ-CONF-01 | Verify `Cargo.toml` workspace version | Matches expected release version | ☐ |
| IQ-CONF-02 | Verify feature flags | Default features match deployment spec | ☐ |
| IQ-CONF-03 | Verify BLAS/LAPACK backend | Correct backend for target platform (Accelerate/OpenBLAS/MKL) | ☐ |

---

## 6. Operational Qualification (OQ)

### 6.1 Core Statistical Functions

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| OQ-CORE-01 | MLE for Gaussian distribution | θ̂ within 1e-6 of analytical MLE | ☐ |
| OQ-CORE-02 | MLE for Poisson distribution | θ̂ within 1e-6 of analytical MLE | ☐ |
| OQ-CORE-03 | Profile likelihood CI | Covers true parameter with 95% probability | ☐ |
| OQ-CORE-04 | NUTS sampler | ESS > 100 for unit normal target | ☐ |
| OQ-CORE-05 | GLM (linear regression) | β̂ matches `lm()` in R to 1e-8 | ☐ |
| OQ-CORE-06 | GLM (logistic regression) | β̂ matches `glm()` in R to 1e-6 | ☐ |

### 6.2 Pharmacometrics — PK Models

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| OQ-PK-01 | 1-compartment oral concentration | Matches analytical solution to 1e-10 | ☐ |
| OQ-PK-02 | 2-compartment IV concentration | Matches eigenvalue solution to 1e-8 | ☐ |
| OQ-PK-03 | Dosing regimen (multiple oral doses) | Superposition matches manual calculation | ☐ |
| OQ-PK-04 | IV infusion (during + post) | Matches closed-form solution to 1e-8 | ☐ |
| OQ-PK-05 | NONMEM dataset parsing | Correctly reads Warfarin NONMEM CSV | ☐ |

### 6.3 Pharmacometrics — FOCE/FOCEI

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| OQ-FOCE-01 | FOCEI fit — Warfarin synthetic (32 subj) | OFV finite, θ̂ within 3×ω of true | ☐ |
| OQ-FOCE-02 | FOCE correlated Ω — CL-V correlation | Ω positive-definite, correlation recovered | ☐ |
| OQ-FOCE-03 | Convergence check | `converged = true` within 100 iterations | ☐ |

### 6.4 Pharmacometrics — SAEM

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| OQ-SAEM-01 | SAEM fit — Warfarin synthetic (32 subj) | OFV finite, θ̂ within 3×ω of true | ☐ |
| OQ-SAEM-02 | SAEM correlated Ω | Ω positive-definite, all ωSD > 0 | ☐ |
| OQ-SAEM-03 | SAEM vs FOCE parity | θ̂ agree within 100% relative | ☐ |
| OQ-SAEM-04 | MCMC acceptance rate | Between 15% and 60% (well-tuned) | ☐ |
| OQ-SAEM-05 | OFV trace stability | Late OFV within 10% of early OFV | ☐ |

### 6.5 Pharmacometrics — PD Models

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| OQ-PD-01 | Emax at C=0 | E = E0 exactly | ☐ |
| OQ-PD-02 | Emax at C=EC50 | E = E0 + Emax/2 | ☐ |
| OQ-PD-03 | Sigmoid Emax γ=1 equals Emax | Predictions match to 1e-10 | ☐ |
| OQ-PD-04 | Sigmoid Emax at EC50 | E = E0 + Emax/2 for all γ | ☐ |
| OQ-PD-05 | IDR Type I — inhibit production | Response decreases from baseline | ☐ |
| OQ-PD-06 | IDR Type II — inhibit loss | Response increases from baseline | ☐ |
| OQ-PD-07 | IDR Type III — stimulate production | Response increases from baseline | ☐ |
| OQ-PD-08 | IDR Type IV — stimulate loss | Response decreases from baseline | ☐ |
| OQ-PD-09 | IDR return to baseline | R → R0 after drug washout | ☐ |

### 6.6 ODE Solvers

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| OQ-ODE-01 | RK45 exponential decay | Matches analytical to 1e-6 | ☐ |
| OQ-ODE-02 | ESDIRK exponential decay | Matches analytical to 1e-4 | ☐ |
| OQ-ODE-03 | RK45 vs ESDIRK agreement | Rel. diff < 1e-6 on non-stiff problem | ☐ |
| OQ-ODE-04 | RK45 transit chain (5 cpts) | Central cpt has drug at t=24h | ☐ |
| OQ-ODE-05 | ESDIRK stiff transit (ktr=50) | Solves without exceeding max_steps | ☐ |
| OQ-ODE-06 | Michaelis–Menten elimination | Monotonically decreasing, C > 0 | ☐ |

### 6.7 Diagnostics — SCM, VPC, GOF

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| OQ-DIAG-01 | SCM selects true covariate | Weight effect on CL selected in forward step | ☐ |
| OQ-DIAG-02 | SCM rejects false covariate | Unrelated covariate not selected | ☐ |
| OQ-DIAG-03 | VPC — observed median within PI | Median within 90% PI for ≥80% of bins | ☐ |
| OQ-DIAG-04 | GOF — CWRES approximately N(0,1) | Mean < 0.5, SD between 0.5 and 2.0 | ☐ |

---

## 7. Performance Qualification (PQ)

### 7.1 Reference Dataset Validation

PQ tests use published or well-characterized datasets to validate against known results.

| ID | Dataset | Reference | Acceptance Criteria | Pass/Fail |
|----|---------|-----------|-------------------|-----------|
| PQ-01 | Warfarin PK (O'Reilly, 1968) | NONMEM User Guide | OFV within 5 units of reference | ☐ |
| PQ-02 | Theophylline PK (Boeckmann, 1994) | Monolix tutorials | θ̂ within 20% of published values | ☐ |
| PQ-03 | Phenobarbital PK (neonatal) | Grasela & Donn, 1985 | Parameter estimates in plausible range | ☐ |

### 7.2 Reproducibility

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| PQ-REPRO-01 | Same seed → identical results | Bit-for-bit reproducibility across runs | ☐ |
| PQ-REPRO-02 | RunBundle provenance | Version, git rev, seeds captured | ☐ |
| PQ-REPRO-03 | Artifact round-trip | JSON export → import → re-export identical | ☐ |

### 7.3 Performance Benchmarks

| ID | Test | Acceptance Criteria | Pass/Fail |
|----|------|-------------------|-----------|
| PQ-PERF-01 | FOCE fit wall time (32 subj) | < 5 seconds on reference hardware | ☐ |
| PQ-PERF-02 | SAEM fit wall time (32 subj) | < 10 seconds on reference hardware | ☐ |
| PQ-PERF-03 | ODE solve (10-cpt transit, 24h) | < 100 ms on reference hardware | ☐ |

---

## 8. Deviation Handling

Any test case that fails must be documented as a deviation:

1. **Record** the deviation (test ID, observed result, expected result)
2. **Assess** impact on intended use (critical / major / minor)
3. **Investigate** root cause
4. **Resolve** via software fix or documented limitation
5. **Re-execute** the failed test case after resolution
6. **Approve** via QA review

---

## 9. Validation Summary Report

Upon completion of all IQ/OQ/PQ test cases, a Validation Summary Report (NS-VAL-002) shall be prepared containing:

- Overall pass/fail status
- List of all deviations and their resolutions
- Confirmation of traceability (requirements → test cases → results)
- Approval signatures (Validation Lead, QA Reviewer, System Owner)

---

## 10. Periodic Revalidation

Revalidation is required when:

- A new version of NextStat is deployed
- The operating environment changes (OS upgrade, hardware change)
- A regulatory audit identifies findings
- A deviation is discovered in production use

Minimum revalidation frequency: **annually** or upon major version upgrade.

---

## Appendix A: Test Execution Commands

```bash
# IQ: Full test suite
cargo test --release 2>&1 | tee iq_test_results.txt

# OQ: Pharma-specific tests
cargo test -p ns-inference --test pharma_benchmark -- --nocapture 2>&1 | tee oq_pharma.txt
cargo test -p ns-inference --test phase3_benchmark -- --nocapture 2>&1 | tee oq_phase3.txt

# OQ: Unit tests for specific modules
cargo test -p ns-inference --lib -- foce::tests saem::tests pd::tests ode_adaptive::tests -- --nocapture 2>&1 | tee oq_unit.txt

# PQ: Reproducibility check
cargo test -p ns-inference --test pharma_benchmark -- --nocapture 2>&1 > pq_run1.txt
cargo test -p ns-inference --test pharma_benchmark -- --nocapture 2>&1 > pq_run2.txt
diff pq_run1.txt pq_run2.txt  # Should show no differences
```

## Appendix B: Traceability Matrix

| Requirement | OQ Test(s) | PQ Test(s) |
|------------|-----------|-----------|
| FOCE estimation | OQ-FOCE-01..03 | PQ-01..03 |
| SAEM estimation | OQ-SAEM-01..05 | PQ-01..03 |
| PD models | OQ-PD-01..09 | — |
| ODE solvers | OQ-ODE-01..06 | PQ-PERF-03 |
| Diagnostics | OQ-DIAG-01..04 | — |
| Reproducibility | — | PQ-REPRO-01..03 |
| Performance | — | PQ-PERF-01..03 |

---

**Document Control:**

| Version | Date | Author | Change Description |
|---------|------|--------|-------------------|
| 1.0.0 | 2026-02-12 | NextStat Team | Initial template |
