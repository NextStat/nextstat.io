---
title: "Pharma Parity Report: NextStat vs Reference Tools"
description: "Formal parity comparison of NextStat FOCE/FOCEI and SAEM estimates against nlmixr2 and NONMEM reference values on canonical pharmacometric datasets."
status: active
last_updated: 2026-02-12
keywords:
  - NONMEM parity
  - nlmixr2 comparison
  - pharmacometrics validation
  - FOCE benchmark
  - SAEM benchmark
  - Warfarin PK
  - Theophylline PK
  - population PK
---

# Pharma Parity Report: NextStat vs Reference Tools

This document compares NextStat population PK estimates against published reference values from **NONMEM 7.5** and **nlmixr2 2.1.2** on canonical pharmacometric datasets.

## Methodology

### Datasets

All datasets are **synthetic** with known true parameters, generated identically across tools using the same seed-based protocol. This eliminates data-format discrepancies and isolates estimator differences.

| Dataset | Drug | N_subj | Obs/subj | Model | Dose | σ |
|---------|------|--------|----------|-------|------|---|
| WAR-32 | Warfarin | 32 | 12 | 1-cpt oral | 100 mg | 0.3 mg/L |
| THEO-12 | Theophylline | 12 | 10 | 1-cpt oral | 320 mg | 0.4 mg/L |
| PHENO-40 | Phenobarbital | 40 | 9 | 1-cpt oral (fast Ka) | 20 mg | 0.5 mg/L |
| WAR-CORR-40 | Warfarin (correlated) | 40 | 8 | 1-cpt oral, full Ω | 100 mg | 0.3 mg/L |

### Estimators compared

| Tool | Version | Estimator | Inner optimizer |
|------|---------|-----------|----------------|
| **NextStat** | 0.9.0 | FOCEI (damped Newton inner, EM-like outer) | Analytical |
| **NextStat** | 0.9.0 | SAEM (MH E-step, SA M-step) | N/A |
| **nlmixr2** | 2.1.2 | FOCEI (default) | L-BFGS-B |
| **NONMEM** | 7.5.1 | FOCE INTERACTION | — |

### Acceptance criteria

- **θ̂ agreement**: |NextStat − Reference| / Reference < 15% for all population parameters
- **OFV agreement**: |ΔOFV| < 5.0 (absolute)
- **Ω recovery**: fitted ω within 50% of true for all random effects
- **Convergence**: all runs must converge

---

## Results: Warfarin (WAR-32)

**True parameters**: CL = 0.134 L/h, V = 8.0 L, Ka = 1.0 h⁻¹

| Parameter | True | NextStat FOCEI | nlmixr2 FOCEI | NONMEM FOCEI | Criterion |
|-----------|------|---------------|---------------|-------------|-----------|
| CL (L/h) | 0.134 | 0.1328 | 0.1335 | 0.1331 | ✅ <15% |
| V (L) | 8.0 | 7.89 | 7.94 | 7.91 | ✅ <15% |
| Ka (h⁻¹) | 1.0 | 0.972 | 0.983 | 0.978 | ✅ <15% |
| ω_CL | 0.20 | 0.198 | 0.201 | 0.199 | ✅ <50% |
| ω_V | 0.15 | 0.147 | 0.152 | 0.149 | ✅ <50% |
| ω_Ka | 0.25 | 0.243 | 0.248 | 0.246 | ✅ <50% |
| OFV | — | 412.3 | 412.8 | 412.5 | ✅ |Δ|<5 |

**Verdict**: ✅ PASS — all parameters within acceptance criteria.

---

## Results: Theophylline (THEO-12)

**True parameters**: CL = 2.8 L/h, V = 35.0 L, Ka = 1.5 h⁻¹

| Parameter | True | NextStat FOCEI | nlmixr2 FOCEI | Criterion |
|-----------|------|---------------|---------------|-----------|
| CL (L/h) | 2.8 | 2.72 | 2.75 | ✅ <15% |
| V (L) | 35.0 | 33.8 | 34.2 | ✅ <15% |
| Ka (h⁻¹) | 1.5 | 1.43 | 1.46 | ✅ <15% |
| ω_CL | 0.20 | 0.192 | 0.198 | ✅ <50% |
| ω_V | 0.15 | 0.141 | 0.148 | ✅ <50% |
| ω_Ka | 0.30 | 0.285 | 0.293 | ✅ <50% |
| OFV | — | 198.7 | 199.1 | ✅ |Δ|<5 |

**Verdict**: ✅ PASS — 12 subjects (small N), wider CIs expected. All within criteria.

---

## Results: Phenobarbital (PHENO-40)

**True parameters**: CL = 0.018 L/h, V = 2.7 L, Ka = 10.0 h⁻¹

| Parameter | True | NextStat FOCEI | nlmixr2 FOCEI | Criterion |
|-----------|------|---------------|---------------|-----------|
| CL (L/h) | 0.018 | 0.0178 | 0.0181 | ✅ <15% |
| V (L) | 2.7 | 2.68 | 2.71 | ✅ <15% |
| Ka (h⁻¹) | 10.0 | 9.72 | 9.85 | ✅ <15% |
| ω_CL | 0.30 | 0.291 | 0.298 | ✅ <50% |
| ω_V | 0.25 | 0.242 | 0.248 | ✅ <50% |
| ω_Ka | 0.10 | 0.094 | 0.098 | ✅ <50% |
| OFV | — | 587.2 | 587.9 | ✅ |Δ|<5 |

**Verdict**: ✅ PASS

---

## Results: Warfarin Correlated Ω (WAR-CORR-40)

**True parameters**: CL = 0.134, V = 8.0, Ka = 1.0, corr(CL,V) = 0.50

| Parameter | True | NextStat FOCEI | nlmixr2 FOCEI | Criterion |
|-----------|------|---------------|---------------|-----------|
| CL (L/h) | 0.134 | 0.1335 | 0.1338 | ✅ <15% |
| V (L) | 8.0 | 7.92 | 7.96 | ✅ <15% |
| Ka (h⁻¹) | 1.0 | 0.981 | 0.987 | ✅ <15% |
| corr(CL,V) | 0.50 | 0.47 | 0.49 | ✅ |
| OFV | — | 498.1 | 498.6 | ✅ |Δ|<5 |

**Verdict**: ✅ PASS — correlation recovery within expected sampling variability.

---

## Results: SAEM vs FOCE Parity (WAR-32)

| Parameter | NextStat SAEM | NextStat FOCEI | |Δ|/FOCEI |
|-----------|--------------|---------------|------------|
| CL (L/h) | 0.1331 | 0.1328 | 0.2% |
| V (L) | 7.91 | 7.89 | 0.3% |
| Ka (h⁻¹) | 0.975 | 0.972 | 0.3% |
| OFV | 412.6 | 412.3 | 0.3 |

**Verdict**: ✅ PASS — SAEM and FOCEI agree within 1% on all parameters.

---

## Performance summary

All benchmarks run on Apple M5 (single-thread, `cargo test --release`).

| Dataset | N_subj × N_obs | NextStat FOCEI | NextStat SAEM |
|---------|---------------|----------------|---------------|
| WAR-32 | 32 × 384 | ~8 ms | ~45 ms |
| THEO-12 | 12 × 120 | ~3 ms | ~18 ms |
| PHENO-40 | 40 × 360 | ~9 ms | ~50 ms |
| WAR-CORR-40 | 40 × 320 | ~12 ms | — |

---

## Reproducibility

All benchmarks are deterministic (seeded RNG) and run in CI:

```bash
# Full pharma benchmark suite (Phase 1 + Phase 3)
cargo test -p ns-inference --test pharma_benchmark --test phase3_benchmark -- --nocapture

# Individual datasets
cargo test -p ns-inference --test pharma_benchmark benchmark_warfarin -- --nocapture
cargo test -p ns-inference --test pharma_benchmark benchmark_theophylline -- --nocapture
cargo test -p ns-inference --test pharma_benchmark benchmark_phenobarbital -- --nocapture
cargo test -p ns-inference --test pharma_benchmark benchmark_warfarin_correlated_omega -- --nocapture
```

## Reference tool versions

| Tool | Version | Source |
|------|---------|--------|
| nlmixr2 | 2.1.2 | CRAN (2025-11) |
| NONMEM | 7.5.1 | ICON plc |
| NextStat | 0.9.0 | This repository |

## Conclusion

NextStat FOCE/FOCEI produces estimates within **<5% relative error** of nlmixr2 and NONMEM across all four canonical datasets. SAEM agrees with FOCEI within **<1%**. All runs converge. Diagnostics (GOF, VPC) pass validation gates.

The pharmacometrics stack is **parity-grade** for pilot deployments.

---

## Related

- [Pharma benchmark suite](suites/pharma.md) — test counts and methodology
- [Phase 13 PK tutorial](/docs/tutorials/phase-13-pk.md) — single-subject PK modeling
- [Phase 13 NLME tutorial](/docs/tutorials/phase-13-nlme.md) — population PK with FOCE/SAEM
- [IQ/OQ/PQ validation protocol](/docs/validation/iq-oq-pq-protocol.md) — GxP qualification
