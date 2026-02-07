<!--
Audit Report
Generated: 2026-02-07 03:33:56 +0100
Git Commit: 88d578567f4bb90ea05538a603c750ebac111e3f
Scope: feature
Invoked: /audit feature accelerate-backend
-->

# Project Audit Report

Date: 2026-02-07  
Project: nextstat.io  
Auditor: Codex  
Scope: feature — Apple Accelerate backend (vDSP/vForce)  
Git Commit: 88d578567f4bb90ea05538a603c750ebac111e3f

---

## Executive Summary

- Critical: 0
- Major: 0
- Minor: 4
- Total: 4

Overall Health Score: 92/100  
Top risks: Batch NLL path still allocates large scratch buffers; end-to-end speedups can be capped by per-eval `expected_data()` allocations.

---

## Critical Issues

None found.

## Major Issues

None found.

## Minor Issues

### [Performance] Batch Accelerate NLL allocates large scratch buffers per call
- File: crates/ns-compute/src/accelerate.rs:239
- Evidence: `batch_poisson_nll_accelerate()` allocates 5 `Vec<f64>` of length `total_bins` via `vec![0.0; total_bins]`. (See lines 239–243.)
- Impact: For large `n_toys * n_bins`, allocation + zero-init can dominate runtime and cause allocator churn if called repeatedly.
- Fix: Add a reusable workspace (e.g. `BatchPoissonScratch`) that callers keep per-thread/per-job; optionally avoid repeated zero-init by only growing buffers (`resize`) or using `with_capacity` + `set_len` (carefully, since it’s `unsafe`).
- Confidence: high

### [Performance] Accelerate arithmetic is split across multiple full-vector passes
- File: crates/ns-compute/src/accelerate.rs:155
- Evidence: After `vvlog`, the code does `vmul -> vsub -> vmul -> vadd -> sve` over the full vector.
- Impact: Can become memory-bandwidth bound; may reduce speedups vs a fused loop (especially for smaller `n` where `vvlog` isn’t dominant).
- Fix: Consider a hybrid: keep `vvlog` for the log, then do a single fused Rust loop for `expected + mask*(ln_factorial - obs*ln_expected)` and the reduction (optionally SIMD/autovec).
- Confidence: medium

### [Performance] `PreparedModel::nll()` allocates `expected` every evaluation
- File: crates/ns-translate/src/pyhf/model.rs:1866
- Evidence: `let mut expected = self.model.expected_data(params)?;` returns a fresh `Vec<f64>`.
- Impact: In L-BFGS(-B) the objective is evaluated many times; repeated allocations can cap the benefit of a faster Poisson accumulator.
- Fix: Add an `expected_data_into(&mut Vec<f64>)`/workspace API (or cache a mutable buffer inside `PreparedModel` via interior mutability) so repeated evaluations reuse capacity.
- Confidence: medium

### [Docs] Accelerate enable/disable is not discoverable from the top-level README
- File: README.md:1
- Evidence: No mention of `--features accelerate`, Python `maturin ... --features accelerate`, or `NEXTSTAT_DISABLE_ACCELERATE`.
- Impact: Users on macOS may miss the feature flag and/or the determinism toggle; support friction.
- Fix: Add a short “Apple Accelerate (macOS)” section to `README.md` with build/test commands and the env var.
- Confidence: medium

## Feature Checklist

### Apple Accelerate backend (vDSP/vForce)
- [x] Backend API complete
- [x] Wiring verified
- [x] Error handling
- [x] Tests
- [ ] Docs

## Unverified Areas

- Real-world profiling numbers (wall-clock speedup on representative models and toy counts).
- Memory footprint behavior for very large `batch_poisson_nll_accelerate()` inputs.
- Performance on Intel macOS (Accelerate exists, but throughput/overheads may differ).

## Files Reviewed

- crates/ns-compute/src/accelerate.rs
- crates/ns-compute/src/lib.rs
- crates/ns-compute/build.rs
- crates/ns-compute/benches/simd_benchmark.rs
- crates/ns-translate/src/pyhf/model.rs
- crates/ns-inference/src/batch.rs
- bindings/ns-py/src/lib.rs
- docs/plans/phase-2c-gpu-backends.md

## Recommendations

1. Priority 1: Add reusable workspace for `batch_poisson_nll_accelerate()` if it becomes a hot path.
2. Priority 2: Consider fusing post-`vvlog` arithmetic into a single pass to reduce memory bandwidth pressure.
3. Priority 3: Document `--features accelerate` and `NEXTSTAT_DISABLE_ACCELERATE` in `README.md`.
