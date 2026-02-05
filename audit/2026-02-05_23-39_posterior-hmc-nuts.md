<!--
Audit Report
Generated: 2026-02-05 23:39:38
Git Commit: e9ecbfc
Scope: Posterior/HMC/NUTS vs docs/plans/standards.md (section 7)
Invoked: /audit posterior-hmc-nuts
-->

# Project Audit Report

Date: 2026-02-05  
Project: NextStat (nextstat.io)  
Auditor: Codex  
Scope: Posterior/HMC/NUTS vs `docs/plans/standards.md` (Bayesian / Posterior contract)  
Git Commit: e9ecbfc

---

## Executive Summary

- Critical: 0
- Major: 2
- Minor: 2
- Total: 4

Overall Health Score: 90/100  
Top risks: NUTS correctness (slice variable + stopping/merge semantics) and diagnostics misreporting (max treedepth rate) could silently invalidate Bayesian results.

---

## Critical Issues

None found in this scoped audit after applying fixes and running `cargo test -p ns-inference`.

---

## Major Issues

### [Correctness] NUTS was missing slice-variable semantics and had incorrect stop/merge behavior
- File: `crates/ns-inference/src/nuts.rs`
- Evidence:
  - Previous transition logic could stop before merging a turning/divergent subtree, excluding valid points from the multinomial proposal set.
  - Slice variable `log_u` was not enforced; weights were computed without truncation to the slice.
- Impact:
  - Samples may not target the intended posterior; divergence/turning behavior could bias proposals.
- Fix:
  - Added slice variable (`log_u = ln(rand) - H0`) and enforced `log_u <= -H` in leaf weights.
  - Ensured subtrees are merged before honoring turning/divergence stop conditions.
  - Propagated turning correctly across merges.
- Confidence: high

### [Correctness] `max_treedepth_rate` did not measure “hit configured max treedepth”
- File: `crates/ns-inference/src/diagnostics.rs`
- Evidence:
  - Previous logic computed the max observed depth in the run and counted depths >= that value, which is not the same as “hit the configured max”.
- Impact:
  - Misleading diagnostics; could hide treedepth saturation regressions.
- Fix:
  - Added `max_treedepth` to `Chain` and compute rate as `depth >= max_treedepth`.
- Confidence: high

---

## Minor Issues

### [Docs/Accuracy] Diagnostics module docstring over-claimed rank-normalized diagnostics
- File: `crates/ns-inference/src/diagnostics.rs`
- Evidence:
  - Header claimed Vehtari et al. (2021) rank-normalized diagnostics, but the implementation is split R-hat + a simple ESS estimator.
- Impact:
  - Readers may assume stronger statistical guarantees than implemented.
- Fix:
  - Updated module docs to explicitly state current behavior and limitation.
- Confidence: high

### [Edge Case] Upper-bounded-only transform is a placeholder
- File: `crates/ns-inference/src/transforms.rs`
- Evidence:
  - Upper-only bounds path uses a “wide Sigmoid” approximation (`Sigmoid(hi-100, hi)`).
- Impact:
  - Incorrect Jacobian/transform if a model introduces upper-only bounded parameters.
- Fix:
  - Not addressed in this patch; recommend implementing a proper `UpperBounded` bijector (`theta = upper - exp(z)` with correct log|J|).
- Confidence: medium

---

## Feature Checklist

### Posterior/HMC/NUTS (Phase 3 Bayesian Contract)
- [x] Posterior logpdf matches `-model.nll + priors` without double-counting constraints
- [x] Unconstrained transforms include `log|det J|` and chain-rule gradients
- [x] Leapfrog integrator uses potential `U = -logpdf` with gradients
- [x] NUTS implements Stan-style multinomial selection with slice truncation
- [x] Determinism via fixed seeds (unit tests)
- [x] Diagnostics include split R-hat and ESS (lightweight)
- [ ] Rank-normalized diagnostics (Vehtari 2021) (explicitly out of scope for this patch)

---

## Unverified Areas

- Statistical calibration on golden Bayesian toy targets (Normal/MVN/funnel) beyond “runs + deterministic” unit tests.

---

## Files Reviewed

- `crates/ns-inference/src/posterior.rs`
- `crates/ns-inference/src/transforms.rs`
- `crates/ns-inference/src/hmc.rs`
- `crates/ns-inference/src/nuts.rs`
- `crates/ns-inference/src/adapt.rs`
- `crates/ns-inference/src/chain.rs`
- `crates/ns-inference/src/diagnostics.rs`
- `docs/plans/standards.md`

---

## Recommendations

1. Priority 1: Add “golden” Bayesian distribution tests (1D Normal, MVN dim=4) with statistical tolerances (mean/var/cov) and divergence/treedepth thresholds.
2. Priority 2: Implement proper upper-bounded bijector to remove placeholder behavior.
3. Priority 3: Upgrade diagnostics to rank-normalized + folded variants if used for serious convergence gating.

