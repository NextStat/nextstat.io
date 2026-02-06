<!--
Audit Report
Generated: 2026-02-06 01:16:44
Git Commit: 07ff9484f65b2ae87b55c7f931cf8344cba1df58
Scope: feature
Invoked: /audit Posterior/HMC/NUTS vs standards
-->

# Project Audit Report

Date: 2026-02-06
Project: nextstat.io
Auditor: Codex
Scope: Posterior/HMC/NUTS implementation in `crates/ns-inference` vs `docs/plans/standards.md`
Git Commit: 07ff9484f65b2ae87b55c7f931cf8344cba1df58

---

## Executive Summary

- Critical: 0
- Major: 4
- Minor: 2
- Total: 6

Overall Health Score: 78/100
Top risks: silent dimension mismatches + incorrect treedepth accounting can hide sampler pathologies and make diagnostics misleading.

---

## Critical Issues

None found in this scope.

## Major Issues

### [Diagnostics] `max_treedepth_rate` can be wrong / never trigger due to treedepth semantics mismatch
- File: `crates/ns-inference/src/nuts.rs:265`
- Evidence: the NUTS loop uses `while depth < max_treedepth` and returns `depth` as “tree depth” (see `crates/ns-inference/src/nuts.rs:263-266` and `crates/ns-inference/src/nuts.rs:344`). Current control-flow returns `depth == max_treedepth` when the last built subtree depth was `max_treedepth - 1` (off-by-one), and may never represent “hit max treedepth” consistently.
- Impact: `crates/ns-inference/src/diagnostics.rs` computes `max_treedepth_rate` by checking `tree_depth >= max_treedepth` (`crates/ns-inference/src/diagnostics.rs:318-326`), which can under-report hitting the treedepth cap or report it inconsistently.
- Fix:
  - Make treedepth semantics explicit and consistent with Stan: allow building depth up to `max_treedepth` and return “depth reached” as the maximum built depth (0-based).
  - Ensure `tree_depths[]` stores this “depth reached” and `max_treedepth_rate` compares against the same convention.
- Confidence: high

### [Transforms] Upper-bounded-only parameters use a heuristic sigmoid window, not the specified bijector
- File: `crates/ns-prob/src/transforms.rs:194-201`
- Evidence: for bounds `(-inf, b)` it uses `SigmoidBijector::new(hi - 100.0, hi)` as an approximation (“for simplicity”) instead of the canonical `theta = b - exp(z)` mapping from `docs/plans/standards.md` section “Unconstrained parameterization (required)”.
- Impact: incorrect geometry / implicit prior in unconstrained space for upper-bounded parameters; can materially affect sampling and MAP/MLE sanity. Also makes behavior depend on an arbitrary constant `100.0`.
- Fix:
  - Implement `UpperBoundedBijector` with `theta = upper - exp(z)`, `log|J| = z`, `d/dz log|J| = 1`.
  - Use it for `(lo=-inf, hi finite)` selection.
- Confidence: high

### [Safety] Posterior can panic or silently compute wrong values on dimension mismatch
- File: `crates/ns-inference/src/posterior.rs:39-113`
- Evidence:
  - `Posterior::new` trusts `parameter_bounds().len()` matches `dim()` (`crates/ns-inference/src/posterior.rs:39-44`).
  - `logpdf` and `grad` index `theta[i]` while iterating priors without checking `theta.len()` (`crates/ns-inference/src/posterior.rs:73-80`, `crates/ns-inference/src/posterior.rs:95-102`).
  - `logpdf_unconstrained`/`grad_unconstrained` call `transform.forward(z)` which uses `zip`, silently truncating when `z.len() != dim()` (`crates/ns-prob/src/transforms.rs:214-217`).
  - `with_priors` hard-panics via `assert_eq!` (`crates/ns-inference/src/posterior.rs:47-51`).
- Impact: panics in core sampling paths, or silent truncation that yields incorrect densities/gradients. Either breaks determinism/quality gates and can create hard-to-debug sampler failures.
- Fix:
  - Validate lengths (`theta.len() == dim`, `z.len() == dim`) and return `ns_core::Error::Validation` on mismatch.
  - Replace `assert_eq!` in `with_priors` with a fallible API (`Result`) or safe handling.
  - Normalize `bounds` length to `dim()` in `Posterior::new` (pad with `(-inf, inf)` or error).
- Confidence: high

### [Validation] Prior parameters aren’t validated (width can be 0/NaN)
- File: `crates/ns-inference/src/posterior.rs:75-100`
- Evidence: `Prior::Normal { width }` is used as divisor and squared divisor without checking `width > 0` and finite.
- Impact: NaNs/inf in logpdf/grad causing divergences and non-reproducible failures (depending on floating behavior).
- Fix: validate `width.is_finite() && width > 0` and error otherwise.
- Confidence: high

## Minor Issues

### [Standards Alignment] Quality-gate thresholds in ignored tests are looser than “Phase 3” targets
- File: `crates/ns-inference/src/nuts.rs:638-674`
- Evidence: ignored quality gate uses `R-hat < 1.05`, divergence < 10%, E-BFMI > 0.2; `docs/plans/standards.md` suggests tighter targets on toy tasks (e.g., `R-hat < 1.01`, divergence < 1% as an orientation).
- Impact: tests may pass despite mediocre sampling quality; not a correctness bug but weakens regression value.
- Fix: tighten thresholds gradually or add a second “strict” ignored test to track progress.
- Confidence: medium

### [Docs/Comments] “HEP doesn’t have upper-only bounds” assumption is baked into code comments
- File: `crates/ns-prob/src/transforms.rs:196-199`
- Evidence: comment assumes domain constraints, but Phase 5 “universal” API expands beyond HEP.
- Impact: maintainability; future contributors may keep adding heuristics instead of correct bijectors.
- Fix: remove assumption and implement canonical transform(s).
- Confidence: high

## Feature Checklist

### Posterior/HMC/NUTS
- [x] Backend API complete (basic implementation exists)
- [ ] Wiring verified (treedepth semantics, validation and transform selection need fixes)
- [ ] Error handling (dimension + prior validation missing)
- [x] Tests (unit tests exist; some ignored “quality gate” tests)
- [ ] Docs (standards alignment notes should be kept consistent)

## Unverified Areas

- External parity vs Stan/NumPyro for NUTS diagnostics beyond the included tests.
- Performance impacts of additional validation checks (should be negligible).

## Files Reviewed

- `docs/plans/standards.md`
- `crates/ns-inference/src/posterior.rs`
- `crates/ns-inference/src/hmc.rs`
- `crates/ns-inference/src/nuts.rs`
- `crates/ns-inference/src/adapt.rs`
- `crates/ns-inference/src/diagnostics.rs`
- `crates/ns-prob/src/transforms.rs`

## Recommendations

1. Priority 1: fix treedepth semantics + `max_treedepth_rate` correctness and add explicit comments about 0-based depth meaning.
2. Priority 2: add dimension/prior validation in `Posterior` and eliminate `assert_eq!` from public API.
3. Priority 3: implement `UpperBoundedBijector` and remove heuristic sigmoid approximation for upper-only bounds.
