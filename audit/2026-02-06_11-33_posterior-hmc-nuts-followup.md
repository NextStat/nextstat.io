<!--
Audit Follow-up
Generated: 2026-02-06 11:33
Scope: Posterior/HMC/NUTS vs standards (follow-up)
-->

# Posterior/HMC/NUTS Audit Follow-up

Date: 2026-02-06
Project: nextstat.io
Scope: follow-up on `/audit Posterior/HMC/NUTS vs standards`

This note is a delta vs the earlier report in `audit/2026-02-06_01-16_posterior-hmc-nuts.md`.

## Status Summary

- Posterior safety: dimension validation and prior validation are now present.
- Bounds transforms: `(-inf, b)` is supported via `UpperBoundedBijector` (canonical `b - exp(z)`).
- Treedepth diagnostics: treedepth semantics are documented as 0-based; added a unit test to ensure
  `max_treedepth_rate` counts `depth >= max_treedepth`.
- Python surface: `grad_nll()` is now exposed on all model classes (HistFactory + regression + composed GLM),
  with matching stubs and contract tests.

## Evidence (Code Pointers)

- Posterior validation:
  - `crates/ns-inference/src/posterior.rs`
- Upper-bounded transform:
  - `crates/ns-prob/src/transforms.rs`
- Treedepth accounting:
  - `crates/ns-inference/src/nuts.rs`
  - `crates/ns-inference/src/diagnostics.rs`
- Python `grad_nll()` exposure:
  - `bindings/ns-py/src/lib.rs`
  - `bindings/ns-py/python/nextstat/_core.pyi`
  - `tests/python/test_bindings_api.py`

## Remaining Gaps / TODOs

- Algorithm choice documentation: the code uses slice-based NUTS (uniform weights inside slice).
  If we later want Stan-style multinomial NUTS, implement it as an explicit mode and add
  dedicated SBC/quality gates so we can evaluate behavior regressions.

