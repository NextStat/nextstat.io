<!--
Audit Report
Generated: 2026-02-05 20:11:41 +0100
Git Commit: unknown
Scope: feature (docs/plans + Phase 1 implementation)
Invoked: /audit phase1-plans-impl
-->

# Project Audit Report

Date: 2026-02-05  
Project: NextStat  
Auditor: Codex  
Scope: `docs/plans` + Phase 1 (pyhf translate + likelihood + MLE + Python + CLI)  
Git Commit: unknown

---

## Executive Summary

- Critical: 0
- Major: 0
- Minor: 4
- Design decisions (accepted): 2
- Total: 6

Overall Health Score: 92/100
Notes: architectural coupling `ns-inference` → `ns-translate` accepted as YAGNI for Phase 1–2; `n_evaluations` stores `n_iter` (argmin limitation, accepted). Remaining stubs (GPU/viz) are expected but need clear gating.

---

## Critical Issues

None found in audited scope.

## Design Decisions (Accepted)

### [Architecture] Inference layer depends on `ns_translate` concrete model
- File: crates/ns-inference/src/mle.rs:6
- Evidence: `use ns_translate::pyhf::HistFactoryModel;` and `fit(&self, model: &HistFactoryModel)`
- Impact: breaks dependency inversion described in plans; makes it harder to reuse inference for non-pyhf models and to route through `ns_core::ComputeBackend` / future GPU backends.
- Fix: introduce a small core trait (e.g., `ns_core::ModelLike` with `nll(params)`, `gradient(params)`, `bounds`, `names`, `poi_index`) implemented by HistFactoryModel; make `MaximumLikelihoodEstimator` generic over it and move HistFactory-specific code to `ns-translate` or adapters.
- Confidence: high
- **Status: Accepted (deferred).** Deliberate pragmatic decision for Phase 1-2. With only one concrete model type (HistFactory), introducing an abstract `ModelLike` trait now would be premature abstraction — it adds a layer of indirection without practical benefit and complicates passing parameters, bounds, and constraint info through a generic trait. When a second model type appears, the trait will be extracted via refactoring. Until then, this is YAGNI.

### [Metrics] `FitResult.n_evaluations` currently stores iterations (`n_iter`)
- File: crates/ns-inference/src/mle.rs:73
- Evidence: passes `result.n_iter as usize` into `FitResult::{new,with_covariance}` `n_evaluations`
- Impact: confusing reporting/telemetry; CLI/py bindings expose this as "evaluations".
- Fix: rename to `n_iter` or plumb real function evaluations from argmin state if available; update CLI/Python field names.
- Confidence: high
- **Status: Accepted (known limitation).** This is a constraint of the `argmin` crate — `IterState` provides `.get_iter()` (iteration count) but not a separate function evaluation counter. Each L-BFGS iteration includes line-search with multiple `cost()` + `gradient()` calls, so `n_iter < n_fev` always. For exact `n_fev`, a counting wrapper around `ObjectiveFunction` would be needed (atomic counter on each `cost`/`gradient` call). The current `n_iter` value still provides useful convergence information and does not affect fit correctness.

## Minor Issues

### [Stubs] GPU backends are compile-time stubs
- File: crates/ns-compute/src/cuda.rs:5
- Evidence: “stub implementation … returns `Error::NotImplemented`”
- Impact: expected for Phase 2C, but ensure feature-gating + docs avoid implying CUDA/Metal works today.
- Fix: keep as-is; ensure these modules are feature-gated and add a small smoke test that the default build does not enable GPU features.
- Confidence: high

### [Stubs] Visualization crate is a placeholder
- File: crates/ns-viz/src/lib.rs:11
- Evidence: `pub fn placeholder() { … }`
- Impact: acceptable pre-Phase 3; but should not be mistaken as shipped API.
- Fix: keep placeholder private or mark as `#[doc(hidden)]`, and/or add a clear “Phase 3” note in crate docs.
- Confidence: medium

### [Testing] Slow toy bias test is opt-in and requires env flag
- File: tests/python/test_bias_pulls.py:1
- Evidence: requires `NS_RUN_SLOW=1` and uses `pytest.mark.slow`
- Impact: might be missed if not documented/automated.
- Fix: add a nightly workflow for `pytest -m slow` and publish bias/coverage artifacts (Phase 3 plan already covers this).
- Confidence: medium

## Feature Checklist

### Phase 1: HistFactory core + parity
- [x] pyhf JSON workspace parsing
- [x] Modifiers (normfactor/shapesys/normsys/histosys/staterror/lumi/shapefactor)
- [x] NLL parity vs pyhf (fixtures) (`tests/python/test_pyhf_validation.py`)
- [x] MLE fit + uncertainties (Hessian)
- [x] Bounds handling (`LbfgsbOptimizer`)
- [x] CLI `nextstat fit` outputs JSON
- [x] Python bindings + convenience API (`nextstat.from_pyhf`, `nextstat.fit`)
- [x] (P1/opt-in) toy pull regression (`tests/python/test_bias_pulls.py`)

## Unverified Areas

- Phase 2A/2B performance work (SIMD, deterministic reductions)
- GPU backends (Metal/CUDA) beyond stub behavior
- NUTS/HMC implementation (Phase 3 plan only)
- Release automation end-to-end (tagging + publishing)

## Files Reviewed

- docs/plans/README.md
- docs/plans/standards.md
- docs/plans/versions.md
- docs/plans/phase-1-mvp-alpha.md
- docs/plans/phase-0-infrastructure.md
- docs/plans/2026-02-05-nextstat-implementation-plan.md
- crates/ns-translate/src/pyhf/model.rs
- crates/ns-inference/src/mle.rs
- crates/ns-inference/src/optimizer.rs
- crates/ns-cli/src/main.rs
- bindings/ns-py/src/lib.rs
- tests/python/test_pyhf_validation.py
- tests/python/test_bias_pulls.py
- .github/workflows/python-tests.yml

## Recommendations

1. ~~Priority 1: decouple `ns-inference` from `ns-translate` via a core model trait + adapters.~~ → **Deferred** (accepted YAGNI, revisit when second model type is added).
2. ~~Priority 2: standardize reporting fields (`n_iter` vs `nfev`).~~ → **Accepted** as-is (argmin limitation); field rename in CLI/Python output planned for Phase 3 polish.
3. Priority 1 (now): add nightly slow-toy workflow + artifacted bias/coverage report (align with Phase 3).
4. Priority 2: feature-gate GPU stubs + smoke test default build excludes them.
5. Priority 3: mark `ns-viz` placeholder as `#[doc(hidden)]`.

## Save Report (Mandatory)

Saved to:
- `audit/2026-02-05_20-11_phase1-plans-impl.md`

