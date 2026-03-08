---
title: "Bayesian Design Report Bundle Acceptance Criteria (Stable Surface v0)"
status: stable
---

# Bayesian Design Report Bundle Acceptance Criteria (Stable Surface v0)

This document defines the release acceptance criteria for the stable public
bundle surface built on top of the shipped Bayesian design report artifacts.

The surface is accepted only when **all** criteria below are true.

## 1. Public surface is explicit

The accepted stable public Python entrypoints are:

- `nextstat.bayes_design.write_beta_binomial_design_report_bundle(bundle_dir, report_or_path) -> dict`
- `nextstat.bayes_design.write_normal_normal_design_report_bundle(bundle_dir, report_or_path) -> dict`

Explicitness requirements:

- bundle writers are family-specific; no hidden generic dispatcher is public API
- bundle writers accept frozen report artifacts only: Python `dict`, JSON string, or filesystem path
- packaging does not accept design spec, observed data, or prior campaign inputs directly
- packaging does not silently re-run analysis, simulation, or prior-sensitivity computation

## 2. Bundle summary contract is versioned and published

The accepted public summary contract is:

- `schema_version = "nextstat_bayesian_design_report_bundle_v0"`

The following must exist and stay in sync:

- published schema:
  - `nextstat config schema --name bayesian_design_report_bundle_v0`
- canonical JSON examples:
  - `docs/specs/pharma/beta_binomial_design_report_bundle_v0.example.json`
  - `docs/specs/pharma/normal_normal_design_report_bundle_v0.example.json`

## 3. Bundle layout is frozen

An accepted bundle summary must expose the frozen layout explicitly:

- `meta.json`
- `manifest.json`
- `inputs/input.json`
- `outputs/design_report.md`
- `outputs/design_spec.json`
- `outputs/current_analysis.json`
- `outputs/operating_characteristics.json`
- `outputs/posterior_predictive.json`
- `outputs/prior_sensitivity.json`
- `outputs/provenance.json`

The layout is part of the stable contract, not an implementation detail.

## 4. Determinism and no-hidden-execution are hard gates

The stable bundle surface is accepted only if:

- repeated packaging of the same frozen report artifact produces byte-identical bundle contents
- `meta.json.created_unix_ms` is normalized to `0`
- `manifest.json` inventories the final on-disk file set after all bundle artifacts are written
- inline JSON inputs do not depend on temporary-path provenance in the final `meta.json`

For this slice, deterministic packaging is part of correctness.

## 5. Canonical tooling path is singular

The accepted implementation path is:

- family-specific Python bundle helpers
- `nextstat.audit.write_bundle(..., deterministic=True)` as the provenance baseline
- bundle extras assembled from the frozen report artifact, not from recomputation

## 6. Mandatory verification gates

The stable surface is accepted only if all of the following pass:

- Python bundle/runtime gates:
  - `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q tests/python/test_audit_run_bundle.py tests/python/test_bayes_design_module_api.py tests/python/test_bayes_design_contract.py tests/python/test_bayes_design_schema_smoke.py tests/python/test_bayes_design_stable_surface_regression.py`
- CLI schema publication gate:
  - `cargo test -p ns-cli config_schema_can_emit_beta_binomial_design_schemas --test cli_config_schema`
- docs hygiene gate:
  - `python3 scripts/docs/terminology_lint.py --check`

For runtime-affecting packaging changes, the following benchmark gate is also mandatory:

- `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python scripts/benchmarks/bench_bayesian_design_report_bundle.py --deterministic --out bench_results/bayesian_design_report_bundle/summary.json`

That benchmark gate is a conditional promotion gate on `nextstat-bench`:

- docs-only and internal-only changes do not require a `nextstat-bench` run
- runtime-affecting packaging changes do require a fresh `nextstat-bench` artifact before promotion

The committed benchmark contract for that gate is:

- runtime gate doc:
  - `docs/benchmarks/bayesian-design-report-packaging-runtime-gate.md`
- benchmark result schema:
  - `docs/schemas/benchmarks/bayesian_design_report_bundle_benchmark_result_v1.schema.json`
- canonical benchmark example:
  - `docs/specs/pharma/bayesian_design_report_bundle_benchmark_result_v1.example.json`
- performance budget manifest:
  - `scripts/bayesian_design_report_bundle_performance_budget_v1.json`
- performance budget schema:
  - `docs/schemas/benchmarks/nextstat_bayesian_design_report_bundle_performance_budget_v1.schema.json`

## 7. Documentation must expose the supported contract

The following published docs are part of the acceptance surface:

- `docs/references/bayesian-trial-design-artifacts.md`
- `docs/references/python-api.md`
- `docs/references/cli.md`
- `docs/whitepapers/fda-bayesian-trial-designs.md`
- `docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md`
- `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`

Those docs must explicitly expose:

- the bundle summary schema name
- canonical example locations
- family-specific bundle writer entrypoints
- deterministic layout and no-hidden-execution guarantees
- the runtime benchmark gate and its committed budget manifest
- regression coverage for frozen-report-only bundle execution and backward-compatible ingress modes
- the `nextstat-bench` rule for runtime-affecting packaging changes
- current scope limits

## 8. Out of scope for v0 acceptance

The following are not part of this bundle-surface acceptance contract:

- validation-pack or PDF appendix packaging
- benchmark-promotion claims beyond the committed `nextstat-bench` packaging gate
- signature sidecars or signed manifests
- historical borrowing posterior engines / actual robust-mixture posterior engines
- non-conjugate fallback engine packaging
