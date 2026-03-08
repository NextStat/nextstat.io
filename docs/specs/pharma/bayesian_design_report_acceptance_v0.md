---
title: "Bayesian Design Report Acceptance Criteria (Stable Surface v0)"
status: stable
---

# Bayesian Design Report Acceptance Criteria (Stable Surface v0)

This document defines the release acceptance criteria for the stable product
surface behind the Bayesian clinical trial design report artifacts.

The surface is considered accepted only when **all** criteria below are true.

## 1. Public surface is explicit

The accepted stable public Python entrypoints are:

- `nextstat.bayes_design.build_beta_binomial_design_report(spec_or_path, observed_or_path, campaign_or_path) -> dict`
- `nextstat.bayes_design.render_beta_binomial_design_report(report_or_path) -> str`
- `nextstat.bayes_design.build_normal_normal_design_report(spec_or_path, observed_or_path, campaign_or_path) -> dict`
- `nextstat.bayes_design.render_normal_normal_design_report(report_or_path) -> str`

Explicitness requirements:

- the report builders are family-specific; no hidden generic dispatcher is part of the stable contract
- each wrapper accepts exactly the documented ingress modes: Python `dict`, JSON string, or filesystem path
- report generation is additive on top of the already-shipped design/analysis/OC/forecast/prior-sensitivity artifacts
- the public surface does not require callers to assemble nested report sections manually

## 2. JSON and Markdown contracts are versioned and published

The accepted public report contracts are:

- `schema_version = "nextstat_beta_binomial_design_report_v0"`
- `schema_version = "nextstat_normal_normal_design_report_v0"`

The following must exist and stay in sync:

- published schemas:
  - `nextstat config schema --name beta_binomial_design_report_v0`
  - `nextstat config schema --name normal_normal_design_report_v0`
- canonical JSON examples:
  - `docs/specs/pharma/beta_binomial_design_report_v0.example.json`
  - `docs/specs/pharma/normal_normal_design_report_v0.example.json`
- canonical Markdown examples:
  - `docs/specs/pharma/beta_binomial_design_report_v0.example.md`
  - `docs/specs/pharma/normal_normal_design_report_v0.example.md`

## 3. Report completeness is mandatory

An accepted report artifact must contain, as explicit machine-readable fields:

- the original versioned design spec
- current named-look analysis
- unconditional operating characteristics
- posterior-predictive forecast from current data
- prior-sensitivity results
- provenance block with software version, schema lineage, simulation seed, and replicate count

An accepted Markdown render must explicitly expose:

- all priors
- all looks
- all decision criteria
- all simulation scenarios
- current decision state
- posterior-predictive summary
- prior-sensitivity summary
- software/provenance fields

## 4. Determinism is a hard gate

The stable surface is accepted only if:

- repeated report builds with the same design spec, observed data, prior-sensitivity campaign, and seed produce identical JSON payloads
- repeated Markdown renders of the same report payload produce byte-identical text
- canonical examples are generated from the live implementation and match committed fixtures exactly

For this v0 slice, deterministic behavior is part of correctness, not a best-effort property.

## 5. Canonical tooling path is singular

The accepted implementation path is the shipped Rust core plus the thin Python compatibility layer.

This means:

- report builders delegate to `ns-inference` rather than reimplementing business logic in Python
- Markdown renderers delegate to the Rust report renderer rather than a second independent Python formatter
- CLI schema export, docs, and Python wrappers point to the same versioned artifact names

## 6. Mandatory verification gates

The stable surface is accepted only if all of the following pass:

- Rust core gate:
  - `cargo test -p ns-inference bayes_design --lib`
- Python extension build gate:
  - `./.venv/bin/maturin develop -m bindings/ns-py/Cargo.toml`
- Python runtime/schema/docs gates:
  - `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q tests/python/test_bayes_design_module_api.py tests/python/test_bayes_design_contract.py tests/python/test_bayes_design_schema_smoke.py tests/python/test_bayes_design_stable_surface_regression.py`
- CLI schema publication gate:
  - `cargo test -p ns-cli config_schema_can_emit_beta_binomial_design_schemas --test cli_config_schema`
- docs hygiene gate:
  - `python3 scripts/docs/terminology_lint.py --check`

For release-quality changes, the acceptance evidence should include the exact
commands that were run and their pass/fail result.

## 7. Documentation must expose the supported contract

The following published docs are part of the acceptance surface:

- `docs/references/bayesian-trial-design-artifacts.md`
- `docs/references/python-api.md`
- `docs/whitepapers/fda-bayesian-trial-designs.md`
- `docs/specs/pharma/bayesian_design_report_acceptance_v0.md`
- `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`

Those docs must explicitly expose:

- accepted schema names
- canonical example locations
- accepted Python entrypoints
- deterministic/report-completeness expectations
- regression coverage for backward-compatible ingress modes and frozen-report render behavior
- current v0 scope limits

## 8. No separate benchmark gate for this slice

This slice does **not** require an additional performance benchmark gate for acceptance when the change is limited to:

- report assembly
- Markdown rendering
- schema publication
- docs/example synchronization

If a future change modifies the underlying exact posterior or simulation numerics,
that is a different acceptance scope and may require a dedicated benchmark or
validation gate.

## 9. Out of scope for v0 acceptance

The following are not part of the stable acceptance contract for this slice:

- RunBundle export helpers for design reports
- validation-pack or PDF appendix packaging
- historical-control borrowing posterior engines
- actual robust-mixture posterior engines
- embedded prior-data conflict diagnostics inside `*_design_report_v0` beyond the
  separately accepted frozen-report post-processing artifact
- non-conjugate posterior-engine fallback for report generation
