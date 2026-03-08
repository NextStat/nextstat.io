---
title: "Bayesian Prior Conflict Diagnostic Acceptance Criteria (Stable Surface v0)"
status: stable
---

# Bayesian Prior Conflict Diagnostic Acceptance Criteria (Stable Surface v0)

This document defines the release acceptance criteria for the stable public
prior-data conflict diagnostic artifact derived from frozen Bayesian design
reports.

The surface is accepted only when **all** criteria below are true.

## 1. Public surface is explicit

The accepted stable public Python entrypoints are:

- `nextstat.bayes_design.build_beta_binomial_prior_conflict_diagnostic(report_or_path) -> dict`
- `nextstat.bayes_design.build_normal_normal_prior_conflict_diagnostic(report_or_path) -> dict`

Explicitness requirements:

- the diagnostic builders are family-specific; no hidden generic dispatcher is
  part of the stable contract
- each wrapper accepts exactly the documented ingress modes: Python `dict`, JSON
  string, or filesystem path
- diagnostic generation consumes only frozen `*_design_report_v0` artifacts
- diagnostic generation must not rebuild analysis, simulation, posterior
  predictive, prior-sensitivity, report, appendix, or bundle artifacts

## 2. JSON contract is versioned and published

The accepted public diagnostic contract is:

- `schema_version = "nextstat_bayesian_prior_conflict_diagnostic_v0"`

The following must exist and stay in sync:

- published schema:
  - `nextstat config schema --name bayesian_prior_conflict_diagnostic_v0`
- canonical JSON examples:
  - `docs/specs/pharma/beta_binomial_prior_conflict_diagnostic_v0.example.json`
  - `docs/specs/pharma/normal_normal_prior_conflict_diagnostic_v0.example.json`

## 3. Diagnostic completeness is mandatory

An accepted diagnostic artifact must contain, as explicit machine-readable
fields:

- source lineage:
  - `source_report_schema_version`
  - `source_prior_sensitivity_schema_version`
  - `generated_from_frozen_report`
- baseline decision state:
  - `baseline_variant_id`
  - `baseline_recommended_action`
- transparent classification:
  - `conflict_severity`
  - `decision_instability`
  - `thresholds`
  - `metrics`
  - `rationale`
- full campaign summaries:
  - `reported_variant_count`
  - `variant_summaries`

This slice is accepted only if the severity logic is transparent rather than
hidden inside undocumented heuristics.

## 4. Determinism and frozen-input discipline are hard gates

The stable surface is accepted only if:

- repeated diagnostic builds from the same frozen report produce identical JSON
  payloads
- conflict diagnostics perform no hidden execution beyond parsing and validating
  the frozen report and summarizing its explicit prior-sensitivity block
- canonical examples are generated from the live helper and match committed
  fixtures exactly

For this slice, determinism is part of correctness.

## 5. Canonical tooling path is singular

The accepted implementation path is:

- exact design builders -> frozen `*_design_report_v0`
- frozen `*_design_report_v0` -> family-specific prior-conflict diagnostic
  builder

This means:

- the diagnostic surface is a frozen-report post-processing layer
- the diagnostic must derive its summary from the published prior-sensitivity
  section already embedded in the frozen report
- no separate runtime or simulation engine is introduced for this slice

## 6. Mandatory verification gates

The stable surface is accepted only if all of the following pass:

- Python runtime/schema/regression gates:
  - `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q tests/python/test_bayes_design_module_api.py tests/python/test_bayes_design_contract.py tests/python/test_bayes_design_schema_smoke.py tests/python/test_bayes_design_stable_surface_regression.py tests/python/test_bayes_design_checklists_smoke.py`
- CLI schema publication gate:
  - `cargo test -p ns-cli config_schema_can_emit_beta_binomial_design_schemas --test cli_config_schema`
- docs hygiene gate:
  - `python3 scripts/docs/terminology_lint.py --check`

For release-quality changes, the acceptance evidence should include the exact
commands that were run and their pass/fail result.

## 7. Documentation must expose the supported contract

The following published docs are part of the acceptance surface:

- `docs/references/python-api.md`
- `docs/references/bayesian-trial-design-artifacts.md`
- `docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md`
- `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`
- `docs/whitepapers/fda-bayesian-trial-designs.md`

Those docs must explicitly expose:

- the accepted schema name
- canonical example locations
- accepted Python entrypoints
- the frozen-input and no-hidden-execution rule
- the fact that this slice is campaign-based conflict summarization, not
  borrowing-weight adaptation

## 8. No separate benchmark gate for this slice

This slice does **not** require a dedicated `nextstat-bench` benchmark artifact
for acceptance when the change is limited to:

- deterministic post-processing of frozen prior-sensitivity results
- schema publication
- docs/example synchronization

If a future change introduces a public runtime budget or performance claim for
prior-conflict diagnostics, that is a separate acceptance scope and may require
its own `nextstat-bench` gate.

## 9. Out of scope for v0 acceptance

The following are not part of the stable acceptance contract for this slice:

- historical-control borrowing
- actual robust-mixture posterior engines
- automatic prior-weight adaptation
- validation-pack integration for prior-conflict diagnostics
- non-conjugate fallback engines
