---
title: "Bayesian Design Regulatory Appendix Acceptance Criteria (Stable Surface v0)"
status: stable
---

# Bayesian Design Regulatory Appendix Acceptance Criteria (Stable Surface v0)

This document defines the release acceptance criteria for the stable product
surface behind the Bayesian regulatory appendix JSON artifact.

The surface is accepted only when **all** criteria below are true.

## 1. Public surface is explicit

The accepted stable public Python entrypoints are:

- `nextstat.bayes_design.build_beta_binomial_regulatory_appendix(report_or_path) -> dict`
- `nextstat.bayes_design.build_normal_normal_regulatory_appendix(report_or_path) -> dict`

Explicitness requirements:

- appendix builders are family-specific; no hidden generic dispatcher is part of the stable contract
- each wrapper accepts exactly the documented ingress modes: Python `dict`, JSON string, or filesystem path
- appendix generation consumes only frozen `*_design_report_v0` artifacts
- appendix generation must not rebuild analysis, simulation, posterior-predictive, or prior-sensitivity outputs

## 2. JSON contract is versioned and published

The accepted public appendix contract is:

- `schema_version = "nextstat_bayesian_design_regulatory_appendix_v0"`

The following must exist and stay in sync:

- published schema:
  - `nextstat config schema --name bayesian_design_regulatory_appendix_v0`
- canonical JSON examples:
  - `docs/specs/pharma/beta_binomial_regulatory_appendix_v0.example.json`
  - `docs/specs/pharma/normal_normal_regulatory_appendix_v0.example.json`

## 3. Appendix completeness is mandatory

An accepted appendix artifact must contain, as explicit machine-readable
sections:

- design summary
- prior specification
- decision rules
- current analysis summary
- operating-characteristics summary
- posterior-predictive summary
- prior-sensitivity summary
- provenance summary

The artifact must expose both:

- `required_sections`
- `section_order`

Those lists are part of the stable render contract for future validation-pack or
PDF layers.

## 4. Determinism and frozen-input discipline are hard gates

The stable surface is accepted only if:

- repeated appendix builds from the same frozen report produce identical JSON payloads
- appendix generation performs no hidden execution beyond parsing and validating the frozen report
- canonical examples are generated from the live helper and match committed fixtures exactly

For this slice, determinism is part of correctness.

## 5. Canonical tooling path is singular

The accepted implementation path is the thin Python layer on top of the already
shipped frozen report surface.

This means:

- appendix builders must validate the frozen report family/schema explicitly
- appendix builders must derive section summaries from the frozen report instead of calling back into analysis/simulation engines
- CLI schema export, docs, examples, and Python wrappers point to the same versioned artifact name

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

- `docs/references/bayesian-trial-design-artifacts.md`
- `docs/references/python-api.md`
- `docs/whitepapers/fda-bayesian-trial-designs.md`
- `docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`
- `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`

Those docs must explicitly expose:

- the accepted schema name
- canonical example locations
- accepted Python entrypoints
- the required appendix sections
- the frozen-input and no-hidden-execution rule
- the fact that deterministic Markdown/PDF rendering is a separately accepted
  layer documented in `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`

## 8. No separate benchmark gate for this slice

This slice does **not** require an additional performance benchmark gate for
acceptance when the change is limited to:

- frozen appendix assembly
- schema publication
- docs/example synchronization

If a future change introduces runtime-affecting validation-pack integration or
appendix-template expansion, that is a different acceptance scope and may
require a dedicated benchmark gate on `nextstat-bench`.

## 9. Out of scope for v0 acceptance

The following are not part of the stable acceptance contract for this slice:

- validation-pack integration
- protocol-language templates beyond the structured JSON appendix blocks
- historical-control borrowing posterior engines
- actual robust-mixture posterior engines
- non-conjugate fallback engines
