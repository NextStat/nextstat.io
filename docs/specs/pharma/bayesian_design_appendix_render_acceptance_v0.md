---
title: "Bayesian Design Appendix Render Acceptance Criteria (Stable Surface v0)"
status: stable
---

# Bayesian Design Appendix Render Acceptance Criteria (Stable Surface v0)

This document defines the release acceptance criteria for the stable public
render surface behind deterministic Markdown/PDF rendering of frozen Bayesian
regulatory appendix artifacts.

The surface is accepted only when **all** criteria below are true.

## 1. Public surface is explicit

The accepted stable public Python entrypoints are:

- `nextstat.bayes_design.render_bayesian_regulatory_appendix_markdown(appendix_or_path) -> str`
- `nextstat.bayes_design.write_bayesian_regulatory_appendix_pdf(pdf_path, appendix_or_path) -> None`

Explicitness requirements:

- the render wrappers accept exactly the documented ingress modes: Python
  `dict`, JSON string, or filesystem path
- the render wrappers accept only the published
  `nextstat_bayesian_design_regulatory_appendix_v0` JSON artifact
- render helpers are pure post-processing and must not rebuild design reports,
  analysis, simulation, posterior-predictive, or prior-sensitivity artifacts
- `write_bayesian_regulatory_appendix_pdf(...)` is additive over the same
  frozen appendix JSON contract used by the Markdown renderer

## 2. Render contract is published and example-backed

The accepted render baseline depends on the published frozen appendix contract:

- `schema_version = "nextstat_bayesian_design_regulatory_appendix_v0"`

The following committed examples must exist and stay in sync with the live
renderers:

- `docs/specs/pharma/beta_binomial_regulatory_appendix_v0.example.json`
- `docs/specs/pharma/beta_binomial_regulatory_appendix_v0.example.md`
- `docs/specs/pharma/normal_normal_regulatory_appendix_v0.example.json`
- `docs/specs/pharma/normal_normal_regulatory_appendix_v0.example.md`

Committed PDF fixtures are not required for the stable contract; deterministic
PDF behavior is enforced through runtime regression tests instead.

## 3. Determinism and frozen-input discipline are hard gates

The stable surface is accepted only if:

- repeated Markdown renders from the same frozen appendix produce identical text
- repeated PDF renders from the same frozen appendix produce byte-identical
  output within the supported project dependency stack
- the PDF metadata is deterministic
- render helpers perform no hidden execution beyond parsing and validating the
  frozen appendix artifact

For this slice, determinism is part of correctness.

## 4. Canonical tooling path is singular

The accepted implementation path is:

- frozen report -> `build_*_regulatory_appendix(...)` -> frozen appendix JSON
- frozen appendix JSON -> `render_bayesian_regulatory_appendix_markdown(...)`
- frozen appendix JSON -> `write_bayesian_regulatory_appendix_pdf(...)`
- `validation-pack/render_validation_pack.sh` for the optional appendix bundle
  path

This means:

- the render surface is appendix-schema-based, not report-schema-based
- `validation-pack/render_validation_pack.sh --bayesian-design-report PATH`
  must use the published appendix render helpers rather than re-deriving prose
  from partial report inputs
- `--json-only` may skip only the appendix PDF file; it must not change the
  frozen appendix JSON or the deterministic Markdown appendix output

## 5. Mandatory verification gates

The stable surface is accepted only if all of the following pass:

- Python runtime/schema/regression gates:
  - `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q tests/python/test_bayes_design_module_api.py tests/python/test_bayes_design_contract.py tests/python/test_bayes_design_stable_surface_regression.py tests/python/test_validation_pack_script_smoke.py tests/python/test_validation_pack_execution_regression.py tests/python/test_bayes_design_checklists_smoke.py`
- docs hygiene gate:
  - `python3 scripts/docs/terminology_lint.py --check`

For release-quality changes, the acceptance evidence should include the exact
commands that were run and their pass/fail result.

## 6. Documentation must expose the supported contract

The following published docs are part of the acceptance surface:

- `docs/references/python-api.md`
- `docs/references/bayesian-trial-design-artifacts.md`
- `docs/references/validation-report.md`
- `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`
- `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`
- `docs/whitepapers/fda-bayesian-trial-designs.md`

Those docs must explicitly expose:

- the accepted render entrypoints
- the committed Markdown example locations
- the frozen-input and no-hidden-execution rule
- the deterministic PDF rule
- the validation-pack behavior for `.md` and optional `.pdf` appendix artifacts

## 7. No separate benchmark gate for this slice

This slice does **not** require a dedicated `nextstat-bench` benchmark artifact
for acceptance when the change is limited to:

- deterministic Markdown/PDF appendix rendering from frozen appendix JSON
- validation-pack wiring for the published render helpers
- docs/example synchronization without a public runtime or performance claim

If a future change introduces a public runtime budget or performance claim for
appendix rendering, that is a separate acceptance scope and may require its own
`nextstat-bench` gate.

## 8. Out of scope for v0 acceptance

The following are not part of the stable acceptance contract for this slice:

- CID / BSA-specific narrative appendix templates
- protocol-language appendix prose generation beyond the deterministic appendix
  JSON/Markdown/PDF render path
- historical-control borrowing posterior engines
- actual robust-mixture posterior engines
- non-conjugate fallback engines
