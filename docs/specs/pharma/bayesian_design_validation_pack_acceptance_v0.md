---
title: "Bayesian Design Validation-Pack Integration Acceptance Criteria (Stable Surface v0)"
status: stable
---

# Bayesian Design Validation-Pack Integration Acceptance Criteria (Stable Surface v0)

This document defines the release acceptance criteria for the stable public
validation-pack integration surface behind frozen Bayesian design appendix
assembly.

The surface is accepted only when **all** criteria below are true.

## 1. Public surface is explicit

The accepted public entrypoint is:

- `bash validation-pack/render_validation_pack.sh --bayesian-design-report PATH`

Explicitness requirements:

- the flag accepts only a filesystem path to a frozen `*_design_report_v0` JSON artifact
- the accepted report families are:
  - `nextstat_beta_binomial_design_report_v0`
  - `nextstat_normal_normal_design_report_v0`
- the flag remains additive: omitting it preserves the pre-existing validation-pack behavior and manifest layout
- the integration consumes frozen evidence only and must not rebuild design analysis, simulation, report generation, or bundle packaging artifacts

## 2. Output contract is fixed

When `--bayesian-design-report PATH` is supplied, the accepted validation-pack
artifact set must include:

- `bayesian_design_report.json`
- `bayesian_design_regulatory_appendix.json`
- `bayesian_design_regulatory_appendix.md`
- `bayesian_design_regulatory_appendix_v0.schema.json`
- exactly one family-specific report schema copy:
  - `beta_binomial_design_report_v0.schema.json`, or
  - `normal_normal_design_report_v0.schema.json`

When `--json-only` is **not** supplied, the accepted artifact set must also
include:

- `bayesian_design_regulatory_appendix.pdf`

The copied report must preserve the original frozen payload unchanged.

## 3. Manifest behavior is deterministic

The accepted manifest behavior is:

- Bayesian appendix artifacts are included in `validation_pack_manifest.json` only when the flag is supplied
- repeated deterministic runs over identical frozen inputs produce byte-identical `validation_pack_manifest.json`
- the manifest continues to cover the core validation-pack artifacts even when Bayesian appendix artifacts are absent

## 4. Frozen-input and no-hidden-execution rules are hard gates

The stable surface is accepted only if:

- appendix assembly is performed from the frozen design report only
- the integration does not call back into `build_*_design_report`, `render_*_design_report`, `write_*_design_report_bundle`, `analyze_*`, `simulate_*`, `forecast_*`, or `analyze_*_prior_sensitivity`
- `validation-pack/render_validation_pack.sh` remains backward-compatible for non-Bayesian invocations

For this slice, no-hidden-execution is part of correctness.

## 5. Canonical tooling path is singular

The accepted implementation path is:

- `validation-pack/render_validation_pack.sh` for orchestration
- `nextstat.bayes_design.build_*_regulatory_appendix(...)` for frozen appendix assembly
- `nextstat.bayes_design.render_bayesian_regulatory_appendix_markdown(...)` for deterministic appendix Markdown
- `nextstat.bayes_design.write_bayesian_regulatory_appendix_pdf(...)` for deterministic appendix PDF when PDF rendering is enabled
- published report and appendix schemas under `docs/schemas/pharma/`

This means:

- family dispatch must be explicit and schema-based
- the appendix schema copy must match the published `bayesian_design_regulatory_appendix_v0` contract
- the validation-pack docs and Bayesian artifact docs must describe the same artifact names and scope limits

## 6. Mandatory verification gates

The stable surface is accepted only if all of the following pass:

- validation-pack runtime and regression gates:
  - `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q tests/python/test_validation_pack_script_smoke.py tests/python/test_validation_pack_execution_regression.py`
- Bayesian docs/checklist gate:
  - `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q tests/python/test_bayes_design_checklists_smoke.py`
- docs hygiene gate:
  - `python3 scripts/docs/terminology_lint.py --check`

For release-quality changes, the evidence should include the exact commands that
were run and their pass/fail result.

## 7. Documentation must expose the supported contract

The following published docs are part of the acceptance surface:

- `docs/references/validation-report.md`
- `docs/references/bayesian-trial-design-artifacts.md`
- `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md`
- `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`

Those docs must explicitly expose:

- the `--bayesian-design-report PATH` entrypoint
- the optional validation-pack artifact names
- the frozen-input and no-hidden-execution rule
- the fact that `bayesian_design_regulatory_appendix.md` is always emitted for
  the Bayesian appendix path
- the fact that `bayesian_design_regulatory_appendix.pdf` is emitted unless
  `--json-only` is supplied

## 8. No separate benchmark gate for this slice

This slice does **not** require a separate benchmark artifact for acceptance
when the change is limited to:

- deterministic copying of a frozen design report
- deterministic appendix assembly from the frozen report
- manifest and docs synchronization for the optional artifact path

If a future change introduces a public runtime budget or public performance
claim for the validation-pack appendix path, that is a separate acceptance
scope and may require its own `nextstat-bench` gate.

## 9. Out of scope for v0 acceptance

The following are not part of the stable acceptance contract for this slice:

- protocol-language appendix templates beyond the structured JSON/Markdown/PDF
  appendix artifacts
- historical borrowing posterior engines or actual robust-mixture posterior engines
- non-conjugate fallback engines
