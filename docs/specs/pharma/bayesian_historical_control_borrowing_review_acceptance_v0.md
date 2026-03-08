# Bayesian Historical-Control Borrowing Review Acceptance v0

**Date**: 2026-03-08  
**Status**: public acceptance policy  
**Scope**: frozen-report historical-control borrowing policy and review surface

## 1. Purpose

This document defines the acceptance criteria for the published
historical-control borrowing review slice on the Bayesian design surface.

It answers one narrow question:

- what must be true before we treat the current borrowing-policy/review
  contracts as an accepted public artifact surface?

## 2. Public entrypoints

The accepted Python entrypoints for this slice are:

- `nextstat.bayes_design.build_beta_binomial_historical_control_borrowing_review(report_or_path, policy_or_path) -> dict`
- `nextstat.bayes_design.build_normal_normal_historical_control_borrowing_review(report_or_path, policy_or_path) -> dict`

These entrypoints are accepted only as frozen-input post-processing helpers.
They must consume:

- a committed `*_design_report_v0` artifact
- a committed `nextstat_bayesian_historical_control_borrowing_policy_v0`
  artifact

They must not implicitly re-run:

- design analysis
- operating-characteristics simulation
- posterior-predictive forecast
- design-report generation
- appendix generation
- bundle generation

## 3. Published contracts

The accepted schema contracts are:

- `nextstat_bayesian_historical_control_borrowing_policy_v0`
- `nextstat_bayesian_historical_control_borrowing_review_v0`

Published schema locations:

- `docs/schemas/pharma/bayesian_historical_control_borrowing_policy_v0.schema.json`
- `docs/schemas/pharma/bayesian_historical_control_borrowing_review_v0.schema.json`

Published example locations:

- `docs/specs/pharma/bayesian_historical_control_borrowing_policy_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_policy_normal_normal_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_review_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_review_normal_normal_v0.example.json`

CLI schema publication must include:

- `nextstat config schema --name bayesian_historical_control_borrowing_policy_v0`
- `nextstat config schema --name bayesian_historical_control_borrowing_review_v0`

## 4. Required artifact behavior

The review artifact is acceptable only if it contains:

- explicit lineage:
  - `source_report_schema_version`
  - `source_policy_schema_version`
  - `source_prior_conflict_schema_version`
- deterministic decision output:
  - `recommended_borrowing_state`
  - `borrowing_eligible`
  - `current_effective_borrowing_fraction`
  - `current_effective_historical_control_sample_size`
- transparent gates:
  - `gating`
  - `diagnostics`
  - `historical_sources`
  - `rationale`

The review must remain explicit about what it is:

- policy review over a frozen exact-slice design report
- not a borrowed posterior update
- not a hidden dynamic-weight engine

## 5. Determinism and hidden-execution rule

This slice is accepted only if:

- identical frozen report + policy inputs produce identical JSON outputs
- the published wrappers accept Python `dict`, JSON string, and filesystem path
  ingress
- the borrowing-review path does not call back into analysis, simulation,
  forecast, report, appendix, or bundle helpers

## 6. Verification gates

The minimum required checks are:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q \
  tests/python/test_bayes_design_module_api.py \
  tests/python/test_bayes_design_contract.py \
  tests/python/test_bayes_design_schema_smoke.py \
  tests/python/test_bayes_design_stable_surface_regression.py \
  tests/python/test_bayes_design_checklists_smoke.py

cargo test -p ns-cli config_schema_can_emit_beta_binomial_design_schemas --test cli_config_schema

python3 scripts/docs/terminology_lint.py --check
```

## 7. Public documentation surface

The accepted docs surface for this slice is:

- `docs/references/bayesian-trial-design-artifacts.md`
- `docs/references/python-api.md`
- `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`
- `docs/whitepapers/fda-bayesian-trial-designs.md`
- this document

Those docs must explicitly expose:

- the accepted schema names
- canonical example locations
- accepted Python entrypoints
- the frozen-input and no-hidden-execution rule
- the fact that this slice is policy review and deterministic gating, not a
  borrowed posterior engine

## 8. No separate benchmark gate for this slice

This slice does **not** require a dedicated `nextstat-bench` benchmark artifact
for acceptance when the change is limited to:

- deterministic frozen-report policy review
- schema publication
- docs/example synchronization

If a future change introduces public runtime budgets or performance wording for
borrowing review or borrowed-engine execution, that is a separate acceptance
scope.

## 9. Out of scope for v0 acceptance

The following are not part of this acceptance contract:

- historical-control borrowing posterior updates inside the exact engine
- automatic borrowing-weight adaptation beyond the published policy/review rules
- report, appendix, or validation-pack embedding of borrowing review artifacts
- actual robust-mixture posterior engines
- non-conjugate fallback engines
