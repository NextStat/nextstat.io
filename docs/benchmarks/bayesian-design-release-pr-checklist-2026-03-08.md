# Bayesian Design Release PR Checklist

**Date**: 2026-03-08
**Status**: release hardening checklist
**Scope**: stable Bayesian design report, prior conflict, borrowing review, borrowing extension operating characteristics, robust-mixture review, robust-mixture extension operating characteristics, appendix, bundle, and validation-pack integration surface

## Purpose

This checklist is the maintainer-side `PR-ready` gate for the currently
accepted Bayesian design stable surface.

Use it after implementation and docs are complete, but before opening or
shipping a release PR that claims the Bayesian design surface as a stable
product surface.

## Release PR scope

This checklist applies only to the promoted stable subset:

- `beta_binomial_design_v0`
- `normal_normal_design_v0`
- `beta_binomial_design_report_v0`
- `normal_normal_design_report_v0`
- `bayesian_design_regulatory_appendix_v0`
- `bayesian_design_report_bundle_v0`
- `bayesian_prior_conflict_diagnostic_v0`
- `bayesian_historical_control_borrowing_policy_v0`
- `bayesian_historical_control_borrowing_review_v0`
- `bayesian_historical_control_borrowing_operating_characteristics_v0`
- `bayesian_robust_mixture_prior_policy_v0`
- `bayesian_robust_mixture_prior_review_v0`
- `bayesian_robust_mixture_prior_operating_characteristics_v0`
- `nextstat.bayes_design.build_*_design_report(...)`
- `nextstat.bayes_design.build_*_prior_conflict_diagnostic(...)`
- `nextstat.bayes_design.build_*_historical_control_borrowing_review(...)`
- `nextstat.bayes_design.simulate_*_historical_control_borrowing_operating_characteristics(...)`
- `nextstat.bayes_design.build_*_robust_mixture_prior_review(...)`
- `nextstat.bayes_design.simulate_*_robust_mixture_prior_operating_characteristics(...)`
- `nextstat.bayes_design.build_*_regulatory_appendix(...)`
- `nextstat.bayes_design.render_bayesian_regulatory_appendix_markdown(...)`
- `nextstat.bayes_design.write_bayesian_regulatory_appendix_pdf(...)`
- `nextstat.bayes_design.render_*_design_report(...)`
- `nextstat.bayes_design.write_*_design_report_bundle(...)`
- `validation-pack/render_validation_pack.sh --bayesian-design-report PATH`

It does not promote:

- CID / BSA-specific appendix narrative templates
- protocol-language appendix templates beyond the deterministic appendix JSON/Markdown/PDF path
- historical-control borrowing posterior engines or robust-mixture posterior engines
- non-conjugate fallback engines
- platform / multi-arm Bayesian trial designs

Those remain outside the current stable claim until separately promoted.

## Pre-PR checklist

### Contract

- [ ] stable schema names are unchanged or deliberately version-bumped
- [ ] family-specific report, prior-conflict, borrowing-review, robust-mixture-review, appendix, and bundle entrypoints remain explicit
- [ ] backward-compatible ingress modes remain intact for the published wrappers:
  - Python `dict`
  - JSON string
  - filesystem path
- [ ] frozen-report render and bundle paths do not introduce hidden execution
- [ ] frozen-report prior-conflict paths do not introduce hidden execution
- [ ] frozen-report borrowing-review paths do not introduce hidden execution
- [ ] frozen-report robust-mixture-review paths do not introduce hidden execution
- [ ] frozen-report appendix paths do not introduce hidden execution
- [ ] frozen appendix render paths do not introduce hidden execution
- [ ] validation-pack appendix path remains additive and does not introduce hidden execution

### Evidence

- [ ] report acceptance policy is present:
  - `docs/specs/pharma/bayesian_design_report_acceptance_v0.md`
- [ ] prior-conflict acceptance policy is present:
  - `docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md`
- [ ] borrowing-review acceptance policy is present:
  - `docs/specs/pharma/bayesian_historical_control_borrowing_review_acceptance_v0.md`
- [ ] borrowing extension-OC acceptance policy is present:
  - `docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_acceptance_v0.md`
- [ ] robust-mixture-review acceptance policy is present:
  - `docs/specs/pharma/bayesian_robust_mixture_prior_review_acceptance_v0.md`
- [ ] robust-mixture extension-OC acceptance policy is present:
  - `docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_acceptance_v0.md`
- [ ] bundle acceptance policy is present:
  - `docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md`
- [ ] appendix acceptance policy is present:
  - `docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md`
- [ ] appendix render acceptance policy is present:
  - `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`
- [ ] validation-pack acceptance policy is present:
  - `docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md`
- [ ] runtime gate is present:
  - `docs/benchmarks/bayesian-design-report-packaging-runtime-gate.md`
- [ ] release PR checklist is present:
  - `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`
- [ ] artifact/reference page is current:
  - `docs/references/bayesian-trial-design-artifacts.md`
- [ ] product positioning page is current when claims changed:
  - `docs/whitepapers/fda-bayesian-trial-designs.md`

### Verification

- [ ] stable-surface Python gates pass:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q \
  tests/python/test_bayes_design_module_api.py \
  tests/python/test_bayes_design_contract.py \
  tests/python/test_bayes_design_schema_smoke.py \
  tests/python/test_bayes_design_stable_surface_regression.py \
  tests/python/test_validation_pack_script_smoke.py \
  tests/python/test_validation_pack_execution_regression.py \
  tests/python/test_bayesian_design_report_bundle_performance_budget.py \
  tests/python/test_bayesian_design_report_bundle_benchmark_smoke.py \
  tests/python/test_bayes_design_checklists_smoke.py
```

- [ ] docs hygiene gate passes:

```bash
python3 scripts/docs/terminology_lint.py --check
```

- [ ] if CLI schema publication changed, schema export gate passes:

```bash
cargo test -p ns-cli config_schema_can_emit_beta_binomial_design_schemas --test cli_config_schema
```

### Promotion evidence

- [ ] if the change affects the published design-report bundle runtime behavior or public performance wording, a fresh `nextstat-bench` artifact exists
- [ ] the `nextstat-bench` artifact is linked from the PR or release note
- [ ] no benchmark claim relies on terminal-only output without an archived JSON artifact
- [ ] docs-only and test-only changes are not blocked on a new `nextstat-bench` run

### Messaging

- [ ] PR summary names the promoted stable subset explicitly
- [ ] PR summary names the deferred layers explicitly:
  - CID / BSA-specific appendix narrative templates
  - protocol-language appendix templates beyond the deterministic appendix JSON/Markdown/PDF path
  - borrowed posterior engines / robust-mixture posterior engines
  - fallback engines
- [ ] no blanket claim is made about all Bayesian trial-design workflows being stable
- [ ] any performance language is scoped to the current `nextstat-bench` artifact when applicable

## Recommended PR summary structure

Use a short structure:

1. what is stable now
2. what remains outside the stable claim
3. what evidence backs the stable claim
4. how to rerun the stable-surface gates

## Exit condition

The release PR is ready only when every checkbox above is green and the
published stable claim stays within the accepted subset defined by the current
Bayesian design acceptance docs.
