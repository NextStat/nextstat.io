# Bayesian Robust-Mixture Prior Operating Characteristics Acceptance v0

**Date**: 2026-03-08  
**Status**: public acceptance policy  
**Scope**: deterministic seeded operating-characteristics surface for robust-mixture prior review

## 1. Purpose

This document defines the acceptance criteria for the published robust-mixture
prior operating-characteristics slice on the Bayesian design surface.

This slice answers one narrow question:

- given an exact design spec, a published prior-sensitivity campaign, and a
  published robust-mixture policy, what review-state frequencies do we observe
  under deterministic seeded scenario simulation?

## 2. Public entrypoints

The accepted Python entrypoints for this slice are:

- `nextstat.bayes_design.simulate_beta_binomial_robust_mixture_prior_operating_characteristics(spec_or_path, campaign_or_path, policy_or_path) -> dict`
- `nextstat.bayes_design.simulate_normal_normal_robust_mixture_prior_operating_characteristics(spec_or_path, campaign_or_path, policy_or_path) -> dict`

These entrypoints are accepted only as explicit seeded execution paths. They
must consume:

- a committed `*_design_v0` artifact
- a committed family-specific prior-sensitivity campaign artifact
- a committed `nextstat_bayesian_robust_mixture_prior_policy_v0` artifact

They must remain explicit about what they are:

- deterministic pathwise review-state simulation
- derived from the exact family-specific analysis path
- not a robust-mixture posterior engine

## 3. Published contracts

The accepted schema contracts are:

- `nextstat_bayesian_robust_mixture_prior_policy_v0`
- `nextstat_bayesian_robust_mixture_prior_review_v0`
- `nextstat_bayesian_robust_mixture_prior_operating_characteristics_v0`

Published schema locations:

- `docs/schemas/pharma/bayesian_robust_mixture_prior_policy_v0.schema.json`
- `docs/schemas/pharma/bayesian_robust_mixture_prior_review_v0.schema.json`
- `docs/schemas/pharma/bayesian_robust_mixture_prior_operating_characteristics_v0.schema.json`

Published example locations:

- `docs/specs/pharma/bayesian_robust_mixture_prior_policy_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_policy_normal_normal_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_review_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_review_normal_normal_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_normal_normal_v0.example.json`

CLI schema publication must include:

- `nextstat config schema --name bayesian_robust_mixture_prior_policy_v0`
- `nextstat config schema --name bayesian_robust_mixture_prior_review_v0`
- `nextstat config schema --name bayesian_robust_mixture_prior_operating_characteristics_v0`

## 4. Required artifact behavior

The operating-characteristics artifact is acceptable only if it contains:

- explicit lineage:
  - `source_design_schema_version`
  - `source_campaign_schema_version`
  - `source_policy_schema_version`
  - `derived_review_schema_version`
- deterministic simulation metadata:
  - `n_replicates`
  - `seed`
- scenario summaries with terminal review-state rates:
  - `retain_rate`
  - `taper_rate`
  - `fallback_to_weak_rate`
  - `mixture_eligible_rate`
  - `decision_instability_rate`
  - `high_conflict_rate`
- look-level review probabilities and state probabilities:
  - `review_probability`
  - `retain_probability`
  - `taper_probability`
  - `fallback_to_weak_probability`

## 5. Determinism and execution rule

This slice is accepted only if:

- identical spec + campaign + policy inputs produce identical JSON outputs
- the published wrappers accept Python `dict`, JSON string, and filesystem path
  ingress
- the simulation path remains explicit and deterministic
- no frozen-report helper contract is widened or silently repurposed

## 6. Verification gates

The minimum required checks are:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q \
  tests/python/test_bayes_design_module_api.py \
  tests/python/test_bayes_design_contract.py \
  tests/python/test_bayes_design_schema_smoke.py \
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

- the accepted schema name
- canonical example locations
- accepted Python entrypoints
- the deterministic seeded execution rule
- the fact that this slice summarizes policy-review operating characteristics,
  not robust-mixture posterior updates

## 8. Benchmark note

This slice does **not** introduce a separate `nextstat-bench` runtime gate.
The canonical `nextstat-bench` benchmark requirement remains scoped to published
design-report bundle runtime behavior and public performance wording.

## 9. Out of scope for v0 acceptance

The following are not part of this acceptance contract:

- robust-mixture posterior updates inside the exact engine
- dynamic informative-weight adaptation beyond the published policy/review rules
- report, appendix, or validation-pack embedding of robust-mixture operating
  characteristics
- historical-control borrowing posterior engines
- non-conjugate fallback engines
