---
title: "Bayesian Trial Design Artifacts"
status: stable
---

# Bayesian Trial Design Artifacts

This repo now defines the accepted stable-surface contracts for the current
FDA-aligned Bayesian trial design line:

- exact `beta-binomial` two-arm designs for binary endpoints
- exact `normal-normal` two-arm designs for continuous endpoints with known SDs

Current schema surfaces:

- `beta_binomial_design_v0` — design-spec contract
- `beta_binomial_design_analysis_v0` — named-look analysis result
- `beta_binomial_operating_characteristics_v0` — seeded OC simulation result
- `beta_binomial_posterior_predictive_v0` — conditional forecast from current data
- `beta_binomial_prior_sensitivity_campaign_v0` — explicit prior-variant input
- `beta_binomial_prior_sensitivity_report_v0` — prior comparison report
- `beta_binomial_design_report_v0` — consolidated stable report artifact
- `bayesian_design_regulatory_appendix_v0` — frozen regulatory appendix JSON contract
- `bayesian_design_report_bundle_v0` — deterministic bundle-summary contract for frozen design reports
- `bayesian_prior_conflict_diagnostic_v0` — frozen prior-data conflict summary derived from a design report
- `bayesian_historical_control_borrowing_policy_v0` — historical-control borrowing policy contract for frozen exact-slice review
- `bayesian_historical_control_borrowing_review_v0` — deterministic policy-review artifact derived from a frozen design report and a published borrowing policy
- `bayesian_historical_control_borrowing_operating_characteristics_v0` — deterministic seeded operating-characteristics artifact for published historical-control borrowing policies
- `bayesian_robust_mixture_prior_policy_v0` — robust-mixture prior policy contract for frozen exact-slice review
- `bayesian_robust_mixture_prior_review_v0` — deterministic policy-review artifact derived from a frozen design report and a published robust-mixture prior policy
- `bayesian_robust_mixture_prior_operating_characteristics_v0` — deterministic seeded operating-characteristics artifact for published robust-mixture prior policies
- `normal_normal_design_v0` — design-spec contract
- `normal_normal_design_analysis_v0` — named-look analysis result
- `normal_normal_operating_characteristics_v0` — seeded OC simulation result
- `normal_normal_posterior_predictive_v0` — conditional forecast from current data
- `normal_normal_prior_sensitivity_campaign_v0` — explicit prior-variant input
- `normal_normal_prior_sensitivity_report_v0` — prior comparison report
- `normal_normal_design_report_v0` — consolidated stable report artifact

All of these schemas are available from the CLI:

```bash
nextstat config schema --name beta_binomial_design_v0
nextstat config schema --name beta_binomial_design_analysis_v0
nextstat config schema --name beta_binomial_operating_characteristics_v0
nextstat config schema --name beta_binomial_posterior_predictive_v0
nextstat config schema --name beta_binomial_prior_sensitivity_campaign_v0
nextstat config schema --name beta_binomial_prior_sensitivity_report_v0
nextstat config schema --name beta_binomial_design_report_v0
nextstat config schema --name bayesian_design_regulatory_appendix_v0
nextstat config schema --name bayesian_design_report_bundle_v0
nextstat config schema --name bayesian_prior_conflict_diagnostic_v0
nextstat config schema --name bayesian_historical_control_borrowing_policy_v0
nextstat config schema --name bayesian_historical_control_borrowing_review_v0
nextstat config schema --name bayesian_historical_control_borrowing_operating_characteristics_v0
nextstat config schema --name bayesian_robust_mixture_prior_policy_v0
nextstat config schema --name bayesian_robust_mixture_prior_review_v0
nextstat config schema --name bayesian_robust_mixture_prior_operating_characteristics_v0
nextstat config schema --name normal_normal_design_v0
nextstat config schema --name normal_normal_design_analysis_v0
nextstat config schema --name normal_normal_operating_characteristics_v0
nextstat config schema --name normal_normal_posterior_predictive_v0
nextstat config schema --name normal_normal_prior_sensitivity_campaign_v0
nextstat config schema --name normal_normal_prior_sensitivity_report_v0
nextstat config schema --name normal_normal_design_report_v0
```

Example payloads live in:

- `docs/specs/pharma/beta_binomial_design_v0.example.json`
- `docs/specs/pharma/beta_binomial_design_analysis_v0.example.json`
- `docs/specs/pharma/beta_binomial_operating_characteristics_v0.example.json`
- `docs/specs/pharma/beta_binomial_posterior_predictive_v0.example.json`
- `docs/specs/pharma/beta_binomial_prior_sensitivity_campaign_v0.example.json`
- `docs/specs/pharma/beta_binomial_prior_sensitivity_report_v0.example.json`
- `docs/specs/pharma/beta_binomial_design_report_v0.example.json`
- `docs/specs/pharma/beta_binomial_design_report_v0.example.md`
- `docs/specs/pharma/beta_binomial_regulatory_appendix_v0.example.json`
- `docs/specs/pharma/beta_binomial_regulatory_appendix_v0.example.md`
- `docs/specs/pharma/beta_binomial_design_report_bundle_v0.example.json`
- `docs/specs/pharma/beta_binomial_prior_conflict_diagnostic_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_policy_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_review_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_policy_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_review_beta_binomial_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_beta_binomial_v0.example.json`
- `docs/specs/pharma/normal_normal_design_v0.example.json`
- `docs/specs/pharma/normal_normal_design_analysis_v0.example.json`
- `docs/specs/pharma/normal_normal_operating_characteristics_v0.example.json`
- `docs/specs/pharma/normal_normal_posterior_predictive_v0.example.json`
- `docs/specs/pharma/normal_normal_prior_sensitivity_campaign_v0.example.json`
- `docs/specs/pharma/normal_normal_prior_sensitivity_report_v0.example.json`
- `docs/specs/pharma/normal_normal_design_report_v0.example.json`
- `docs/specs/pharma/normal_normal_design_report_v0.example.md`
- `docs/specs/pharma/normal_normal_regulatory_appendix_v0.example.json`
- `docs/specs/pharma/normal_normal_regulatory_appendix_v0.example.md`
- `docs/specs/pharma/normal_normal_design_report_bundle_v0.example.json`
- `docs/specs/pharma/normal_normal_prior_conflict_diagnostic_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_policy_normal_normal_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_review_normal_normal_v0.example.json`
- `docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_normal_normal_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_policy_normal_normal_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_review_normal_normal_v0.example.json`
- `docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_normal_normal_v0.example.json`

Formal acceptance gates for this stable surface are defined in:

- `docs/specs/pharma/bayesian_design_report_acceptance_v0.md`
- `docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md`
- `docs/specs/pharma/bayesian_historical_control_borrowing_review_acceptance_v0.md`
- `docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_acceptance_v0.md`
- `docs/specs/pharma/bayesian_robust_mixture_prior_review_acceptance_v0.md`
- `docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md`
- `docs/benchmarks/bayesian-design-report-packaging-runtime-gate.md`
- `docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md`

Committed benchmark evidence contract for runtime-affecting bundle changes:

- `docs/schemas/benchmarks/bayesian_design_report_bundle_benchmark_result_v1.schema.json`
- `docs/specs/pharma/bayesian_design_report_bundle_benchmark_result_v1.example.json`
- `scripts/bayesian_design_report_bundle_performance_budget_v1.json`
- `docs/schemas/benchmarks/nextstat_bayesian_design_report_bundle_performance_budget_v1.schema.json`
- canonical promotion host: `nextstat-bench`

Python surface:

- `nextstat.bayes_design.analyze_beta_binomial_design(...)`
- `nextstat.bayes_design.forecast_beta_binomial_design(...)`
- `nextstat.bayes_design.analyze_beta_binomial_prior_sensitivity(...)`
- `nextstat.bayes_design.simulate_beta_binomial_design(...)`
- `nextstat.bayes_design.build_beta_binomial_design_report(...)`
- `nextstat.bayes_design.build_beta_binomial_prior_conflict_diagnostic(...)`
- `nextstat.bayes_design.build_beta_binomial_historical_control_borrowing_review(...)`
- `nextstat.bayes_design.simulate_beta_binomial_historical_control_borrowing_operating_characteristics(...)`
- `nextstat.bayes_design.build_beta_binomial_robust_mixture_prior_review(...)`
- `nextstat.bayes_design.simulate_beta_binomial_robust_mixture_prior_operating_characteristics(...)`
- `nextstat.bayes_design.build_beta_binomial_regulatory_appendix(...)`
- `nextstat.bayes_design.render_bayesian_regulatory_appendix_markdown(...)`
- `nextstat.bayes_design.write_bayesian_regulatory_appendix_pdf(...)`
- `nextstat.bayes_design.render_beta_binomial_design_report(...)`
- `nextstat.bayes_design.write_beta_binomial_design_report_bundle(...)`
- `nextstat.bayes_design.analyze_normal_normal_design(...)`
- `nextstat.bayes_design.forecast_normal_normal_design(...)`
- `nextstat.bayes_design.analyze_normal_normal_prior_sensitivity(...)`
- `nextstat.bayes_design.simulate_normal_normal_design(...)`
- `nextstat.bayes_design.build_normal_normal_design_report(...)`
- `nextstat.bayes_design.build_normal_normal_prior_conflict_diagnostic(...)`
- `nextstat.bayes_design.build_normal_normal_historical_control_borrowing_review(...)`
- `nextstat.bayes_design.simulate_normal_normal_historical_control_borrowing_operating_characteristics(...)`
- `nextstat.bayes_design.build_normal_normal_robust_mixture_prior_review(...)`
- `nextstat.bayes_design.simulate_normal_normal_robust_mixture_prior_operating_characteristics(...)`
- `nextstat.bayes_design.build_normal_normal_regulatory_appendix(...)`
- `nextstat.bayes_design.render_normal_normal_design_report(...)`
- `nextstat.bayes_design.write_normal_normal_design_report_bundle(...)`

Regulatory appendix surface:

- family-specific appendix builders consume frozen `*_design_report_v0` artifacts only
- appendix contract is `nextstat_bayesian_design_regulatory_appendix_v0`
- appendix sections are fixed and explicit: design summary, prior specification, decision rules, current analysis, operating characteristics, posterior predictive, prior sensitivity, provenance
- appendix generation is pure frozen-report post-processing and must not re-run analysis, simulation, posterior-predictive, or prior-sensitivity computation

Appendix render surface:

- `render_bayesian_regulatory_appendix_markdown(...)` consumes only the frozen
  `nextstat_bayesian_design_regulatory_appendix_v0` artifact
- `write_bayesian_regulatory_appendix_pdf(...)` consumes only the frozen
  `nextstat_bayesian_design_regulatory_appendix_v0` artifact
- repeated renders over identical frozen appendix inputs are deterministic
- appendix render helpers must not call back into report builders, analysis, or
  simulation paths

Report packaging surface:

- family-specific bundle writers consume frozen `*_design_report_v0` artifacts only
- packaging uses `nextstat.audit.write_bundle(..., deterministic=True)` as the provenance baseline
- bundle summary contract is `nextstat_bayesian_design_report_bundle_v0`
- bundle inventory includes `meta.json`, `manifest.json`, the frozen report JSON copy, deterministic Markdown rendering, and extracted section artifacts
- runtime-affecting packaging changes are benchmark-gated on `nextstat-bench` through `docs/benchmarks/bayesian-design-report-packaging-runtime-gate.md`

Prior conflict diagnostic surface:

- family-specific prior conflict builders consume frozen `*_design_report_v0` artifacts only
- diagnostic contract is `nextstat_bayesian_prior_conflict_diagnostic_v0`
- the diagnostic is campaign-based post-processing over the frozen report’s explicit prior-sensitivity block
- conflict severity is transparent through published `thresholds`, `metrics`, `rationale`, and `variant_summaries`
- prior conflict builders must not call back into analysis, simulation, forecast, report, appendix, or bundle paths

Historical-control borrowing review surface:

- family-specific borrowing-review builders consume a frozen `*_design_report_v0` artifact plus a published `nextstat_bayesian_historical_control_borrowing_policy_v0` artifact
- policy-review contract is `nextstat_bayesian_historical_control_borrowing_review_v0`
- the review is deterministic frozen-report post-processing; it is not a borrowed posterior engine
- the review exposes explicit `gating`, `diagnostics`, `historical_sources`, and `rationale`
- borrowing-review builders must not call back into analysis, simulation, forecast, report, appendix, bundle, or public prior-conflict wrappers

Historical-control borrowing operating-characteristics surface:

- family-specific extension-OC wrappers consume a versioned design spec, a versioned prior-sensitivity campaign, and a published `nextstat_bayesian_historical_control_borrowing_policy_v0` artifact
- OC contract is `nextstat_bayesian_historical_control_borrowing_operating_characteristics_v0`
- this is a seeded execution surface, not frozen-report post-processing
- the artifact exposes scenario-level terminal state rates plus look-level review probabilities and effective-borrowing summaries
- committed examples intentionally use a compact seeded demo budget to keep the release gate tractable

Robust-mixture prior review surface:

- family-specific robust-mixture-review builders consume a frozen `*_design_report_v0` artifact plus a published `nextstat_bayesian_robust_mixture_prior_policy_v0` artifact
- policy-review contract is `nextstat_bayesian_robust_mixture_prior_review_v0`
- the review is deterministic frozen-report post-processing; it is not a robust-mixture posterior engine
- the review exposes explicit `effective_component_weights`, `gating`, `diagnostics`, and `rationale`
- robust-mixture-review builders must not call back into analysis, simulation, forecast, report, appendix, bundle, or public prior-conflict wrappers

Robust-mixture prior operating-characteristics surface:

- family-specific extension-OC wrappers consume a versioned design spec, a versioned prior-sensitivity campaign, and a published `nextstat_bayesian_robust_mixture_prior_policy_v0` artifact
- OC contract is `nextstat_bayesian_robust_mixture_prior_operating_characteristics_v0`
- this is a seeded execution surface, not frozen-report post-processing
- the artifact exposes scenario-level terminal state rates plus look-level review probabilities and informative-weight summaries
- committed examples intentionally use a compact seeded demo budget to keep the release gate tractable

Validation-pack integration surface:

- `validation-pack/render_validation_pack.sh --bayesian-design-report PATH` consumes a frozen `*_design_report_v0` artifact only
- the optional pack artifacts are fixed:
  - `bayesian_design_report.json`
  - `bayesian_design_regulatory_appendix.json`
  - `bayesian_design_regulatory_appendix.md`
  - `bayesian_design_regulatory_appendix.pdf` unless `--json-only` is supplied
  - `bayesian_design_regulatory_appendix_v0.schema.json`
  - one family-specific report schema copy: `beta_binomial_design_report_v0.schema.json` or `normal_normal_design_report_v0.schema.json`
- appendix assembly uses the published family-specific `build_*_regulatory_appendix(...)` helpers only
- Markdown/PDF appendix rendering uses the published
  `render_bayesian_regulatory_appendix_markdown(...)` and
  `write_bayesian_regulatory_appendix_pdf(...)` helpers only
- validation-pack integration is deterministic and must not re-run analysis, simulation, report generation, or bundle packaging from partial inputs

Current v0 constraints:

- Two-arm parallel-group `beta-binomial` only
- Two-arm parallel-group `normal-normal` only
- Conjugate Beta and Normal priors only on the exact execution path
- Robust-mixture artifacts are policy/review only and do not modify exact engine outputs
- `normal-normal` v0 assumes known arm-level outcome SDs
- Common `treatment_effect_margin` across success and futility rules
- Deterministic seeded simulation with explicit named scenarios

Not yet covered by these artifacts:

- actual historical borrowing posterior engines / actual robust-mixture posterior engines
- CID / BSA-specific appendix templates and protocol-language templates

## PR and Release Gates

Any PR that changes the public Bayesian design stable surface should treat the following as required merge gates:

- Keep schemas, examples, and `docs/references/bayesian-trial-design-artifacts.md` in sync.
- Preserve backward-compatible ingress for the published report, appendix, and bundle wrappers.
- Preserve backward-compatible ingress for the published prior conflict diagnostic wrappers.
- Preserve backward-compatible ingress for the published borrowing-review wrappers.
- Preserve backward-compatible ingress for the published robust-mixture-review wrappers.
- Preserve backward-compatible ingress for the published borrowing extension operating-characteristics wrappers.
- Preserve backward-compatible ingress for the published robust-mixture extension operating-characteristics wrappers.
- Preserve seeded determinism for the published borrowing and robust-mixture extension operating-characteristics wrappers.
- Preserve backward-compatible ingress for the published appendix render wrappers.
- Do not introduce hidden execution into frozen Bayesian report render/bundle paths.
- Preserve backward-compatible ingress for the published appendix wrappers.
- Do not introduce hidden execution into frozen Bayesian appendix paths.
- Do not introduce hidden execution into frozen Bayesian prior conflict diagnostic paths.
- Do not introduce hidden execution into frozen Bayesian historical-control borrowing review paths.
- Do not introduce hidden execution into frozen Bayesian robust-mixture prior review paths.
- Do not introduce hidden execution into frozen Bayesian appendix render paths.
- Preserve backward-compatible behavior for `validation-pack/render_validation_pack.sh --bayesian-design-report PATH`.
- Do not introduce hidden execution into frozen Bayesian validation-pack appendix paths.
- Keep `nextstat-bench` packaging evidence attached to runtime-affecting packaging changes.

Recommended targeted checks:

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

python3 scripts/docs/terminology_lint.py --check
```

If CLI schema publication changed:

```bash
cargo test -p ns-cli config_schema_can_emit_beta_binomial_design_schemas --test cli_config_schema
```

If packaging runtime behavior or public performance wording changed, run the canonical `nextstat-bench` gate:

```bash
ssh nextstat-bench
cd /path/to/nextstat.io
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
  scripts/benchmarks/bench_bayesian_design_report_bundle.py \
  --deterministic \
  --out bench_results/bayesian_design_report_bundle/summary.json
```

Maintainer release gate for Bayesian-design-affecting releases:

- Confirm the published stable claim stays within the current report/prior-conflict/borrowing-review/robust-mixture-review/appendix/render/bundle/validation-pack subset.
- Confirm current `nextstat-bench` packaging evidence is linked when runtime-affecting behavior changed.
- Confirm acceptance docs, release checklist, runtime gate doc, and artifact/reference page stay aligned.
