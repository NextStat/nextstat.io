---
title: "Python API Reference (nextstat)"
status: stable
---

# Python API Reference (nextstat)

This page documents the public Python surface exported by `nextstat`.

Notes:
- The compiled extension is `nextstat._core` (PyO3/maturin).
- Convenience wrappers and optional modules live under `nextstat.*` (loaded on first access).
- Type stubs for the native extension (including overloads) are in `bindings/ns-py/python/nextstat/_core.pyi`.
- Installation, optional extras, and wheel build notes: `docs/references/python-packaging.md`.
- For the end-to-end HEP GVM workflow, start with `docs/tutorials/hep-gvm-measurement-combinations.md`.

## Lazy submodules

- `nextstat.hep` — HEP-specific helpers that are intentionally not part of the top-level stable API surface.
- `nextstat.bayes` — Bayesian sampling helpers (ArviZ integration).
- `nextstat.bayes_design` — schema-first Bayesian trial design helpers for exact beta-binomial and normal-normal slices, plus frozen policy-review artifacts layered on top of committed design reports.
- `nextstat.ads` — ads-native observation, response, and variance-reduction helpers for overdispersed conversion rates, delay correction, CUPED/CURE adjustment, and spend curves.

### `nextstat.bayes_design`

- The public contract is additive and schema-first: each design family has a versioned spec, analysis result, and operating-characteristics artifact.
- Wrappers accept either a Python `dict`, a JSON string, or a filesystem path.
- `nextstat.bayes_design.analyze_beta_binomial_design(spec_or_path, observed_or_path) -> dict`
- `nextstat.bayes_design.forecast_beta_binomial_design(spec_or_path, observed_or_path) -> dict`
- `nextstat.bayes_design.analyze_beta_binomial_prior_sensitivity(spec_or_path, observed_or_path, campaign_or_path) -> dict`
- `nextstat.bayes_design.simulate_beta_binomial_design(spec_or_path) -> dict`
- `nextstat.bayes_design.build_beta_binomial_design_report(spec_or_path, observed_or_path, campaign_or_path) -> dict`
- `nextstat.bayes_design.build_beta_binomial_regulatory_appendix(report_or_path) -> dict` — build the frozen regulatory appendix JSON artifact from a committed beta-binomial design report. Returns `nextstat_bayesian_design_regulatory_appendix_v0`.
- `nextstat.bayes_design.build_beta_binomial_prior_conflict_diagnostic(report_or_path) -> dict` — build the frozen campaign-based prior conflict diagnostic from a committed beta-binomial design report. Returns `nextstat_bayesian_prior_conflict_diagnostic_v0`.
- `nextstat.bayes_design.build_beta_binomial_historical_control_borrowing_review(report_or_path, policy_or_path) -> dict` — build the frozen historical-control borrowing policy review from a committed beta-binomial design report and a committed borrowing policy. Returns `nextstat_bayesian_historical_control_borrowing_review_v0`.
- `nextstat.bayes_design.simulate_beta_binomial_historical_control_borrowing_operating_characteristics(spec_or_path, campaign_or_path, policy_or_path) -> dict` — compute seeded historical-control borrowing policy-review operating characteristics from a versioned beta-binomial design spec, prior-sensitivity campaign, and borrowing policy. Returns `nextstat_bayesian_historical_control_borrowing_operating_characteristics_v0`.
- `nextstat.bayes_design.build_beta_binomial_robust_mixture_prior_review(report_or_path, policy_or_path) -> dict` — build the frozen robust-mixture prior policy review from a committed beta-binomial design report and a committed robust-mixture policy. Returns `nextstat_bayesian_robust_mixture_prior_review_v0`.
- `nextstat.bayes_design.simulate_beta_binomial_robust_mixture_prior_operating_characteristics(spec_or_path, campaign_or_path, policy_or_path) -> dict` — compute seeded robust-mixture prior policy-review operating characteristics from a versioned beta-binomial design spec, prior-sensitivity campaign, and robust-mixture policy. Returns `nextstat_bayesian_robust_mixture_prior_operating_characteristics_v0`.
- `nextstat.bayes_design.render_bayesian_regulatory_appendix_markdown(appendix_or_path) -> str` — render deterministic Markdown from a committed `nextstat_bayesian_design_regulatory_appendix_v0` artifact.
- `nextstat.bayes_design.write_bayesian_regulatory_appendix_pdf(pdf_path, appendix_or_path) -> None` — write a deterministic PDF from a committed `nextstat_bayesian_design_regulatory_appendix_v0` artifact.
- `nextstat.bayes_design.render_beta_binomial_design_report(report_or_path) -> str`
- `nextstat.bayes_design.write_beta_binomial_design_report_bundle(bundle_dir, report_or_path) -> dict` — write a deterministic run bundle from a frozen beta-binomial design report. Returns `nextstat_bayesian_design_report_bundle_v0`.
- `nextstat.bayes_design.analyze_normal_normal_design(spec_or_path, observed_or_path) -> dict`
- `nextstat.bayes_design.forecast_normal_normal_design(spec_or_path, observed_or_path) -> dict`
- `nextstat.bayes_design.analyze_normal_normal_prior_sensitivity(spec_or_path, observed_or_path, campaign_or_path) -> dict`
- `nextstat.bayes_design.simulate_normal_normal_design(spec_or_path) -> dict`
- `nextstat.bayes_design.build_normal_normal_design_report(spec_or_path, observed_or_path, campaign_or_path) -> dict`
- `nextstat.bayes_design.build_normal_normal_regulatory_appendix(report_or_path) -> dict` — build the frozen regulatory appendix JSON artifact from a committed normal-normal design report. Returns `nextstat_bayesian_design_regulatory_appendix_v0`.
- `nextstat.bayes_design.build_normal_normal_prior_conflict_diagnostic(report_or_path) -> dict` — build the frozen campaign-based prior conflict diagnostic from a committed normal-normal design report. Returns `nextstat_bayesian_prior_conflict_diagnostic_v0`.
- `nextstat.bayes_design.build_normal_normal_historical_control_borrowing_review(report_or_path, policy_or_path) -> dict` — build the frozen historical-control borrowing policy review from a committed normal-normal design report and a committed borrowing policy. Returns `nextstat_bayesian_historical_control_borrowing_review_v0`.
- `nextstat.bayes_design.simulate_normal_normal_historical_control_borrowing_operating_characteristics(spec_or_path, campaign_or_path, policy_or_path) -> dict` — compute seeded historical-control borrowing policy-review operating characteristics from a versioned normal-normal design spec, prior-sensitivity campaign, and borrowing policy. Returns `nextstat_bayesian_historical_control_borrowing_operating_characteristics_v0`.
- `nextstat.bayes_design.build_normal_normal_robust_mixture_prior_review(report_or_path, policy_or_path) -> dict` — build the frozen robust-mixture prior policy review from a committed normal-normal design report and a committed robust-mixture policy. Returns `nextstat_bayesian_robust_mixture_prior_review_v0`.
- `nextstat.bayes_design.simulate_normal_normal_robust_mixture_prior_operating_characteristics(spec_or_path, campaign_or_path, policy_or_path) -> dict` — compute seeded robust-mixture prior policy-review operating characteristics from a versioned normal-normal design spec, prior-sensitivity campaign, and robust-mixture policy. Returns `nextstat_bayesian_robust_mixture_prior_operating_characteristics_v0`.
- `nextstat.bayes_design.render_normal_normal_design_report(report_or_path) -> str`
- `nextstat.bayes_design.write_normal_normal_design_report_bundle(bundle_dir, report_or_path) -> dict` — write a deterministic run bundle from a frozen normal-normal design report. Returns `nextstat_bayesian_design_report_bundle_v0`.
- See `docs/references/bayesian-trial-design-artifacts.md` for schema names and example payloads.

### `nextstat.hep`

- This namespace is the Python entry point for the stable-first scalar measurement-combination workflow, plus the wider research-grade reporting stack.
- The recommended default is `solver="auto"`, which keeps the perturbative path on the fast path and falls back safely to the paper-faithful numerical reference when needed.
- For calibration-style wrappers, deterministic toy generation uses the paper-faithful `numerical-paper` reference whenever the requested solver is `numerical-paper`, `analytic-perturbative`, or `auto`.

- `nextstat.hep.build_measurement_combination_spec(measurements_table, stat_covariance_table, *, poi="mu", systematics_table=None, correlations_table=None) -> dict` — stable-first tabular ingress helper. Accepts raw CSV/TSV text or filesystem paths and returns the canonical `nextstat_measurement_combination_v0` JSON payload expected by the stable fit/calibration wrappers.
- `nextstat.hep.build_measurement_combination_spec_from_manifest(manifest_path) -> dict` — stable-first manifest ingress helper. Accepts a YAML/JSON manifest path pointing at the same table bundle and returns the canonical `nextstat_measurement_combination_v0` JSON payload expected by the stable fit/calibration wrappers.
- The tabular helper expects:
  - `measurements_table`: `name,value`
  - `stat_covariance_table`: named square matrix with row/column measurement names
  - optional `systematics_table`: `systematic,measurement,magnitude,error_on_error,aux_mean`
  - optional `correlations_table`: `systematic,row_measurement,col_measurement,corr`
- The manifest helper expects a `nextstat_measurement_combination_manifest_v0` file with `poi`, `measurements_table`, `stat_covariance_table`, and optional `systematics_table` / `correlations_table` entries.
- If `correlations_table` is omitted, each systematic defaults to identity correlation.
- `nextstat.hep.combine_measurements(spec_or_path, *, ci_level=0.68, solver="auto") -> dict` — stable-first scalar measurement combination wrapper. Accepts either a Python `dict` matching `nextstat_measurement_combination_v0` or a path to a JSON spec and returns a plain `dict` matching the CLI result schema.
- `nextstat.hep.combine_measurements(...)` uses the same Rust core engine as `nextstat combine-measurements`.
- `solver="auto"` is the default stable path: it tries the Eq. (21)-(28) / Appendix B perturbative approximation first and falls back to `numerical-paper` outside the validity radius. `solver="numerical"` keeps the existing reduced-basis numerical GVM path, `solver="numerical-paper"` runs the paper-faithful original-`theta_s^i` numerical path, and `solver="analytic-perturbative"` forces the perturbative approximation.
- When runtime dispatch differs from the requested solver, the returned `dict` records it through `diagnostics.requested_solver` and `diagnostics.effective_solver`.
- `nextstat.hep.calibrate_measurements(spec_or_path, *, ci_level=0.68, solver="auto", n_toys=128, seed=42) -> dict` — stable-first toy-calibration wrapper. Returns a calibration report matching the CLI schema from `nextstat combine-measurements-calibrate`.
- `nextstat.hep.calibrate_measurements_study(spec_or_path, *, ci_level=0.68, solver="auto", n_toys=128, seeds=[42, 43, 44]) -> dict` — stable-first repeated-seed calibration wrapper. Returns a study report matching the CLI schema from `nextstat combine-measurements-calibrate-study`, including `per_seed` summaries and aggregate stability diagnostics.
- `nextstat.hep.study_measurement_combination_scenarios(spec_or_path, scenarios_or_path, *, ci_level=0.68, solver="auto") -> dict` — research-grade scenario-study wrapper. Returns a baseline-relative comparison report matching the CLI schema from `nextstat combine-measurements-scenario-study`.
- `nextstat.hep.calibrate_measurement_combination_scenarios(spec_or_path, scenarios_or_path, *, ci_level=0.68, solver="auto", n_toys=128, seeds=[42, 43, 44]) -> dict` — research-grade calibration-campaign wrapper. Returns one artifact matching the CLI schema from `nextstat combine-measurements-calibration-campaign`, combining scenario-level fits with repeated-seed calibration summaries per scenario.
- `nextstat.hep.compare_measurement_combination_scenario_study_solvers(spec_or_path, scenarios_or_path, *, ci_level=0.68, lhs_solver="numerical-paper", rhs_solver="analytic-perturbative") -> dict` — research-grade solver-parity wrapper for scenario studies. Returns a direct parity artifact matching `nextstat combine-measurements-solver-parity-scenario-study`.
- `nextstat.hep.compare_measurement_combination_calibration_campaign_solvers(spec_or_path, scenarios_or_path, *, ci_level=0.68, lhs_solver="numerical-paper", rhs_solver="analytic-perturbative", n_toys=128, seeds=[42, 43, 44]) -> dict` — research-grade solver-parity wrapper for repeated-seed calibration campaigns. Returns a direct parity artifact matching `nextstat combine-measurements-solver-parity-calibration-campaign`.
- `nextstat.hep.compare_measurement_combination_scenario_study_solver_reports(lhs_report_or_path, rhs_report_or_path, *, lhs_solver="numerical-paper", rhs_solver="analytic-perturbative") -> dict` — research-grade cached solver-parity wrapper. Reads two precomputed scenario-study reports and returns the same parity schema as `nextstat combine-measurements-solver-parity-scenario-study-from-reports`.
- `nextstat.hep.compare_measurement_combination_calibration_campaign_solver_reports(lhs_report_or_path, rhs_report_or_path, *, lhs_solver="numerical-paper", rhs_solver="analytic-perturbative") -> dict` — research-grade cached solver-parity wrapper for repeated-seed campaigns. Reads two precomputed calibration-campaign reports and returns the same parity schema as `nextstat combine-measurements-solver-parity-calibration-campaign-from-reports`.
- `nextstat.hep.render_measurement_combination_scenario_study_solver_parity(report_or_path) -> str` — Markdown renderer for the scenario-study solver-parity artifact.
- `nextstat.hep.render_measurement_combination_calibration_campaign_solver_parity(report_or_path) -> str` — Markdown renderer for the calibration-campaign solver-parity artifact.
- `nextstat.hep.summarize_measurement_combination_scenario_study_solver_parity(report_or_path) -> dict` — research-grade post-processing wrapper. Reads an existing scenario-study solver-parity artifact and returns the compact digest schema from `nextstat combine-measurements-solver-parity-scenario-study-summarize`.
- `nextstat.hep.render_measurement_combination_scenario_study_solver_parity_summary(summary_or_path) -> str` — Markdown renderer for the scenario-study solver-parity digest emitted by `nextstat combine-measurements-solver-parity-scenario-study-summarize --format markdown`.
- `nextstat.hep.summarize_measurement_combination_calibration_campaign_solver_parity(report_or_path) -> dict` — research-grade post-processing wrapper. Reads an existing calibration-campaign solver-parity artifact and returns the compact digest schema from `nextstat combine-measurements-solver-parity-calibration-campaign-summarize`.
- `nextstat.hep.render_measurement_combination_calibration_campaign_solver_parity_summary(summary_or_path) -> str` — Markdown renderer for the calibration-campaign solver-parity digest emitted by `nextstat combine-measurements-solver-parity-calibration-campaign-summarize --format markdown`.
- These wrappers accept the same solver contract as `nextstat.hep.combine_measurements(...)`. For calibration-style wrappers, toy generation uses the paper-faithful `numerical-paper` reference whenever the requested solver is `numerical-paper`, `analytic-perturbative`, or `auto`.
- The two `...solver_reports(...)` wrappers are pure post-processing helpers for expensive published workflows: run each solver once, persist the study/campaign JSON, then compare them later without rerunning fits or toys.
- `nextstat.hep.summarize_measurement_combination_calibration_campaign(report_or_path) -> dict` — research-grade digest wrapper. Reads an existing calibration-campaign artifact and returns the compact summary schema from `nextstat combine-measurements-calibration-campaign-summarize`.
- `nextstat.hep.render_measurement_combination_calibration_campaign_summary(summary_or_path) -> str` — research-grade Markdown renderer. Reads an existing campaign digest and returns the same human-readable note emitted by `nextstat combine-measurements-calibration-campaign-summarize --format markdown`.
- `nextstat.hep.build_measurement_combination_calibration_campaign_brief(summaries_or_paths, *, labels=None) -> dict` — research-grade comparative brief builder. Reads multiple existing campaign digests and returns a cross-artifact brief matching `nextstat combine-measurements-calibration-campaign-brief --format json`.
- `nextstat.hep.render_measurement_combination_calibration_campaign_brief(brief_or_path) -> str` — research-grade Markdown renderer for the comparative brief schema emitted by `nextstat combine-measurements-calibration-campaign-brief --format markdown`.
- `nextstat.hep.build_measurement_combination_calibration_campaign_family_report(briefs_or_paths, *, labels=None) -> dict` — research-grade family-report builder. Reads multiple existing campaign briefs and returns a cross-family report matching `nextstat combine-measurements-calibration-campaign-family-report --format json`.
- `nextstat.hep.render_measurement_combination_calibration_campaign_family_report(report_or_path) -> str` — research-grade Markdown renderer for the family-report schema emitted by `nextstat combine-measurements-calibration-campaign-family-report --format markdown`.
- `nextstat.hep.build_measurement_combination_calibration_campaign_family_matrix(report_or_path) -> dict` — research-grade dominance-matrix builder. Reads one existing family report and returns a machine-readable ranking/pairwise-comparison artifact matching `nextstat combine-measurements-calibration-campaign-family-matrix --format json`.
- `nextstat.hep.render_measurement_combination_calibration_campaign_family_matrix(matrix_or_path) -> str` — research-grade Markdown renderer for the dominance-matrix schema emitted by `nextstat combine-measurements-calibration-campaign-family-matrix --format markdown`.
- `nextstat.hep.build_measurement_combination_calibration_campaign_portfolio(matrices_or_paths, *, labels=None) -> dict` — research-grade portfolio builder. Reads multiple existing family-matrix artifacts and returns a cross-campaign comparison artifact matching `nextstat combine-measurements-calibration-campaign-portfolio --format json`.
- `nextstat.hep.render_measurement_combination_calibration_campaign_portfolio(report_or_path) -> str` — research-grade Markdown renderer for the portfolio schema emitted by `nextstat combine-measurements-calibration-campaign-portfolio --format markdown`.
- `nextstat.hep.build_measurement_combination_calibration_campaign_portfolio_stability(portfolios_or_paths, *, labels=None) -> dict` — research-grade portfolio-stability builder. Reads multiple existing portfolio artifacts and returns a cross-run stability artifact matching `nextstat combine-measurements-calibration-campaign-portfolio-stability --format json`.
- `nextstat.hep.render_measurement_combination_calibration_campaign_portfolio_stability(report_or_path) -> str` — research-grade Markdown renderer for the portfolio-stability schema emitted by `nextstat combine-measurements-calibration-campaign-portfolio-stability --format markdown`.
- The returned `dict` includes `diagnostics.bartlett`, which reports the Lawley/Bartlett correction factors used to refine profile-likelihood and GOF diagnostics in the GVM path.
- The returned `dict` also includes `diagnostics.perturbative_validity`, which reports per-systematic convergence indicators for the perturbative expansion used by the Bartlett layer.
- Existing `nextstat.workspace_combine(...)` remains the workspace-merge API; it is not affected by the measurement-combination path.

Recommended usage order:

1. `nextstat.hep.combine_measurements(...)`
2. `nextstat.hep.calibrate_measurements(...)`
3. `nextstat.hep.calibrate_measurements_study(...)`
4. `nextstat.hep.study_measurement_combination_scenarios(...)`
5. `nextstat.hep.calibrate_measurement_combination_scenarios(...)`
6. `summarize_*` / `render_*` post-processing helpers
7. `compare_*_solvers(...)` only when you need explicit numerical-vs-perturbative parity artifacts

## Simplified-likelihood support class (March 2026)

The promoted simplified-likelihood subset on the Python surface is intentionally
narrow:

- `nextstat.workspace_audit(...)` is `stable` for pyhf and simplified-likelihood inputs; simplified-likelihood returns the published `nextstat_simplified_likelihood_audit_v0` artifact and HS3 is rejected explicitly on that audit path
- for simplified-likelihood inference from Python, the promoted stable contract lives on `nextstat.tools` / `nextstat-server`: `nextstat_workspace_audit`, `nextstat_fit`, `nextstat_hypotest`, `nextstat_upper_limit`, and `nextstat_scan`
- `nextstat_discovery_asymptotic`, `nextstat_ranking`, and `nextstat_hypotest_toys` remain compatibility-tested but `research-grade` for simplified-likelihood inputs
- the future stable exporter claim is now published separately and remains narrow: `pyhf` source only, single-POI only, and `constraint_covariance_source="source_model_constraints"` for Gaussian-constrained source nuisances; derived artifacts still remain reduced-coordinate rather than source-level nuisance identities

Support and release references:

- `docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md`
- `docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md`
- `docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md`
- `docs/references/tool-api.md`

## Top-level functions

### Model construction

- `nextstat.from_pyhf(json_str) -> HistFactoryModel` — create model from pyhf JSON workspace.
- `nextstat.from_histfactory_xml(xml_path) -> HistFactoryModel` — create model from HistFactory XML.
- `nextstat.UnbinnedModel.from_config(path) -> UnbinnedModel` — compile an event-level (unbinned) model from an `unbinned_spec_v0` JSON/YAML file. Supported PDF types: `gaussian`, `crystal_ball`, `double_crystal_ball`, `exponential`, `chebyshev`, `argus`, `voigtian`, `spline`, `histogram`, `histogram_from_tree`, `kde`, `kde_from_tree`, `product`, `flow`, `conditional_flow`, `dcr_surrogate`. The `flow`, `conditional_flow`, and `dcr_surrogate` types require building with `--features neural` (ONNX Runtime). See `docs/neural-density-estimation.md` for the full workflow.
- `nextstat.unbinned.from_config(path) -> nextstat.unbinned.UnbinnedAnalysis` — high-level unbinned workflow wrapper (compile + fit/fit_toys/scan/hypotest/toys/ranking helpers).
- `nextstat.workspace_audit(json_str) -> dict` — audit a pyhf or simplified-likelihood workspace. Pyhf inputs return compatibility counts and unsupported-feature warnings; simplified-likelihood inputs return the published `nextstat_simplified_likelihood_audit_v0` artifact. HS3 is rejected explicitly.
- `nextstat.apply_patchset(workspace_json, patchset_json, *, patch_name=None) -> str` — apply a pyhf patchset.
- `nextstat.workspace_combine(ws1_json, ws2_json, *, join="none") -> str` — combine two pyhf workspace JSON strings. Join modes: `"none"` (error on conflict), `"outer"` (union), `"left_outer"`, `"right_outer"`.
- `nextstat.workspace_prune(ws_json, *, channels=[], samples=[], modifiers=[], measurements=[]) -> str` — remove channels, samples, modifiers, and/or measurements from a workspace.
- `nextstat.workspace_rename(ws_json, *, channels=None, samples=None, modifiers=None, measurements=None) -> str` — rename workspace elements. Each argument is an `{old: new}` dict.
- `nextstat.workspace_sorted(ws_json) -> str` — return workspace with all components in canonical (sorted) order.
- `nextstat.workspace_digest(ws_json) -> str` — compute SHA-256 digest of the canonical workspace.
- `nextstat.workspace_to_xml(ws_json, output_prefix="output") -> list[tuple[str, str]]` — export workspace to HistFactory XML. Returns `[(filename, xml_content), ...]`.
- `nextstat.simplemodel_uncorrelated(signal, bkg, bkg_uncertainty) -> str` — build workspace with uncorrelated background (shapesys). pyhf-compatible.
- `nextstat.simplemodel_correlated(signal, bkg, bkg_up, bkg_down) -> str` — build workspace with correlated background (histosys). pyhf-compatible.
- `nextstat.read_root_histogram(root_path, hist_path) -> dict` — read a TH1 histogram from a ROOT file. Returns `{name, title, bin_edges, bin_content, sumw2, underflow, overflow, underflow_sumw2, overflow_sumw2}`.
- `nextstat.histfactory_bin_edges_by_channel(xml_path) -> dict[str, list[float]]` — extract bin edges per channel from HistFactory XML.

Notes on HistFactory XML ingest (`nextstat.from_histfactory_xml`):
- `ShapeSys` histograms are treated as **relative** per-bin uncertainties and converted to absolute `sigma_abs = rel * nominal`.
- `StatError` histograms are treated as **relative** per-bin uncertainties and converted to absolute `sigma_abs = rel * nominal`.
- `StatError` follows channel `<StatErrorConfig ConstraintType="Poisson">` or `<StatErrorConfig ConstraintType="Gaussian">`:
  - `ConstraintType="Poisson"` => preserves `staterror` (per-channel, name `staterror_<channel>`) and attaches per-bin `Gamma` constraint
    metadata (non-standard extension) to `measurement.config.parameters` entries named `staterror_<channel>[i]`.
  - `ConstraintType="Gaussian"` => preserves `staterror` (per-channel, name `staterror_<channel>`) with Gaussian penalty (pyhf-style).
  - ROOT/HistFactory defaults when `<StatErrorConfig>` is omitted: `ConstraintType="Poisson"` and `RelErrorThreshold=0.05` (bins with
    relative stat error below threshold are pruned, i.e. the corresponding `staterror_<channel>[i]` is fixed at 1.0).
- Samples with `NormalizeByTheory="True"` receive a `lumi` modifier named `Lumi`, and `LumiRelErr` is surfaced via
  measurement parameter config (`auxdata=[1]`, `sigmas=[LumiRelErr]`).
- `NormFactor Val/Low/High` is surfaced via measurement parameter config (`inits` and `bounds`).

#### HS3 (HEP Statistics Serialization Standard)

- `HistFactoryModel.from_workspace(json_str) -> HistFactoryModel` — **auto-detects** pyhf vs HS3 format. If the JSON contains `"distributions"` + `"hs3_version"`, it is parsed as HS3; otherwise as pyhf.
- `HistFactoryModel.from_xml(xml_path) -> HistFactoryModel` — create model from HistFactory XML (`combination.xml` + referenced ROOT histograms).
- `HistFactoryModel.from_hs3(json_str, analysis=None, param_points=None) -> HistFactoryModel` — explicit HS3 loading with optional analysis selection and parameter point set.

```python
import json, nextstat

# Auto-detect: works with both pyhf and HS3
json_str = open("workspace-postFit_PTV.json").read()
model = nextstat.HistFactoryModel.from_workspace(json_str)

# Explicit HS3 with analysis selection
model = nextstat.HistFactoryModel.from_hs3(
    json_str,
    analysis="combPdf_obsData",        # default: first analysis
    param_points="default_values",     # default: "default_values"
)

result = nextstat.fit(model, device="cpu")
```

HS3 v0.2 support covers all modifier types produced by ROOT 6.37+: `normfactor`, `normsys`, `histosys`, `staterror`, `shapesys`, `shapefactor`, `lumi`. Unknown modifier/distribution types are silently skipped (forward-compatible).

### Fitting

- `nextstat.fit(model, *, data=None, init_pars=None, device="cpu") -> FitResult` — maximum likelihood estimation (`device="cuda"` for `HistFactoryModel`, CUDA build only).
- `nextstat.map_fit(posterior) -> FitResult` — MAP estimation for Bayesian posteriors.
- `nextstat.fit_batch(models_or_model, datasets=None) -> list[FitResult]` — batch fitting (homogeneous model lists; `datasets=` is supported for `HistFactoryModel` only).

### Hypothesis testing

- `nextstat.hypotest(poi_test, model, *, data=None, return_tail_probs=False) -> float | (float, list[float])` — asymptotic CLs. Dispatches on model type: `HistFactoryModel` (binned) or `UnbinnedModel` (unbinned q_mu).
- `nextstat.hypotest_toys(poi_test, model, *, n_toys=1000, seed=42, expected_set=False, data=None, return_tail_probs=False, return_meta=False) -> float | tuple | dict` — toy-based CLs. Dispatches on model type: `HistFactoryModel` or `UnbinnedModel` (qtilde).

### Monte Carlo / Safety

- `nextstat.fault_tree_mc_ce_is(spec, *, n_per_level=10000, elite_fraction=0.01, max_levels=20, q_max=0.99, seed=42) -> dict` — Cross-Entropy Importance Sampling for rare-event fault tree probability estimation with multi-level adaptive biasing. Handles probabilities down to ~1e-16 via soft importance function when no TOP failures are observed. Returns `{p_failure, se, ci_lower, ci_upper, n_levels, n_total_scenarios, final_proposal, coefficient_of_variation, wall_time_s}`. Current CE-IS implementation is Bernoulli-only for component failure modes.
- `nextstat.fault_tree_mc(spec, n_scenarios, seed=42, device='cpu', chunk_size=0) -> dict` — Monte Carlo fault tree simulation. `device`: `'cpu'`, `'cuda'`, `'metal'`.

### Profile likelihood

- `nextstat.profile_ci(model, fit_result, *, param_idx=None, chi2_level=3.841, tol=1e-4) -> dict | list[dict]` — profile likelihood confidence intervals for any `LogDensityModel`. If `param_idx` is given, returns a single dict; otherwise returns CI for all parameters. Each dict: `{param_idx, mle, ci_lower, ci_upper, n_evals}`.
- `nextstat.profile_scan(model, mu_values, *, data=None, device="cpu", return_params=False, return_curve=False) -> dict` — profile likelihood scan. Dispatches on model type: `HistFactoryModel` or `UnbinnedModel`. When `return_curve=True`, adds `mu_values`, `q_mu_values`, `twice_delta_nll` arrays (replaces the old `profile_curve` function).
- `nextstat.upper_limit(model, *, method="bisect", alpha=0.05, lo=0.0, hi=None, rtol=1e-4, max_iter=80, data=None) -> float | (float, list[float])` — upper limit. `method="bisect"` (observed only) or `method="root"` (observed + 5 expected bands).
- `nextstat.upper_limits(model, scan, *, alpha=0.05, data=None) -> (float, list[float])` — observed + expected limits from scan.

### Sampling

- `nextstat.sample(model, *, method="nuts", return_idata=False, out=None, out_format="json", **kwargs) -> dict | InferenceData` — **Unified sampling interface**. Dispatches to NUTS, WALNUTS, MAMS, or LAPS based on `method`. Set `return_idata=True` to get an ArviZ `InferenceData` object (requires `arviz`). Set `out="trace.json"` to save results to disk. All method-specific kwargs are forwarded to the underlying sampler.
- `nextstat.sample_nuts(model, *, n_chains=4, n_warmup=500, n_samples=1000, seed=42, max_treedepth=10, target_accept=0.8, init_strategy="random", metric="diagonal", init_jitter=0.0, init_jitter_rel=None, init_overdispersed_rel=None, stepsize_jitter=0.0, data=None) -> dict` — NUTS (No-U-Turn Sampler). Also available via `nextstat.sample(model, method="nuts", ...)`. Accepts `Posterior` as well; `data=` is not supported when sampling a `Posterior`. `init_strategy`: `"random"` (default, Stan-style Uniform(-2,2)), `"mle"` (L-BFGS mode), or `"pathfinder"` (L-BFGS mode + Hessian-derived metric as initial mass matrix for faster warmup; produces dense metric when `metric="dense"` or `metric="auto"` for dim ≤ 32). `metric`: `"diagonal"` (default, CmdStan-compatible), `"dense"` (full covariance, better for correlated posteriors), or `"auto"` (dense for dim ≤ 32, diagonal otherwise). The `sample_stats` dict in results includes `metric_type` ("diagonal"/"dense"), `mass_diag`, `inv_mass_matrix` (row-major, only for dense metric), and `n_leapfrog_warmup_total` (per-chain warmup leapfrog totals).
- `nextstat.sample_walnuts(model, *, n_chains=4, n_warmup=500, n_samples=1000, seed=42, max_treedepth=10, max_step_halvings=4, min_micro_steps=1, max_energy_error=2.0, target_accept=0.8, target_tree_depth=4.0, init_strategy="random", metric="diagonal", init_jitter=0.0, init_jitter_rel=None, init_overdispersed_rel=None, stepsize_jitter=0.0, data=None) -> dict` — WALNUTS (Window-Adaptive NUTS). Also available via `nextstat.sample(model, method="walnuts", ...)`, though the unified `sample(..., method="walnuts")` entry point remains the recommended public surface. Accepts `Posterior` as well; `data=` is not supported when sampling a `Posterior`. `init_strategy`: `"random"` (default), `"mle"`, or `"pathfinder"`; Pathfinder can seed a dense initial metric when `metric="dense"` or `metric="auto"` for dim ≤ 32. `metric`: `"diagonal"` (default), `"dense"` (full covariance), or `"auto"` (dense for dim ≤ 32, diagonal otherwise). Returns the same top-level `posterior` / `sample_stats` / `diagnostics` contract as NUTS, including `metric_type`, `mass_diag`, `inv_mass_matrix` (row-major, only for dense metric), and `sample_stats["n_leapfrog_warmup_total"]` for per-chain warmup telemetry. See `docs/references/walnuts-sampler.md` for the dedicated WALNUTS surface note.
- `nextstat.sample_mams(model, *, n_chains=4, n_warmup=3500, n_samples=1000, seed=42, target_accept=0.985, init_strategy="random", metric="diagonal", init_step_size=0.0, init_l=0.0, max_leapfrog=1024, diagonal_precond=True, eps_jitter=0.0, data=None) -> dict` — MAMS (Metropolis-Adjusted Microcanonical Sampler, arXiv:2503.01707). Also available via `nextstat.sample(model, method="mams", ...)`. Exact sampler using isokinetic dynamics on the unit velocity sphere. The stable CPU public surface currently supports only `metric="diagonal"`; `"dense"` and `"auto"` are rejected explicitly rather than silently downgraded. The stabilized CPU default regime now uses a longer warmup (`3500`), stricter `target_accept=0.985`, a tracked `max_leapfrog=1024` cap, and disables `eps_jitter` by default because that combination clears both funnel and eight-schools repeatability on the canonical `nextstat-bench` seed set while preserving the rest of the canonical suite. `init_l=0.0` means use the stable default `sqrt(d)` trajectory length in preconditioned space; it is not an auto-tuning surface. 4-phase Stan-style DualAveraging warmup with adaptive phase durations: when the `pathfinder` init strategy provides a Hessian-derived diagonal preconditioner, warmup phases are rebalanced (10%/15%/10%/65% vs default 15%/40%/15%/30%) to spend less time on mass-matrix collection and more on equilibration. Returns ArviZ-compatible dict with `posterior`, `sample_stats`, `diagnostics`; `sample_stats` includes `metric_type="diagonal"`, per-chain `n_leapfrog_warmup_total`, and divergence flags that now correctly mark early-terminated non-finite energy-error transitions. Typically 1.3–1.7x better ESS/gradient than NUTS on hierarchical models. `init_strategy`: `"random"` (default), `"mle"`, or `"pathfinder"` (mode + Hessian-derived diagonal preconditioner for faster warmup, not a standalone Pathfinder posterior-approximation surface; avoid on funnel-like geometries).
- `nextstat.sample_laps(model, *, model_data=None, n_chains=4096, n_warmup=500, n_samples=2000, seed=42, target_accept=0.9, init_step_size=0.0, init_l=0.0, max_leapfrog=1024, device_ids=None, sync_interval=100, welford_chains=256, batch_size=1000, fused_transitions=1000, divergence_threshold=1000.0) -> dict` — **LAPS** (Late-Adjusted Parallel Sampler): GPU-accelerated MAMS on CUDA. Also available via `nextstat.sample(model, method="laps", ...)`. Runs `n_chains` chains simultaneously on GPU with zero warp divergence (fixed trajectory length). Four-phase warmup: Phase 1 (fast DA, step-size adapt) + Phase 2 (DA + Welford, mass matrix) + Phase 3 (DA with new metric) + Phase 4 (L tuning + equilibrate). All phases use exact MH. `model`: `"std_normal"`, `"eight_schools"`, `"neal_funnel"`, `"neal_funnel_ncp"`, `"neal_funnel_riemannian"`, `"glm_logistic"`, `"glm_linear"`, `"glm_poisson"`, `"glm_negbin"` (or `"glm_negative_binomial"`), `"glm_composed_logistic"`, or a `RawCudaModel` instance. GLM models require `model_data` with `x`, `y`, `n`, `p`. Poisson/NegBin accept optional `offset` (length-n array). NegBin samples an additional `log_alpha` dispersion parameter (dim = p+1). ComposedLogistic requires `group_idx`, `n_groups`, and optional `re_prior_sigma` (default 1.0); dim = p + n_groups. For Neal's funnel, prefer `"neal_funnel_ncp"` (non-centered parametrization, R-hat < 1.02, ESS/s > 40k). `"neal_funnel_riemannian"` uses hybrid Riemannian metric for x-components but has known v-bias — experimental. `model_data`: dict with model-specific data (e.g. `{"y": [...], "sigma": [...]}` for eight_schools, `{"dim": 10}` for std_normal). `device_ids`: list of GPU device indices (default `None` = auto-detect all GPUs). Multi-GPU: chains are split across devices with synchronized warmup adaptation and independent sampling. `sync_interval`: warmup diagnostics sync frequency (default 100). `welford_chains`: chains per device for mass matrix estimation (default 256). `batch_size`: transitions per GPU-side accumulation batch (default 1000). `fused_transitions`: when >0, a single kernel launch executes N transitions keeping chain state in registers, eliminating per-transition launch overhead (default 1000; set to 0 to disable). Returns same format as `sample_mams()` plus `wall_time_s`, `n_kernel_launches`, `n_gpu_chains`, `n_devices`, `device_ids`. Requires `cuda` or `metal` feature and a compatible GPU at runtime. On Apple Silicon (Metal, f32), only built-in models are supported (no JIT). When both CUDA and Metal are available, CUDA is preferred (f64 precision).
- `nextstat.RawCudaModel(dim, cuda_src, *, data=None, param_names=None)` — User-defined CUDA model for LAPS JIT compilation via NVRTC. The `cuda_src` must define `__device__ double user_nll(const double* x, int dim, const double* model_data)` and `__device__ void user_grad(const double* x, double* grad, int dim, const double* model_data)`. The `data` array is uploaded to GPU as `model_data`. PTX is cached to disk (`~/.cache/nextstat/ptx/`) keyed by SHA-256(source + GPU arch). Requires `cuda` feature.
- `nextstat.bayes.sample(model, *, method="nuts", return_idata=True, **kwargs)` — convenience wrapper that returns ArviZ `InferenceData` by default. Supports all four methods (nuts/walnuts/mams/laps).
- `nextstat.bayes.to_inferencedata(raw) -> InferenceData` — convert a raw sampling dict into ArviZ `InferenceData`.

#### Sampling quick start

```python
import nextstat as ns

# 1. Eight Schools — NUTS (3 lines)
model = ns.EightSchoolsModel([28,8,-3,7,-1,1,18,12], [15,10,16,11,9,11,10,18])
idata = ns.sample(model, method="nuts", n_samples=2000, return_idata=True)

# 2. Same model — MAMS (typically better ESS/grad on hierarchical models)
idata = ns.sample(model, method="mams", n_samples=2000, return_idata=True)

# 3. ArviZ diagnostics and plots
import arviz as az
az.summary(idata)          # R-hat, ESS, posterior summary
az.plot_trace(idata)       # trace + density plots
az.plot_pair(idata)        # pairwise scatter

# 4. Save to disk
idata = ns.sample(model, n_samples=2000, return_idata=True, out="trace.json")

# 5. GPU sampling with LAPS (requires CUDA build)
result = ns.sample("eight_schools", method="laps",
                   model_data={"y": [28,8,-3,7,-1,1,18,12],
                               "sigma": [15,10,16,11,9,11,10,18]},
                   n_chains=4096, n_samples=2000)

# 6. User-defined GPU model (NVRTC JIT, requires CUDA)
model = ns.RawCudaModel(dim=10, cuda_src=r'''
    __device__ double user_nll(const double* x, int dim, const double* data) {
        double v = data[0];
        double nll = 0.0;
        for (int i = 0; i < dim; i++) nll += 0.5 * x[i] * x[i] / v;
        return nll;
    }
    __device__ void user_grad(const double* x, double* grad, int dim, const double* data) {
        double v = data[0];
        for (int i = 0; i < dim; i++) grad[i] = x[i] / v;
    }
''', data=[1.0])
result = ns.sample_laps(model, n_chains=4096, n_samples=2000)

# 7. Raw dict (no ArviZ dependency needed)
raw = ns.sample(model, method="nuts", n_samples=1000)
print(raw["diagnostics"]["quality"]["status"])  # "ok" / "warn" / "fail"
```

### Toy data

- `nextstat.asimov_data(model, params) -> list[float]` — Asimov dataset (expected counts).
- `nextstat.poisson_toys(model, params, *, n_toys=1000, seed=42) -> list[list[float]]` — Poisson fluctuated toy datasets.
- `nextstat.fit_toys(model, params, *, n_toys=1000, seed=42, device="cpu", batch=True, compute_hessian=False, max_retries=3) -> list[FitResult]` — unified toy fitting. Dispatches on model type (`HistFactoryModel` or `UnbinnedModel`) and device (`"cpu"`, `"cuda"`, `"metal"`). CPU path uses Rayon parallelism; GPU paths use lockstep L-BFGS-B kernels. `batch=True` (default) skips Hessian/covariance for throughput; `compute_hessian=True` when parameter pulls are needed.

### Visualization artifacts

- `nextstat.cls_curve(model, scan, *, alpha=0.05, data=None) -> dict` — asymptotic CLs exclusion curve. Returns `{alpha, nsigma_order, obs_limit, exp_limits, mu_values, cls_obs, cls_exp, points}`.
- `nextstat.viz.profile_curve(model, mu_values, *, data=None) -> dict` — profile likelihood curve (convenience wrapper calling `profile_scan(return_curve=True)`). Returns `{poi_index, mu_hat, nll_hat, mu_values, q_mu_values, twice_delta_nll, points}`.
- `nextstat.viz.ranking_artifact(model, *, top_n=None) -> dict` — ranking artifact with `entries`.
- `nextstat.viz.ranking_arrays(artifact_or_entries) -> dict` — normalized ranking arrays `{names, delta_mu_up, delta_mu_down, pull, constraint}`.
- `nextstat.viz.corr_arrays(artifact) -> dict` — normalized corr arrays `{parameter_names, corr}`.
- `nextstat.viz.corr_subset(artifact, *, include=None, exclude=None, top_n=None, order="input"|"max_abs_corr"|"group_base") -> dict` — filtered/reordered matrix view.
- Plotting helpers (require matplotlib): `plot_cls_curve`, `plot_brazil_limits`, `plot_profile_curve`, `plot_pulls`, `plot_ranking`, `plot_corr_matrix`.

### Native Rust rendering (no matplotlib required)

- `nextstat.viz.render_svg(artifact, kind, *, config=None) -> str` — render artifact dict to SVG string using the native Rust renderer. Supports all 17 artifact kinds. `config` is an optional dict with VizConfig overrides (theme, colors, experiment label, etc.).
- `nextstat.viz.render_to_file(artifact, kind, path, *, config=None, dpi=None) -> None` — render artifact to file. Format inferred from extension (`.svg`, `.pdf`, `.png`). `dpi` applies to PNG output (default 220).
- `nextstat._core.render_viz(artifact_json, kind, format="svg", *, config_yaml=None, dpi=None) -> bytes` — low-level: render artifact JSON string to bytes.

### Parameter ranking

- `nextstat.ranking(model, *, device="cpu") -> list[dict]` — nuisance parameter ranking (impact on POI). Dispatches on model type: `HistFactoryModel` or `UnbinnedModel`. `device="cuda"` or `device="metal"` for GPU-accelerated ranking (requires corresponding build feature). For simplified-likelihood models, entries rank reduced nuisance coordinates from the compiled model; covariance-form and `derived_from_workspace` artifacts do not preserve source-level nuisance identities. This remains consistent with the published exporter stable-source-semantics boundary: reduced artifacts are not a source-level nuisance-identity surface.

### Utilities

- `nextstat.ols_fit(x, y, *, include_intercept=True) -> list[float]` — closed-form OLS.
- `nextstat.rk4_linear(a, y0, t0, t1, dt, *, max_steps=100000) -> dict` — RK4 ODE solver for linear systems.
- `nextstat.rk4_linear_dde(a, b, y0, y_history, t0, t1, tau, dt, *, max_steps=100000) -> dict` — fixed-step RK4 for linear delay differential equations `dy/dt = A y(t) + B y(t - tau)` (method-of-steps, requires `dt <= tau`).
- `nextstat.set_eval_mode(mode: str) -> None` — set evaluation mode (`"fast"` or `"parity"`).
- `nextstat.set_threads(threads: int) -> bool` — best-effort: configure the global Rayon thread pool size (returns `True` if applied).
- `nextstat.get_eval_mode() -> str` — query current evaluation mode.
- `nextstat.has_accelerate() -> bool` — check Apple Accelerate backend availability.
- `nextstat.has_cuda() -> bool` — check CUDA backend availability.
- `nextstat.has_metal() -> bool` — check Metal backend availability.

---

## Core classes

### `FitResult`

Fields:
- `parameters: list[float]` — best-fit parameters (NextStat order)
- `uncertainties: list[float]` — 1-sigma uncertainties (diagonal or covariance-derived)
- `nll: float` — NLL at the optimum
- `converged: bool` — optimizer convergence flag
- `n_iter: int` — number of optimizer iterations
- `n_fev: int` — number of function evaluations
- `n_gev: int` — number of gradient evaluations
- `termination_reason: str` — optimizer termination message (e.g. `"SolverConverged"`, `"TargetCostReached"`, `"MaxIterReached"`, `"1D golden-section search"`)
- `final_grad_norm: float` — L2 norm (Euclidean) of the gradient at minimum
- `initial_nll: float` — NLL at the starting point
- `n_active_bounds: int` — number of parameters at their box constraint boundary
- `edm: float` — Estimated Distance to Minimum (EDM = g^T H^{-1} g). Uses the L-BFGS inverse Hessian approximation. `NaN` if unavailable (gradient-free paths). Minuit-compatible convergence metric.
- `warnings: list[str]` — identifiability warnings (near-singular Hessian, non-finite uncertainties, near-zero Hessian diagonal). Empty list when model is well-identified.

Compatibility aliases:
- `bestfit` (same as `parameters`)
- `twice_nll = 2 * nll`
- `success` (same as `converged`)
- `n_evaluations` (back-compat alias for `n_iter`)

### `FitMinimumResult`

Fast-path optimizer result (no covariance/Hessian). Returned by `MaximumLikelihoodEstimator.fit_minimum(model, *, data=None, init_pars=None, bounds=None)`.

Fields:
- `parameters: list[float]`
- `nll: float`
- `converged: bool`
- `n_iter: int`
- `n_fev: int`, `n_gev: int`
- `message: str`
- `initial_nll: float`
- `final_gradient: list[float] | None`
- `edm: float` — Estimated Distance to Minimum (EDM = g^T H^{-1} g). `NaN` if unavailable.

Compatibility aliases:
- `bestfit` (same as `parameters`)
- `twice_nll = 2 * nll`
- `success` (same as `converged`)

### `MaximumLikelihoodEstimator`

The object-oriented MLE surface:

```python
import nextstat

mle = nextstat.MaximumLikelihoodEstimator(max_iter=1000, tol=1e-6, m=0, smooth_bounds=False)
res = mle.fit(model)
```

 Constructor args (keyword-only):
 - `max_iter=1000`: max optimizer iterations
 - `tol=1e-6`: convergence tolerance (gradient norm)
 - `m=0`: L-BFGS memory size (0 = auto-select based on model dimension: `max(10, min(50, n_params/5))`)
 - `smooth_bounds=False`: enable smooth bounds transform instead of hard clamping
 
 Also supports:
 - `fit_batch(models_or_model, datasets=None)` for homogeneous lists of models, or `HistFactoryModel` + multiple datasets.
 - `fit_minimum(model, *, data=None, init_pars=None, bounds=None) -> FitMinimumResult` — fast-path NLL minimization intended for profile scans and conditional fits.
   - `bounds=` is currently supported for `HistFactoryModel` only; clamp a parameter to `(value, value)` to fix it.
 - `fit_toys(model, params, *, n_toys=1000, seed=42) -> list[FitResult]` — CPU parallel toy fitting (Rayon).
 - `ranking(model) -> list[dict]` — nuisance parameter ranking.
 - `q0_like_loss_and_grad_nominal(model, *, channel, sample, nominal) -> (float, list[float])` — discovery q₀ and gradient w.r.t. one sample's nominal yields. Runs profiled fit internally. For ML training loops where the signal histogram is a differentiable function of NN weights.
 - `qmu_like_loss_and_grad_nominal(model, *, mu_test, channel, sample, nominal) -> (float, list[float])` — exclusion qμ and gradient. Same contract as q₀ but tests `mu = mu_test` instead of `mu = 0`.

### `Posterior`

Wraps a model and exposes constrained/unconstrained log density for sampling and MAP:

```python
import nextstat

post = nextstat.Posterior(model)
post.set_prior_normal("mu", center=0.0, width=5.0)
res = nextstat.map_fit(post)
```

Methods:
- `dim()`, `parameter_names()`, `suggested_init()`, `suggested_bounds()`
- `set_prior_flat(name)`, `set_prior_normal(name, center, width)`, `clear_priors()`, `priors() -> dict`
- `logpdf(theta)`, `grad(theta)` — constrained space
- `to_unconstrained(theta)`, `to_constrained(z)` — bijective transforms
- `logpdf_unconstrained(z)`, `grad_unconstrained(z)` — unconstrained space (for NUTS)

---

## Models

All models implement a shared minimal contract:
- `n_params()` / `dim()`
- `nll(params)`, `grad_nll(params)`
- `parameter_names()`, `suggested_init()`, `suggested_bounds()`

### HEP (HistFactory / pyhf JSON)

- `HistFactoryModel`: build from pyhf JSON (via `nextstat.from_pyhf` or `HistFactoryModel.from_workspace`).
  - `expected_data(params, *, include_auxdata=True) -> list[float]`
  - `with_observed_main(observed_main) -> HistFactoryModel` — return model with replaced observed data.
  - `set_sample_nominal(*, channel, sample, nominal)` — override one sample's nominal yields in-place (for ML/RL).
  - `poi_index() -> int | None`
  - `observed_main_by_channel() -> list[dict]`
  - `expected_main_by_channel_sample(params) -> list[dict]`

### HEP / Unbinned (event-level)

- `UnbinnedModel.from_config(path) -> UnbinnedModel` — compile from `unbinned_spec_v0` JSON/YAML.
  - `schema_version() -> str`
  - `poi_index() -> int | None`
  - `with_fixed_param(param_idx, value) -> UnbinnedModel`
- `nextstat.unbinned.UnbinnedAnalysis(model)` — high-level workflow helper over `UnbinnedModel`.
  - `UnbinnedAnalysis.from_config(path) -> UnbinnedAnalysis`
  - `fit(*, init_pars=None) -> FitResult`
  - `fit_toys(params=None, *, n_toys=1000, seed=42) -> list[FitResult]`
  - `scan(mu_values) -> dict` (delegates to `profile_scan`)
  - `hypotest(mu_test) -> dict`
  - `hypotest_toys(poi_test, *, n_toys=1000, seed=42, expected_set=False, return_tail_probs=False, return_meta=False) -> float | tuple | dict`
  - `ranking() -> list[dict]`
  - `parameter_index(param: int | str) -> int`
  - `with_fixed_param(param: int | str, value: float) -> UnbinnedAnalysis`
  - `summary() -> dict`

Toy-fit parity (Python vs CLI, same spec/seed):

```python
import nextstat

model = nextstat.UnbinnedModel.from_config("spec.json")
params = model.suggested_init()

# CPU batch: Rayon-parallel toy fits (dispatches on model type)
results = nextstat.fit_toys(model, params, n_toys=100, seed=42)

# Each result has: .parameters, .nll, .converged, .n_iter, .n_fev, .n_gev
converged = sum(1 for r in results if r.converged)
print(f"{converged}/{len(results)} toys converged")
```

```bash
nextstat unbinned-fit-toys --config spec.json --n-toys 100 --seed 42 --threads 1
# GPU parity variant:
nextstat unbinned-fit-toys --config spec.json --n-toys 100 --seed 42 --threads 1 --gpu cuda
```

### HEP / Hybrid (binned + unbinned)

- `HybridModel.from_models(binned, unbinned, poi_from="binned") -> HybridModel` — combine a `HistFactoryModel` and an `UnbinnedModel` into a single likelihood with shared parameters matched by name.
  - `poi_from`: `"binned"` (default) or `"unbinned"` — which model provides the POI.
  - `n_shared() -> int` — number of shared parameters.
  - `poi_index() -> int | None`
  - `with_fixed_param(param_idx, value) -> HybridModel`

```python
import nextstat

binned = nextstat.HistFactoryModel.from_workspace(open("workspace.json").read())
unbinned = nextstat.UnbinnedModel.from_config("unbinned.yaml")
hybrid = nextstat.HybridModel.from_models(binned, unbinned, poi_from="binned")

result = nextstat.fit(hybrid)
print(f"Shared params: {hybrid.n_shared()}, Total: {hybrid.n_params()}")
```

### Regression / GLM

- `LinearRegressionModel(x, y, *, include_intercept=True)`
- `LogisticRegressionModel(x, y, *, include_intercept=True)`
- `PoissonRegressionModel(x, y, *, include_intercept=True, offset=None)`
- `NegativeBinomialRegressionModel(x, y, *, include_intercept=True, offset=None)`
- `GammaRegressionModel(x, y, *, include_intercept=True)` — Gamma GLM with log link. Parameters: regression coefficients β + `log_alpha` (shape). For strictly positive continuous data (insurance claims, hospital costs).
- `TweedieRegressionModel(x, y, *, p=1.5, include_intercept=True)` — Tweedie compound Poisson-Gamma GLM with log link. Power `p ∈ (1, 2)`. Handles exact zeros. Parameters: β + `log_phi` (dispersion). For insurance aggregate claims, rainfall.
  - `.power()` — returns the Tweedie power parameter.
- `ComposedGlmModel` — hierarchical GLMs via static constructors:
  - `.linear_regression(x, y, *, include_intercept, group_idx, n_groups, coef_prior_mu, coef_prior_sigma, penalize_intercept, obs_sigma_prior_m, obs_sigma_prior_s, random_intercept_non_centered, random_slope_feature_idx, random_slope_non_centered, correlated_feature_idx, lkj_eta)`
  - `.logistic_regression(x, y, *, include_intercept, group_idx, n_groups, coef_prior_mu, coef_prior_sigma, penalize_intercept, random_intercept_non_centered, random_slope_feature_idx, random_slope_non_centered, correlated_feature_idx, lkj_eta)`
  - `.poisson_regression(x, y, *, include_intercept, offset, group_idx, n_groups, coef_prior_mu, coef_prior_sigma, penalize_intercept, random_intercept_non_centered, random_slope_feature_idx, random_slope_non_centered, correlated_feature_idx, lkj_eta)`

### Ordinal regression

- `OrderedLogitModel(x, y, *, n_levels)`
- `OrderedProbitModel(x, y, *, n_levels)`

### Hierarchical / mixed models

- `LmmMarginalModel(x, y, *, include_intercept, group_idx, n_groups, random_slope_feature_idx)` — Gaussian mixed model (marginal likelihood).

### Extreme Value Theory (EVT)

- `GevModel(data)` — Generalized Extreme Value distribution for block maxima. Parameters: `[mu, log_sigma, xi]` (location, log-scale, shape). Fréchet (ξ>0), Gumbel (ξ≈0), Weibull (ξ<0).
  - `GevModel.return_level(params, return_period)` — static method, computes the T-block return level (e.g. 100-year flood).
- `GpdModel(exceedances)` — Generalized Pareto Distribution for peaks-over-threshold. Parameters: `[log_sigma, xi]` (log-scale, shape).
  - `GpdModel.quantile(params, p)` — static method, computes excess quantile (VaR/ES).

### Meta-analysis

- `nextstat.meta_fixed(estimates, standard_errors, *, labels=None, conf_level=0.95) -> dict` — fixed-effects meta-analysis (inverse-variance weighting).
- `nextstat.meta_random(estimates, standard_errors, *, labels=None, conf_level=0.95) -> dict` — random-effects meta-analysis (DerSimonian–Laird).

**Returns** a dict with keys:
- `estimate`, `se`, `ci_lower`, `ci_upper`, `z`, `p_value` — pooled effect
- `method` — `"fixed"` or `"random"`
- `conf_level`, `k` — confidence level and number of studies
- `heterogeneity` — dict with `q`, `df`, `p_value`, `i_squared`, `h_squared`, `tau_squared`
- `forest` — list of per-study dicts with `label`, `estimate`, `se`, `ci_lower`, `ci_upper`, `weight`

### Survival analysis
- `ExponentialSurvivalModel(times, events)`
- `WeibullSurvivalModel(times, events)`
- `LogNormalAftModel(times, events)`
- `CoxPhModel(times, events, x, *, ties="efron")` — Cox proportional hazards model (partial likelihood).
- `IntervalCensoredWeibullModel(time_lower, time_upper, censor_type)` — Weibull model with interval censoring (exact, right, left, interval).
- `IntervalCensoredWeibullAftModel(time_lower, time_upper, censor_type, covariates)` — Weibull AFT with covariates and interval censoring. Parameters: `[log_k, beta_0, ..., beta_{p-1}]`. `log(λ_i) = x_i^T β`.
- `IntervalCensoredExponentialModel(time_lower, time_upper, censor_type)` — Exponential model (Weibull k=1) with interval censoring.
- `IntervalCensoredLogNormalModel(time_lower, time_upper, censor_type)` — LogNormal model with interval censoring.
 
High-level helpers (recommended for most users):
 
- `nextstat.survival.exponential.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.weibull.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.lognormal_aft.fit(times, events) -> ParametricSurvivalFit`
- `nextstat.survival.cox_ph.fit(times, events, x, *, ties="efron", robust=True, compute_cov=True, groups=None, cluster_correction=True, compute_baseline=True) -> CoxPhFit`
  - `robust=True` returns sandwich SE (`fit.robust_se`) (requires `compute_cov=True`)
  - `compute_cov=True` computes covariance/SE (`fit.cov`, `fit.se`)
  - `groups=cluster_ids` switches sandwich SE to cluster-robust (`fit.robust_kind == "cluster"`)
  - `cluster_correction=True` applies small-sample correction for clustered sandwich covariance (requires `groups`)
  - `compute_baseline=True` enables baseline hazard estimation (required for `fit.predict_survival(x_new, times=grid)`)
 
### Time series / state space

Low-level:
- `KalmanModel(f, q, h, r, m0, p0)` — linear state-space model (matrices as lists of lists).
  - `n_state()`, `n_obs()`
- `nextstat.kalman_filter(model, ys) -> dict` — forward Kalman filter (supports missing obs as `None`).
- `nextstat.kalman_smooth(model, ys) -> dict` — RTS smoother.
- `nextstat.kalman_em(model, ys, *, max_iter, tol, estimate_q, estimate_r, estimate_f, estimate_h, min_diag) -> dict` — EM parameter estimation.
- `nextstat.kalman_forecast(model, ys, *, steps, alpha) -> dict` — multi-step forecast with intervals.
- `nextstat.kalman_simulate(model, *, t_max, seed, init, x0) -> dict` — simulate from state-space model.

High-level wrappers:
- `nextstat.timeseries.*` — convenience helpers and plotting.
- `nextstat.timeseries.local_level_model(*, q, r, m0=0.0, p0=1.0) -> KalmanModel`
- `nextstat.timeseries.local_linear_trend_model(*, q_level, q_slope, r, level0=0.0, slope0=0.0, p0_level=1.0, p0_slope=1.0) -> KalmanModel`
- `nextstat.timeseries.local_level_seasonal_model(*, period, q_level, q_season, r, level0=0.0, p0_level=1.0, p0_season=1.0) -> KalmanModel`
- `nextstat.timeseries.local_linear_trend_seasonal_model(*, period, q_level, q_slope, q_season, r, level0=0.0, slope0=0.0, p0_level=1.0, p0_slope=1.0, p0_season=1.0) -> KalmanModel`
- `nextstat.timeseries.local_level_weekly_model(*, q_level, q_weekly, r, level0=0.0, p0_level=1.0, p0_weekly=1.0) -> KalmanModel` — fixed weekly alias for `period=7`.
- `nextstat.timeseries.local_linear_trend_weekly_model(*, q_level, q_slope, q_weekly, r, level0=0.0, slope0=0.0, p0_level=1.0, p0_slope=1.0, p0_weekly=1.0) -> KalmanModel` — fixed weekly alias for `period=7`.

### Ads-native measurement / response models

Low-level:
- `nextstat.BetaBinomialModel(alpha, beta)` — beta prior over latent conversion rates.
  - `fit(conversion_rates, sample_sizes) -> BetaBinomialModel`
  - `fit_from_counts(successes, trials) -> BetaBinomialModel`
  - `alpha`, `beta`
  - `mean()`, `variance()`, `overdispersion()`
  - `posterior(successes, trials) -> BetaBinomialModel`
  - `predictive_mean(trials)`, `predictive_variance(trials)`
- `nextstat.DelayCorrectionModel(lambda_, lambda_se=None)` — exponential delay-censoring correction model.
  - `fit_from_lag_buckets(buckets) -> DelayCorrectionModel`
  - `lambda_`, `lambda_se`
  - `observed_fraction(window_days) -> float`
  - `correct(observed_count, window_days) -> (corrected_count, uncertainty)`

High-level wrappers:
- `nextstat.ads.BetaBinomialModel` — lazy convenience export of the native class.
- `nextstat.ads.DelayCorrectionModel` — lazy convenience export of the native class.
- `nextstat.ads.cuped_adjust(control_outcomes, control_covariates, variant_outcomes, variant_covariates, *, covariate_name=None, covariate_provenance=None, pre_treatment_only=True) -> dict`
  - one-covariate CUPED adjustment
  - `covariate_provenance` may provide `{name, timing, source_dataset}` for
    fail-fast leakage validation
  - returns adjusted means, `theta`, `rho`, `r_squared`,
    `variance_reduction_factor`, `effective_sample_multiplier`,
    `selected_covariates`, `covariate_provenance`, `provenance_validated`,
    `solver`, `condition_number`, `ridge_lambda`, and `pre_treatment_only`
- `nextstat.ads.cure_adjust(control_outcomes, control_covariates, variant_outcomes, variant_covariates, *, covariate_names=None, covariate_provenance=None, pre_treatment_only=True) -> dict`
  - multivariate CURE adjustment
  - `control_covariates` / `variant_covariates` are row-major pre-treatment
    covariate matrices
  - `covariate_provenance` accepts one `{name, timing, source_dataset}` mapping
    per covariate
  - returns adjusted means, coefficient vector `theta`, pooled `r_squared`,
    `variance_reduction_factor`, `effective_sample_multiplier`,
    `selected_covariates`, `covariate_provenance`, `provenance_validated`,
    `solver`, `condition_number`, `ridge_lambda`, and `pre_treatment_only`
- Architectural rule: `nextstat.ads.cuped_adjust(...)` is the one-covariate
  case of the shared CURE layer.
- Guardrails:
  - only pre-treatment covariates are accepted
  - `covariate_provenance[*]["timing"]` must remain `pre_treatment`
  - ill-conditioned designs report `solver="ridge"` and the chosen
    `ridge_lambda`
  - committed public-surface reference fixtures live under
    `tests/fixtures/variance_reduction/`
- Surface boundary:
  - these helpers are part of the stable Python API surface
  - they are also exposed through `nextstat.tools` as
    `nextstat_ads_cuped_adjust` and `nextstat_ads_cure_adjust`
  - the same two tools are available through the authenticated
    `nextstat-server` server-safe subset
  - performance-evidence and benchmark governance live in
    `docs/benchmarks/ads-variance-reduction-runbook-2026-03-08.md` and
    `docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md`
  - local acceptance / CI gate details live in
    `docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09.md`
    and `docs/benchmarks/ads-variance-reduction-runtime-gate.md`
- `nextstat.ads.hill(x, ec, slope) -> float` — Hill saturation helper.
- `nextstat.ads.adstock_geometric(spend, decay) -> list[float]` — geometric adstock helper.

### GARCH / Stochastic Volatility

Volatility models for financial time series. Available via the `nextstat.volatility` convenience module or directly from `nextstat._core`.

- `nextstat.volatility.garch(ys, *, max_iter=1000, tol=1e-6, alpha_beta_max=0.999, min_var=1e-18) -> dict` — fit Gaussian GARCH(1,1) by MLE.
  - Returns: `params` (`{mu, omega, alpha, beta}`), `conditional_variance`, `conditional_sigma`, `log_likelihood`, `converged`, `n_iter`, `message`.
- `nextstat.volatility.sv(ys, *, max_iter=1000, tol=1e-6, log_eps=1e-12) -> dict` — fit approximate stochastic volatility (SV) via log(χ²₁) Gaussian approximation + Kalman MLE.
  - Returns: `params` (`{mu, phi, sigma}`), `smoothed_h` (log-variance), `smoothed_sigma` (exp(h/2)), `log_likelihood`, `converged`, `n_iter`, `message`.

- `nextstat.volatility.egarch(ys, *, max_iter=1000, tol=1e-6) -> dict` — fit EGARCH(1,1) by MLE. Nelson (1991) log-variance formulation with leverage.
  - Returns: `params` (`{mu, omega, alpha, gamma, beta}`), `conditional_variance`, `conditional_sigma`, `log_likelihood`, `converged`, `n_iter`, `message`.
- `nextstat.volatility.gjr_garch(ys, *, max_iter=1000, tol=1e-6, persistence_max=0.999, min_var=1e-18) -> dict` — fit GJR-GARCH(1,1) by MLE. Glosten–Jagannathan–Runkle threshold GARCH.
  - Returns: `params` (`{mu, omega, alpha, gamma, beta}`), `conditional_variance`, `conditional_sigma`, `log_likelihood`, `converged`, `n_iter`, `message`.

Low-level (same functions, no wrapper):
- `nextstat._core.garch11_fit(ys, *, max_iter, tol, alpha_beta_max, min_var) -> dict`
- `nextstat._core.sv_logchi2_fit(ys, *, max_iter, tol, log_eps) -> dict`
- `nextstat._core.egarch11_fit(ys, *, max_iter, tol) -> dict`
- `nextstat._core.gjr_garch11_fit(ys, *, max_iter, tol, persistence_max, min_var) -> dict`

```python
from nextstat.volatility import garch, sv

returns = [0.01, -0.02, 0.005, 0.03, -0.015, 0.002, -0.04, 0.035]

# GARCH(1,1)
g = garch(returns)
print(f"omega={g['params']['omega']:.4f}, alpha={g['params']['alpha']:.4f}, beta={g['params']['beta']:.4f}")
print(f"Conditional sigma: {g['conditional_sigma'][:5]}")

# Stochastic Volatility
s = sv(returns)
print(f"phi={s['params']['phi']:.4f}, sigma_eta={s['params']['sigma']:.4f}")
print(f"Smoothed sigma: {s['smoothed_sigma'][:5]}")
```

### Pharmacometrics

#### Individual PK Models

- `OneCompartmentOralPkModel(times, y, *, dose, bioavailability=1.0, sigma=0.05, lloq=None, lloq_policy="censored")` — 1-compartment oral PK (3 params: CL, V, Ka).
  - `predict(params) -> list[float]` — predicted concentrations.
- `TwoCompartmentIvPkModel(times, y, *, dose, error_model="additive", sigma=0.05, sigma_add=None, lloq=None, lloq_policy="censored")` — 2-compartment IV bolus PK (4 params: CL, V1, V2, Q). Supports additive/proportional/combined error models.
  - `predict(params) -> list[float]` — predicted concentrations.
- `TwoCompartmentOralPkModel(times, y, *, dose, bioavailability=1.0, error_model="additive", sigma=0.05, sigma_add=None, lloq=None, lloq_policy="censored")` — 2-compartment oral PK (5 params: CL, V1, V2, Q, Ka). Supports additive/proportional/combined error models.
  - `predict(params) -> list[float]` — predicted concentrations.
- `ThreeCompartmentIvPkModel(times, y, *, dose, error_model="additive", sigma=0.05, sigma_add=None, lloq=None, lloq_policy="censored")` — 3-compartment IV bolus PK (6 params: CL, V1, Q2, V2, Q3, V3). Analytical macro-constant solution with eigenvalue gradients.
  - `predict(params) -> list[float]` — predicted concentrations.
- `ThreeCompartmentOralPkModel(times, y, *, dose, bioavailability=1.0, error_model="additive", sigma=0.05, sigma_add=None, lloq=None, lloq_policy="censored")` — 3-compartment oral PK (7 params: CL, V1, Q2, V2, Q3, V3, Ka). Analytical macro-constant solution.
  - `predict(params) -> list[float]` — predicted concentrations.

All PK models implement the `LogDensityModel` interface: `nll(params)`, `grad_nll(params)`, `parameter_names()`, `suggested_init()`, `suggested_bounds()`, `n_params()`, `dim()`.

#### Competing Risks

- `nextstat._core.cumulative_incidence(times, events, target_cause, *, conf_level=0.95) -> dict` — Aalen-Johansen CIF estimator.
  - Returns: `cause`, `times`, `cif`, `se`, `ci_lower`, `ci_upper`, `n`, `n_events`.
- `nextstat._core.gray_test(times, events, groups, target_cause) -> dict` — Gray's K-sample test for comparing CIF.
  - Returns: `statistic`, `df`, `p_value`.
- `nextstat._core.fine_gray_fit(times, events, x, p, target_cause) -> dict` — Fine-Gray subdistribution hazard regression.
  - `x` is a flat row-major covariate matrix (n × p).
  - Returns: `coefficients`, `se`, `z`, `p_values`, `n`, `n_events`, `log_likelihood`.

#### Pharmacodynamic (PD) Models

Direct-effect and indirect-response models for dose-response analysis.

- `nextstat._core.emax_predict(e0, emax, ec50, conc) -> dict` — Emax model predictions.
  - Returns: `predictions` (list), `e0`, `emax`, `ec50`.
- `nextstat._core.emax_nll(e0, emax, ec50, conc, obs, *, error_model="additive", sigma=0.05) -> float` — Emax negative log-likelihood.
- `nextstat._core.sigmoid_emax_predict(e0, emax, ec50, gamma, conc) -> dict` — Sigmoid-Emax (Hill) predictions.
  - Returns: `predictions` (list), `e0`, `emax`, `ec50`, `gamma`.
- `nextstat._core.sigmoid_emax_nll(e0, emax, ec50, gamma, conc, obs, *, error_model="additive", sigma=0.05) -> float` — Sigmoid-Emax negative log-likelihood.
- `nextstat._core.idr_simulate(idr_type, kin, kout, max_effect, c50, conc_times, conc_values, output_times, *, r0=None) -> dict` — Indirect response model ODE simulation.
  - `idr_type`: `"inhibit_production"`, `"inhibit_loss"`, `"stimulate_production"`, `"stimulate_loss"` (or `"type1"`–`"type4"`).
  - Returns: `times`, `response`, `baseline`.
- `nextstat._core.idr_nll(idr_type, kin, kout, max_effect, c50, conc_times, conc_values, obs_times, obs_values, *, error_model="additive", sigma=0.05, r0=None) -> float` — IDR negative log-likelihood.

#### ODE PK Models

- `nextstat._core.ode_pk_solve(model_type, params, dose_times, dose_amounts, dose_routes, obs_times, *, obs_compartment=None, dose_compartment=None, solver="rk45", rtol=1e-8, atol=1e-10) -> dict` — Solve ODE-based PK models.
  - `model_type`: `"transit_1cpt"`, `"transit_2cpt"`, `"mm_1cpt"`, `"mm_2cpt"`, `"tmdd"`.
  - `params`: model-specific parameter list (transit_1cpt: `[n_transit, ktr, cl, v1]`; transit_2cpt: `[n_transit, ktr, cl, v1, q, v2]`; mm_1cpt: `[vmax, km, v1]`; mm_2cpt: `[vmax, km, v1, q, v2]`; tmdd: `[kon, koff, kel, ksyn, kdeg, kint, v]`).
  - `dose_routes`: list of `"iv"`, `"oral"`, `"oral_f0.8"` (custom bioavailability), `"infusion_1h"`.
  - `solver`: `"rk45"` (non-stiff), `"esdirk4"` (stiff, recommended for TMDD), `"lsoda"` (automatic).
  - Returns: `concentrations`, `times`, `auc`, `cmax`, `tmax`.
- `nextstat._core.ode_pk_nll_py(model_type, params, dose_times, dose_amounts, dose_routes, obs_times, obs_values, *, sigma=0.05, obs_compartment=None, dose_compartment=None, solver="rk45", rtol=1e-8, atol=1e-10) -> float` — Compute additive Gaussian NLL for ODE PK models. Same arguments as `ode_pk_solve` plus `obs_values` and `sigma`.

#### Population PK (NLME)

- `OneCompartmentOralPkNlmeModel(times, y, subject_idx, n_subjects, *, dose, bioavailability=1.0, sigma=0.05, lloq=None, lloq_policy="censored")` — population PK (NLME with per-subject random effects). `LogDensityModel` interface.

- `nextstat.nlme_foce(times, y, subject_idx, n_subjects, *, model="1cpt_oral", method="focei", doses, bioavailability=1.0, error_model="proportional", sigma=0.1, sigma_add=None, theta_init, omega_init, omega_matrix=None, omega_fixed=None, diagonal_omega=False, max_outer_iter=100, max_inner_iter=20, tol=1e-4, rel_tol=1e-8, interaction=True, omega_damping=0.7, omega_max_ratio=100.0, estimate_sigma=True, lloq=None, its_max_iter=30, its_max_individual_iter=60, its_tol=1e-4, its_omega_damping=0.3, imp_n_iter=15, imp_n_samples=300, imp_proposal_scale=1.0, imp_seed=42, imp_tol=1e-4, imp_e_only=False, time_varying_covariates=None, iov=None, random_effect_transforms=None, regimens=None) -> FoceResult` — FOCE/FOCEI/FO/ITS/IMP population estimation with multi-model dispatch. `doses` is per-subject (or length 1 for broadcast). `omega_fixed` is a per-parameter boolean list that fixes specific random effects to their initial values (e.g. `[False, False, True]` to fix Ka). `diagonal_omega=True` forces a diagonal Ω matrix (no off-diagonal correlations). `time_varying_covariates` and `iov` are currently exposed for `model="1cpt_oral"` with `method in {"foce","focei"}` and are mutually exclusive per call. Returns `theta`, `omega`, `omega_matrix`, `correlation`, `eta`, `ofv`, `converged`, `n_iter`, `sigma`, `sigma_init`.

- `nextstat.nlme_saem(times, y, subject_idx, n_subjects, *, model="1cpt_oral", doses, bioavailability=1.0, error_model="proportional", sigma=0.1, sigma_add=None, theta_init, omega_init, omega_matrix=None, covariates=None, time_varying_covariates=None, iov=None, n_burn=200, n_iter=100, n_chains=1, seed=12345, tol=1e-4, return_theta_trace=False, lloq=None) -> SaemResult` — SAEM population estimation with multi-model dispatch and covariate support. `model`: `"1cpt_oral"` (θ: CL, V, Ka), `"2cpt_iv"` (θ: CL, V1, Q, V2), `"2cpt_oral"` (θ: CL, V1, Q, V2, Ka), `"3cpt_iv"` (θ: CL, V1, Q2, V2, Q3, V3), `"3cpt_oral"` (θ: CL, V1, Q2, V2, Q3, V3, Ka). `doses` is per-subject (or length 1 for broadcast). Only one of `covariates`, `time_varying_covariates`, `iov` can be provided per call; `time_varying_covariates` and `iov` are currently exposed for `model="1cpt_oral"`. `return_theta_trace=True` enables per-iteration θ trace in diagnostics.

  **`SaemResult` (TypedDict):**
  - `theta`, `omega`, `omega_matrix`, `correlation`, `eta`, `ofv`, `converged`, `n_iter`
  - `saem: SaemDiagnosticsResult` — `acceptance_rates`, `ofv_trace`, `burn_in_only`, `theta_trace` (if requested), `relative_change`, `geweke_scores`

  **`CovariateDict` (TypedDict):**
  - `param_idx: int` — index into θ vector
  - `values: list[float]` — per-subject covariate values
  - `reference: float` — centering value (e.g. median weight)
  - `relationship: "power" | "exponential" | "proportional" | "categorical"`
  - `exponent: float` — fixed exponent (e.g. 0.75 for allometric)

  **`TimeVaryingCovariateDict` (TypedDict):**
  - `param_idx: int` — index into θ vector
  - `trajectories: list[list[tuple[float, float] | {"time": float, "value": float}]]` — per-subject time series
  - `reference: float` — centering value
  - `relationship: "power" | "exponential" | "proportional"`
  - `exponent: float` — used for `"power"` relationship
  - `interpolation: "locf" | "linear"`

  **`IovDict` (TypedDict):**
  - `param_indices: list[int]` — θ indices with inter-occasion variability
  - `occasion_start_times: list[list[float]]` — per-subject occasion boundaries
  - `omega_iov_init: list[float]` — IOV SD initial values (same length as `param_indices`)

- `nextstat.bootstrap_nlme(times, y, subject_idx, n_subjects, *, model="1cpt_oral", doses, bioavailability=1.0, error_model="proportional", sigma=0.1, sigma_add=None, theta, omega, covariates=None, n_bootstrap=200, conf_level=0.95, ci_method="percentile", n_burn=200, n_iter=100, n_chains=1, seed=42, tol=1e-4) -> BootstrapNlmeResult` — nonparametric bootstrap for NLME parameter uncertainty. Resamples subjects, refits SAEM, collects θ/Ω distributions.

  **`BootstrapNlmeResult` (TypedDict):**
  - `theta_ci: list[list[float]]` — per-parameter [lower, upper]
  - `omega_ci: list[list[float]]` — per-omega [lower, upper]
  - `theta_se: list[float]`, `omega_se: list[float]`
  - `n_successful: int`

**Estimation Methods** (`method=` in `nlme_foce`):

| Method | Description | When to use |
|--------|-------------|-------------|
| `"focei"` | FOCE with Interaction (default). Laplace approximation with individual-level η re-optimization and η–ε interaction. | General purpose; recommended default. |
| `"foce"` | FOCE without Interaction. Same as FOCEI but drops the η–ε interaction term. | Faster; adequate when residual error is small relative to BSV. |
| `"fo"` | First Order. Linearizes at η=0 (no individual optimization). Equivalent to `interaction=False` with a single outer pass. | Quick exploratory fits; large datasets where FOCE is too slow. |
| `"its"` | Iterative Two-Stage. Maps `ItsConfig` params onto the FOCE engine: `its_max_iter` → outer iterations, `its_max_individual_iter` → inner iterations, `its_omega_damping` → omega blend. | Robust starting values for FOCE/SAEM; multi-stage workflows. |
| `"imp"` | Importance Sampling MC-EM. True IS-EM: MAP + importance sampling from N(η̂, scale²Ω), weighted M-step for Ω, gradient-based θ update. Returns `(FoceResult, ImpDiagnostics)`. | Most robust for ill-conditioned models; gold-standard SE via sampling. |

**IMP-specific parameters** (prefixed `imp_`):
- `imp_n_iter` (15) — number of IS-EM iterations
- `imp_n_samples` (300) — importance samples per subject per iteration
- `imp_proposal_scale` (1.0) — proposal variance multiplier
- `imp_seed` (42) — RNG seed for reproducibility
- `imp_tol` (1e-4) — convergence tolerance on relative OFV change
- `imp_e_only` (False) — if True, run E-step only (no parameter updates)

**Error Models** (`error_model=`):

| Model | Formula | Parameter(s) |
|-------|---------|-------------|
| `"additive"` | y = f + ε, ε ~ N(0, σ²) | `sigma` |
| `"proportional"` | y = f·(1 + ε), ε ~ N(0, σ²) | `sigma` |
| `"combined"` | y = f·(1 + ε₁) + ε₂, ε ~ N(0, σ²) | `sigma` (prop), `sigma_add` (add) |
| `"exponential"` | ln(y) = ln(f) + ε, ε ~ N(0, σ²) | `sigma` |
| `"power"` | y = f + f^γ · ε, ε ~ N(0, σ²) | `sigma`, power γ estimated |

**Random Effect Transforms** (`random_effect_transforms=`):

By default, individual parameters use log-normal transforms: θᵢ = θ_pop · exp(ηᵢ). For bounded parameters (e.g., bioavailability ∈ [0, 1]), use logit-normal:

```python
result = nextstat.nlme_foce(
    ...,
    random_effect_transforms=[
        "lognormal",  # CL: θ·exp(η) (default)
        "lognormal",  # V: θ·exp(η)
        {"type": "logit_normal", "lower": 0.0, "upper": 1.0},  # Ka bounded
    ],
)
```

**Dosing Regimens** (`regimens=`):

For multi-dose studies, pass per-subject dosing regimens instead of scalar `doses`:

```python
regimen = {
    "events": [
        {"time": 0.0, "amount": 100.0, "route": {"type": "oral", "bioavailability": 1.0}},
        {"time": 12.0, "amount": 100.0, "route": {"type": "oral", "bioavailability": 1.0}},
        {"time": 24.0, "amount": 50.0, "route": {"type": "iv_bolus"}},
    ]
}
result = nextstat.nlme_foce(
    ..., regimens=[regimen] * n_subjects,  # one per subject
)
```

Route types: `"iv_bolus"`, `"oral"` (with optional `bioavailability`), `"infusion"` (with `duration`).

**Inter-Occasion Variability (IOV)** (`iov=`):

For studies with multiple occasions (e.g., crossover), IOV adds per-occasion random effects:

```python
result = nextstat.nlme_foce(
    ...,
    model="1cpt_oral",
    iov={
        "param_indices": [0, 1],  # CL, V have IOV
        "occasion_start_times": [[0.0, 24.0]] * n_subjects,  # 2 occasions per subject
        "omega_iov_init": [0.1, 0.1],  # IOV SD for CL, V
    },
)
```

**Time-Varying Covariates** (`time_varying_covariates=`):

For covariates that change over time (e.g., body weight, renal function):

```python
result = nextstat.nlme_foce(
    ...,
    model="1cpt_oral",
    time_varying_covariates=[{
        "param_idx": 0,  # CL
        "trajectories": [
            [{"time": 0.0, "value": 70.0}, {"time": 24.0, "value": 72.0}],  # subject 0
            [{"time": 0.0, "value": 85.0}, {"time": 24.0, "value": 84.0}],  # subject 1
        ],
        "reference": 70.0,
        "relationship": "power",
        "exponent": 0.75,
        "interpolation": "locf",  # "locf" (last observation carried forward) or "linear"
    }],
)
```

**Covariance Step** (returned in `FoceResult` and `SaemResult`):

Both `nlme_foce` and `nlme_saem` compute the covariance step automatically when the model converges. Access via `result["covariance_step"]`:

```python
cov = result["covariance_step"]
if cov is not None:
    print("Standard errors:", cov["se"])
    print("RSE%:", cov["rse_pct"])
    print("R matrix:", cov["r_matrix"])          # Hessian of OFV
    print("S matrix:", cov["s_matrix"])          # Outer product of per-subject gradients
    print("Covariance:", cov["covariance"])      # R⁻¹
    print("Robust cov:", cov["robust_covariance"])  # R⁻¹ S R⁻¹ (sandwich)
    print("R eigenvalues:", cov["r_eigenvalues"])
    print("Condition number:", cov["r_condition_number"])
    print("Parameter names:", cov["parameter_names"])
```

#### Model Diagnostics

- `nextstat.pk_vpc(times, y, subject_idx, n_subjects, *, doses, bioavailability=1.0, theta, omega_matrix, error_model="proportional", sigma=0.1, sigma_add=None, n_sim=200, quantiles=None, n_bins=10, seed=42, pi_level=0.90) -> dict` — Visual Predictive Check. `doses` is per-subject (or length 1 for broadcast). Returns `bins` (list of per-bin quantile comparisons), `quantiles`, `n_sim`.

- `nextstat.pk_gof(times, y, subject_idx, *, doses, bioavailability=1.0, theta, eta, error_model="proportional", sigma=0.1, sigma_add=None) -> list[dict]` — Goodness of Fit. `doses` is per-subject (or length 1 for broadcast). Returns per-observation records with `subject`, `time`, `dv`, `pred`, `ipred`, `iwres`, `cwres`.

- `nextstat.pk_npde(times, y, subject_idx, n_subjects, *, doses, bioavailability=1.0, theta, omega_matrix, error_model="proportional", sigma=0.1, sigma_add=None, n_sim=1000, seed=42) -> dict` — Normalized Prediction Distribution Errors. Simulation-based GOF diagnostic: simulates `n_sim` replicates, computes per-observation NPDE by comparing observed DV to the simulated distribution, and returns `records` (list of per-observation dicts with `subject`, `time`, `dv`, `percentile`, `npde`) plus aggregate `mean` and `variance`. Well-calibrated model → NPDE ~ N(0,1), mean ≈ 0, variance ≈ 1.

#### Stepwise Covariate Modeling (SCM)

- `nextstat.scm(times, y, subject_idx, n_subjects, covariates, covariate_names, *, dose, bioavailability=1.0, error_model="proportional", sigma=0.1, sigma_add=None, theta_init, omega_init, param_names=None, relationships=None, forward_alpha=0.05, backward_alpha=0.01, max_outer_iter=100, max_inner_iter=20, tol=1e-4, rel_tol=0.0) -> ScmResult` — Stepwise Covariate Modeling (1-cpt oral). Forward selection + backward elimination of covariate–parameter relationships using ΔOFV (χ²(1) likelihood ratio test). `covariates` is a list of per-covariate observation vectors (each length `n_obs`); per-subject values extracted from first observation. Auto-centers at median. `relationships`: `"power"` (default), `"proportional"`, `"exponential"`.

  **`ScmResult` (TypedDict):**
  - `selected: list[ScmStepResult]` — final selected covariates
  - `forward_trace`, `backward_trace: list[ScmStepResult]`
  - `base_ofv`, `final_ofv: float`
  - `n_forward_steps`, `n_backward_steps: int`
  - `theta: list[float]`, `omega: list[list[float]]`

  **`ScmStepResult` (TypedDict):**
  - `name: str`, `param_index: int`, `relationship: str`
  - `delta_ofv: float`, `p_value: float`, `coefficient: float`, `included: bool`

#### MAP Estimation

- `nextstat.map_estimate(model, priors, *, max_iter=1000, tol=1e-8, compute_se=True, init=None) -> MapEstimateResult` — Maximum A Posteriori estimation for any `LogDensityModel`. `priors` is a list of `(mean, sd)` tuples defining Normal priors on each parameter. Useful for individual Bayesian PK estimation with population priors.

  **`MapEstimateResult` (TypedDict):**
  - `params: list[float]` — MAP point estimate
  - `se: list[float] | None` — posterior standard errors (if `compute_se=True`)
  - `nll_posterior: float` — negative log-posterior at MAP
  - `nll: float` — negative log-likelihood at MAP
  - `log_prior: float` — log-prior at MAP
  - `n_iter: int`, `converged: bool`
  - `param_names: list[str] | None`
  - `shrinkage: list[float] | None` — shrinkage toward prior mean

#### Bioequivalence

- `nextstat.average_be(test_values, ref_values, *, alpha=0.05, limits=(0.80, 1.25), design="2x2") -> BeResult` — Average bioequivalence (TOST) for 2×2 crossover design. Input values must be **log-transformed** (e.g. `ln(AUC)`). The `design` parameter selects the crossover layout.

  **`BeResult` (TypedDict):**
  - `geometric_mean_ratio: float`, `ci_lower: float`, `ci_upper: float`
  - `pe_log: float`, `se_log: float`, `df: float`
  - `t_lower: float`, `t_upper: float`, `p_lower: float`, `p_upper: float`
  - `conclusion: str` — `"bioequivalent"` or `"not bioequivalent"`

- `nextstat.be_power(n_total, *, cv=0.30, gmr=0.95, alpha=0.05, design="2x2") -> float` — Statistical power for an ABE study given sample size and intra-subject CV.

- `nextstat.be_sample_size(*, cv=0.30, gmr=0.95, target_power=0.80, alpha=0.05, design="2x2") -> BeSampleSizeResult` — Minimum sample size for target power.

  **`BeSampleSizeResult` (TypedDict):**
  - `n_per_sequence: int`, `n_total: int`, `achieved_power: float`

#### Clinical Trial Simulation

- `nextstat.simulate_trial(*, n_subjects, dose, obs_times, pk_model="1cpt_oral", theta, omega, sigma, error_model="proportional", bioavailability=1.0, seed=42) -> TrialSimResult` — Monte Carlo clinical trial simulation with population PK. Generates individual parameters from Ω, simulates concentrations with residual error. `pk_model`: `"1cpt_oral"`, `"2cpt_iv"`, `"2cpt_oral"`, `"3cpt_iv"`, `"3cpt_oral"`.

  **`TrialSimResult` (TypedDict):**
  - `concentrations: list[list[float]]` — per-subject concentration profiles
  - `individual_params: list[list[float]]` — per-subject PK parameters
  - `auc: list[float]`, `cmax: list[float]`, `tmax: list[float]`, `ctrough: list[float]`

#### Data I/O

- `nextstat.read_nonmem(csv_text) -> dict` — parse NONMEM-format CSV. Filters EVID=1 (dose) records, returns observation data. Returns `n_subjects`, `subject_ids`, `times`, `dv`, `subject_idx`.

- `nextstat.read_xpt(path) -> list[XptDataset]` — read CDISC .xpt (SAS Transport v5) file. Returns list of datasets. Pure Rust parser, no SAS dependency.

- `nextstat.write_xpt(path, datasets) -> None` — write CDISC .xpt (SAS Transport v5) file from Python data.

- `nextstat.xpt_to_nonmem(dataset) -> dict` — convert XPT dataset to NONMEM format by auto-detecting SDTM/ADaM columns (ID, TIME, DV, EVID, AMT, etc.).

  **`XptDataset` (TypedDict):**
  - `name: str`, `label: str`
  - `variables: list[XptVariable]` — column descriptors
  - `data: list[list[Any]]` — row-major data

  **`XptVariable` (TypedDict):**
  - `name: str`, `label: str`
  - `var_type: "numeric" | "character"`, `length: int`, `format: str`

### Test / utility models

- `GaussianMeanModel(y, sigma)` — simple Gaussian mean estimation.
- `FunnelModel(dim=2)` — Neal's funnel (sampler stress test). `dim` controls dimensionality: `y ~ N(0,3)`, `x_i|y ~ N(0, exp(y/2))` for `i = 1..dim-1`.
- `StdNormalModel(dim=2)` — standard normal (sampler validation).

---

## Optional modules

These modules live under `nextstat.*` as convenience helpers. Some require optional dependencies.

- `nextstat.viz` — plot-friendly artifacts + plotting helpers (CLs/profile/pulls/ranking/corr).
- `nextstat.bayes` — Bayesian helpers (ArviZ integration).
- `nextstat.ads` — ads-native observation and response helpers.
- `nextstat.torch` — PyTorch differentiable wrappers (see below).
- `nextstat.timeseries` — higher-level time series helpers and plotting.
- `nextstat.survival` — high-level survival helpers (parametric right-censoring + Cox PH).
- `nextstat.econometrics` — robust SE, FE baseline, DiD/event-study, IV/2SLS, and reporting.
- `nextstat.causal` — propensity + AIPW baselines and sensitivity hooks.
- `nextstat.volatility` — GARCH(1,1) and stochastic volatility convenience wrappers.
- `nextstat.gym` — Gymnasium/Gym environments for RL / design-of-experiments (requires `gymnasium` + `numpy`). See below.
- `nextstat.mlops` — fit metrics extraction for experiment loggers (W&B, MLflow, Neptune).
- `nextstat.interpret` — systematic-impact ranking as ML-style Feature Importance.
- `nextstat.glm` — regression/GLM convenience wrappers.
- `nextstat.ordinal` — ordinal regression convenience wrappers.
- `nextstat.formula` — Patsy-like formula interface (see below).
- `nextstat.ppc` — posterior predictive checks (see below).
- `nextstat.missing` — missing data helpers (see below).
- `nextstat.arrow_io` — Arrow/Parquet authoring helpers (schema validation + manifest writing; requires `pyarrow`).
- `nextstat.analysis` — TREx replacement workflow helpers (ROOT HIST ingest + expression eval; requires `numpy`, and `_core` for ROOT IO).
- `nextstat.trex_config` — TRExFitter `.config` parser + conversion to analysis spec (no native deps).
- `nextstat.audit` — reproducible local run bundles (no optional deps).
- `nextstat.report` — render report artifacts to PDF/SVG (requires `matplotlib`).
- `nextstat.validation_report` — render `validation_report.json` to PDF (requires `matplotlib`).
- `nextstat.viz_render` — render one artifact JSON (`pulls`/`corr`/`ranking`) directly to PNG/SVG/PDF (requires `matplotlib`).

### `nextstat.econometrics` (core functions)

Low-level econometrics functions exposed via `nextstat._core`:

- `nextstat._core.panel_fe(y, x, entity_ids, p, *, cluster_ids=None) -> dict` — panel fixed-effects regression. Returns `coefficients`, `se_ols`, `se_cluster`, `r_squared_within`, `n_obs`, `n_entities`, `n_regressors`, `rss`.
- `nextstat._core.did(y, treat, post, cluster_ids) -> dict` — difference-in-differences estimator. Returns `att`, `se`, `se_cluster`, `t_stat`, `mean_treated_post`, `mean_treated_pre`, `mean_control_post`, `mean_control_pre`, `n_obs`.
- `nextstat._core.event_study(y, entity_ids, time_ids, relative_time, min_lag, max_lag, reference_period, cluster_ids) -> dict` — event study with dynamic treatment effects. Returns `relative_times`, `coefficients`, `se_cluster`, `ci_lower`, `ci_upper`, `n_obs`, `reference_period`.
- `nextstat._core.iv_2sls(y, x_exog, k_exog, x_endog, k_endog, z, m, *, exog_names=None, endog_names=None, cluster_ids=None) -> dict` — instrumental variables / 2SLS. Returns `coefficients`, `names`, `se`, `se_cluster`, `n_obs`, `n_instruments`, `first_stage`.
- `nextstat._core.aipw_ate(y, treat, propensity, mu1, mu0, *, trim=0.01) -> dict` — augmented IPW average treatment effect. Returns `ate`, `se`, `ci_lower`, `ci_upper`, `n_treated`, `n_control`, `mean_propensity`.
- `nextstat.rosenbaum_bounds(y_treated, y_control, gammas) -> dict` — Rosenbaum bounds in the native extension (matched pairs). Returns `gammas`, `p_upper`, `p_lower`, `gamma_critical`.
- `nextstat.causal.aipw.rosenbaum_bounds(y_treated, y_control, *, gammas=None) -> RosenbaumBoundsResult` — Python convenience wrapper for the same bounds (matched pairs).

The `nextstat.econometrics` convenience module provides higher-level, dependency-light Python estimators and reporting helpers.

### `nextstat.glm` (convenience)

Convenience wrappers around `nextstat._core` regression models:

- `nextstat.glm.linear(x, y, *, intercept=True) -> FitResult`
- `nextstat.glm.logistic(x, y, *, intercept=True) -> FitResult`
- `nextstat.glm.poisson(x, y, *, intercept=True, offset=None) -> FitResult`
- `nextstat.glm.negbin(x, y, *, intercept=True, offset=None) -> FitResult`
- `nextstat.glm.gamma(x, y, *, intercept=True) -> FitResult`
- `nextstat.glm.tweedie(x, y, *, p=1.5, intercept=True) -> FitResult`

### `nextstat.ordinal` (convenience)

- `nextstat.ordinal.logit(x, y, *, n_levels) -> FitResult`
- `nextstat.ordinal.probit(x, y, *, n_levels) -> FitResult`

### `nextstat.formula` (Patsy-like interface)

- `nextstat.formula.parse_formula(formula_str) -> (str, list[str], bool)` — parse a minimal Wilkinson-style formula and return `(y_name, terms, include_intercept)`.
- `nextstat.formula.to_columnar(data, columns) -> Mapping[str, Sequence[Any]]` — normalize tabular inputs (dict-of-columns, list-of-dict rows, or pandas DataFrame) into a dict-of-columns view.
- `nextstat.formula.design_matrices(formula_str, data, *, categorical=None) -> (list[float], list[list[float]], list[str])` — build deterministic `(y, X, column_names)`.

### `nextstat.ppc` (posterior predictive checks)

- `nextstat.ppc.ppc_glm_from_sample(spec, sample_raw, *, param_names=None, n_draws=50, seed=0, stats_fn=None) -> PpcStats` — PPC from a raw `nextstat.sample(...)` dict for GLM specs.
- `nextstat.ppc.ppc_negbin_from_sample(spec, sample_raw, *, param_names=None, n_draws=50, seed=0, stats_fn=None) -> PpcStats` — PPC for NB2 (mean/dispersion) regression.
- `nextstat.ppc.ppc_ordered_from_sample(spec, sample_raw, *, param_names=None, n_draws=50, seed=0, stats_fn=None) -> PpcStats` — PPC for ordered outcomes (logit/probit).

### `nextstat.missing` (missing data)

- `nextstat.missing.apply_policy(x, y=None, *, policy="drop_rows") -> MissingResult` — apply an explicit missing-data policy (`"drop_rows"` or `"impute_mean"`).

### `nextstat.arrow_io` (Arrow/Parquet authoring, `pyarrow`)

Requires `pyarrow` (install with: `pip install "nextstat[io]"`).

- `nextstat.arrow_io.validate_histogram_table(table) -> HistogramTableStats` — validate the histogram table contract.
- `nextstat.arrow_io.validate_modifiers_table(table) -> ModifiersTableStats` — validate the modifiers table contract (binned Parquet v2).
- `nextstat.arrow_io.write_histograms_parquet(table, path, *, compression="zstd", write_manifest=True, manifest_path=None, poi="mu", observations=None, observations_path=None) -> dict` — write Parquet + optional manifest JSON; returns manifest dict.
- `nextstat.arrow_io.validate_histograms_parquet_manifest(manifest, *, check_sha256=True) -> None` — validate a manifest against the referenced Parquet file.
- `nextstat.arrow_io.load_parquet_as_histfactory_model(path, *, poi="mu", observations=None) -> HistFactoryModel` — validate Parquet schema with PyArrow, then call `nextstat.from_parquet()`.
- `nextstat.arrow_io.load_parquet_v2_as_histfactory_model(yields_path, modifiers_path, *, poi="mu", observations=None) -> HistFactoryModel` — validate Parquet v2 schemas, then call `nextstat.from_parquet_with_modifiers()`.
- `nextstat.arrow_io.validate_event_table(table) -> EventTableStats` — validate an unbinned event table contract.
- `nextstat.arrow_io.write_events_parquet(table, path, *, observables=None, compression="zstd") -> dict` — write an unbinned event table to Parquet with NextStat metadata.

Types: `HistogramTableStats`, `ModifiersTableStats`, `EventTableStats`. Constants: `HISTOGRAM_TABLE_MANIFEST_V1`, `UNBINNED_EVENTS_SCHEMA_V1`.

Contract references:

- Manifest schema: `docs/schemas/io/histograms_parquet_manifest_v1.schema.json`
- Canonical example: `docs/specs/histograms_parquet_manifest_v1.example.json`
- Example regen/check: `python scripts/generate_histograms_parquet_schema_examples.py [--check]`
- Family reproducibility gate: `python scripts/check_io_contracts.py --family histograms_parquet [--dry-run] [--report-json ...]`
- Aggregate runner report schema: `docs/schemas/io/nextstat_io_contract_runner_report_v1.schema.json`
- Narrative contract reference: `docs/references/arrow-parquet-io.md`

### `nextstat.analysis` (TREx replacement helpers)

- `nextstat.analysis.read_root_histogram(root_path, hist_path, *, flow_policy="drop") -> dict` — read one ROOT TH1 histogram and optionally fold under/overflow into the edge bins. Guarantees `sumw2` (and adds `sumw2_policy`) and adds `flow_policy` to the returned dict.
- `nextstat.analysis.read_root_histograms(root_path, hist_paths, *, flow_policy="drop") -> dict[str, dict]` — read many histograms from the same ROOT file.
- `nextstat.analysis.eval_expr(expr, env, *, n=None) -> np.ndarray` — vectorized TREx-style expression evaluation (requires `numpy`). `env` maps variable names to 1D NumPy arrays/scalars (and list-of-arrays for indexed vars like `jets_pt[0]`).

### `nextstat.trex_config` (TRExFitter config migration)

- `nextstat.trex_config.parse_trex_config(text, *, path=None) -> TrexConfigDoc` — parse a TREx `.config` file from a string.
- `nextstat.trex_config.parse_trex_config_file(path) -> TrexConfigDoc` — parse a TREx `.config` file from disk.
- `nextstat.trex_config.trex_doc_to_analysis_spec_v0(doc, *, source_path=None, out_path=None, threads=1, workspace_out="tmp/trex_workspace.json") -> (dict, dict)` — convert TREx config doc to analysis spec v0. Returns `(spec, report)`; unsupported keys are recorded in the report.
- `nextstat.trex_config.trex_config_file_to_analysis_spec_v0(config_path, *, out_path=None, threads=1, workspace_out="tmp/trex_workspace.json") -> (dict, dict)` — convenience wrapper: parse file -> convert -> `(spec, report)`.
- `nextstat.trex_config.dump_yaml(obj) -> str` — deterministic minimal YAML emitter (no external deps).

Types: `TrexConfigParseError`, `TrexConfigImportError`, `TrexConfigDoc`, `TrexConfigBlock`, `TrexConfigEntry`, `TrexValue`.

### `nextstat.audit` (run bundles)

- `nextstat.audit.environment_fingerprint() -> dict[str, Any]` — small, privacy-preserving environment fingerprint for reproducibility.
- `nextstat.audit.write_bundle(bundle_dir, *, command, args, input_path, output_value, tool_version=None, deterministic=False) -> None` — write a reproducible run bundle to a directory (Python mirror of CLI `--bundle`). When `deterministic=True`, timestamps are normalized to `0`.

Dataclasses: `BundleMeta`, `BundleInputMeta`.

### `nextstat.report` (rendering, matplotlib)

- `nextstat.report.render_report(input_dir, *, pdf, svg_dir, corr_include=None, corr_exclude=None, corr_top_n=None) -> None` — render a report PDF (+ optional per-plot SVG) from an artifacts directory (requires `matplotlib`).

CLI entry: `python -m nextstat.report render --input-dir ... --pdf ... [--svg-dir ...]`.

### `nextstat.viz_render` (single-artifact rendering, matplotlib)

- `nextstat.viz_render.render_artifact(kind, artifact, output, *, title=None, dpi=220, corr_include=None, corr_exclude=None, corr_top_n=None) -> None` — render one artifact dict (`pulls`/`corr`/`ranking`) to a file.

CLI entry: `python -m nextstat.viz_render render --kind pulls|corr|ranking --input artifact.json --output out.png`.

### `nextstat.validation_report` (rendering, matplotlib)

CLI entry: `python -m nextstat.validation_report render --input validation_report.json --pdf out.pdf` (requires `matplotlib`).

### `GpuFlowSession` (CUDA, feature-gated)

GPU-accelerated NLL reduction for unbinned models with neural PDFs:

- `nextstat._core.GpuFlowSession(n_events, n_params, processes, *, gauss_constraints=None, constraint_const=0.0)` — create session.
  - `nll(logp_flat, params) -> float` — compute NLL from pre-computed log-probabilities.
  - `nll_device_ptr_f32(d_logp_flat_ptr, params) -> float` — device-resident path (zero-copy from ONNX CUDA EP).
  - `compute_yields(params) -> list[float]` — yield computation.
  - `n_events() -> int`, `n_procs() -> int`, `n_params() -> int`

---
## Differentiable Layer (PyTorch integration)

The `nextstat.torch` module provides `torch.autograd.Function` wrappers for end-to-end differentiable HEP inference. Two backends:

### CPU path (no CUDA required)

- `nextstat.torch.NextStatQ0` — `torch.autograd.Function` for discovery q0. CPU profile fit with envelope-theorem gradient.
- `nextstat.torch.NextStatZ0` — same but returns discovery significance Z0.
- `nextstat.torch.discovery_z0(nominal, *, mle, model, channel, sample, eps=1e-12) -> torch.Tensor` — convenience wrapper returning differentiable Z0.

### CUDA path (zero-copy, requires `cuda` feature build)

- `nextstat.torch.create_session(model, signal_sample_name="signal") -> nextstat._core.DifferentiableSession` — create CUDA session for differentiable NLL (requires building with `--features cuda`).
- `nextstat.torch.nll_loss(signal_histogram, session, params=None) -> torch.Tensor` — differentiable NLL (zero-copy). `signal_histogram` must be contiguous CUDA `float64`.
- `nextstat.torch.create_profiled_session(model, signal_sample_name="signal", device="auto") -> nextstat._core.ProfiledDifferentiableSession | nextstat._core.MetalProfiledDifferentiableSession` — create GPU session for profiled test statistics.
- `nextstat.torch.profiled_q0_loss(signal_histogram, session) -> torch.Tensor` — differentiable profiled q₀.
- `nextstat.torch.profiled_z0_loss(signal_histogram, session, eps=1e-12) -> torch.Tensor` — differentiable Z₀ = sqrt(q₀).
- `nextstat.torch.profiled_qmu_loss(signal_histogram, session, mu_test) -> torch.Tensor` — differentiable profiled qμ.
- `nextstat.torch.batch_profiled_qmu_loss(signal_histogram, session, mu_values) -> list[torch.Tensor]` — qμ for multiple `mu_test` values.
- `nextstat.torch.profiled_zmu_loss(signal_histogram, session, mu_test, eps=1e-12) -> torch.Tensor` — differentiable Zμ = sqrt(qμ).
- `nextstat.torch.batch_profiled_q0_loss(signal_histograms, session) -> list[torch.Tensor]` — q₀ over a batch of histograms.
- `nextstat.torch.signal_jacobian(signal_histogram, session) -> torch.Tensor` — extract ∂q₀/∂signal without going through autograd.
- `nextstat.torch.signal_jacobian_numpy(signal_histogram, session) -> np.ndarray` — NumPy variant of `signal_jacobian`.

### Metal path (f32, requires `metal` feature build)

- `MetalProfiledDifferentiableSession(model, signal_sample_name)` — Metal GPU session for profiled test statistics (f32 compute). Signal is uploaded via CPU (no raw-pointer interop with MPS tensors).

High-level helpers:
- `nextstat.torch.SignificanceLoss(model, signal_sample_name="signal", *, device="auto", negate=True, eps=1e-12)` — ML-friendly callable loss wrapper around profiled Z₀.
- `nextstat.torch.SoftHistogram(bin_edges, bandwidth="auto", mode="kde")` — differentiable binning (KDE/sigmoid) to produce a signal histogram for `SignificanceLoss`.

## Evaluation Modes (Parity / Fast)

NextStat supports two evaluation modes for NLL computation, controllable at runtime:

```python
import nextstat

# Default: maximum speed (naive summation, SIMD/Accelerate/CUDA, multi-threaded)
nextstat.set_eval_mode("fast")

# Parity: deterministic (Kahan summation, Accelerate disabled, single-thread recommended)
nextstat.set_eval_mode("parity")

# Optional (recommended for parity in CI): force single-thread execution.
# Note: Rayon global thread pool can only be configured once per process,
# so call this early (before any parallel NextStat calls).
nextstat.set_threads(1)

# Query current mode
print(nextstat.get_eval_mode())  # "fast" or "parity"
```

| Mode | Summation | Backend | Use Case |
|------|-----------|---------|----------|
| `"fast"` | Naive | SIMD / Accelerate / CUDA | Production inference |
| `"parity"` | Kahan compensated | SIMD only | CI, pyhf parity validation |

**When to use parity mode:**
- Validating numerical results against pyhf NumPy backend
- CI regression tests requiring bit-exact reproducibility
- Debugging numerical discrepancies

**Tolerance contract** (Parity mode vs pyhf):
- Per-bin expected data: **1e-12** (bit-exact arithmetic)
- NLL value: **1e-10** absolute
- Gradient: **1e-6** atol + **1e-4** rtol (AD vs FD noise)
- Best-fit params: **2e-4** (optimizer surface)

Full 7-tier tolerance hierarchy: `tests/python/_tolerances.py`.

**Measured overhead:** <5% (Kahan vs naive at same thread count).

## Batch Toy Fitting

```python
import nextstat

model = nextstat.UnbinnedModel.from_config("spec.json")
params = model.suggested_init()

# CPU parallel: Rayon parallel
results = nextstat.fit_toys(model, params, n_toys=100, seed=42)

# Each result has: .parameters, .nll, .converged, .n_iter, .n_fev, .n_gev
converged = sum(1 for r in results if r.converged)
print(f"{converged}/{len(results)} toys converged")

# GPU: CUDA (f64) or Metal (f32)
results = nextstat.fit_toys(model, params, n_toys=1000, seed=42, device="cuda")
```

## GPU acceleration

- `nextstat.has_cuda() -> bool` — check CUDA availability.
- `nextstat.has_metal() -> bool` — check Metal availability (Apple Silicon).
- `nextstat.fit_toys(model, params, *, n_toys=1000, seed=42, device="cpu") -> list[FitResult]`:
  - `device="cuda"` — NVIDIA GPU, f64 precision, lockstep L-BFGS-B with fused NLL+gradient kernel.
  - `device="metal"` — Apple Silicon GPU, f32 precision, lockstep L-BFGS-B with Metal kernel. Tolerance relaxed to 1e-3.
  - `device="cpu"` (default) — CPU Rayon parallel, f64.
- `nextstat.ranking(model, *, device="cpu")` — GPU-accelerated ranking with `device="cuda"` or `device="metal"`.
- `nextstat.profile_scan(model, mu_values, *, device="cpu")` — GPU-accelerated profile scan with `device="cuda"`.

Build with GPU support:

```bash
cd bindings/ns-py

# CUDA (NVIDIA)
maturin develop --release --features cuda

# Metal (Apple Silicon)
maturin develop --release --features metal

# Both
maturin develop --release --features "cuda,metal"
```

## MLOps Integration (`nextstat.mlops`)

Lightweight helpers to extract NextStat metrics as plain dicts for experiment loggers.

- `nextstat.mlops.metrics_dict(fit_result, *, prefix="", include_time=True, extra=None) -> dict[str, float]` — flat dict from `FitResult`.
- `nextstat.mlops.significance_metrics(z0, q0=0.0, *, prefix="", step_time_ms=0.0) -> dict[str, float]`
- `nextstat.mlops.StepTimer` — lightweight wall-clock timer: `.start()`, `.stop() -> float` (ms).

## Interpretability (`nextstat.interpret`)

Systematic-impact ranking translated into ML-style Feature Importance.

- `nextstat.interpret.rank_impact(model, *, gpu=False, sort_by="total", top_n=None, ascending=False) -> list[dict]`
- `nextstat.interpret.rank_impact_df(model, **kwargs) -> pd.DataFrame` (requires `pandas`)
- `nextstat.interpret.plot_rank_impact(model, *, top_n=20, gpu=False, figsize=(8,6), title="Systematic Impact on Signal Strength (μ)", ax=None, **kwargs) -> matplotlib.Figure` (requires `matplotlib`)

## Agentic Analysis (`nextstat.tools`)

- `nextstat.tools.get_toolkit(*, transport="local", server_url=None, api_key=None, timeout_s=10.0) -> list[dict]`
- `nextstat.tools.get_toolkit_descriptor(*, transport="local", server_url=None, api_key=None, timeout_s=10.0) -> dict`
  - Returns the versioned discovery descriptor (`schema_version`, `transport`, `tools`, `capabilities`, `guidance`).
- `nextstat.tools.execute_tool(name, arguments, *, transport="local", server_url=None, api_key=None, timeout_s=30.0, fallback_to_local=True) -> dict`
- `nextstat.tools.execute_tool_raw(name, arguments) -> dict`
- `nextstat.tools.get_langchain_tools() -> list[StructuredTool]` (requires `langchain-core`)
- `nextstat.tools.get_mcp_tools() -> list[dict]`
- `nextstat.tools.handle_mcp_call(name, arguments) -> dict`
- `nextstat.tools.get_tool_names() -> list[str]`
- `nextstat.tools.get_tool_schema(name) -> dict | None`

Model-specific bootstrap packs for `Codex`, `Gemini`, `Ollama/local`, and IDE assistants live in `docs/references/agent-bootstrap.md`.

## Neural PDFs (`FlowPdf`, `DcrSurrogate`)

Standalone ONNX-backed normalizing flow and DCR surrogate classes. Requires building with `--features neural` (`maturin develop --release --features neural`).

## Surrogate Distillation (`nextstat.distill`)

Generate training datasets for neural likelihood surrogates. NextStat serves as the ground-truth oracle; the surrogate provides nanosecond inference.

- `nextstat.distill.generate_dataset(model, n_samples=100_000, *, method="sobol", bounds=None, seed=42, include_gradient=True, batch_size=10_000, gpu=False) -> SurrogateDataset` — sample parameter space and evaluate NLL + gradient at each point. Methods: `"sobol"`, `"lhs"`, `"uniform"`, `"gaussian"`.
  - `gpu` is currently reserved/ignored by the Python implementation.
- `nextstat.distill.SurrogateDataset` — container with `.parameters`, `.nll`, `.gradient` (NumPy arrays), `.parameter_names`, `.parameter_bounds`, `.metadata`.
- `nextstat.distill.to_torch_dataset(ds) -> TensorDataset` — convert for PyTorch DataLoader.
- `nextstat.distill.to_numpy(ds) -> dict[str, ndarray]` — export as dict of arrays.
- `nextstat.distill.to_npz(ds, path)` / `nextstat.distill.from_npz(path)` — save/load compressed `.npz`.
- `nextstat.distill.to_parquet(ds, path)` — export to Parquet (zstd, requires `pyarrow`).
- `nextstat.distill.train_mlp_surrogate(ds, *, hidden_layers=(256,256,128), epochs=100, lr=1e-3, batch_size=4096, val_fraction=0.1, grad_weight=0.1, device="cpu", verbose=True) -> nn.Module` — convenience MLP trainer with Sobolev loss.
- `nextstat.distill.predict_nll(surrogate, params_np) -> ndarray` — evaluate trained surrogate on raw parameters.

## Gymnasium / RL Environment (`nextstat.gym`)

Gymnasium-compatible environment for optimizing a single sample's nominal yields via reinforcement learning. Requires `gymnasium` (or legacy `gym`) + `numpy`.

- `nextstat.gym.HistFactoryEnv(*, workspace_json, channel, sample, reward_metric="nll", mu_test=5.0, max_steps=128, action_scale=0.05, action_mode="logmul", init_noise=0.0, clip_min=1e-12, clip_max=1e12, fixed_params=None, mle_max_iter=200, mle_tol=1e-6, mle_m=10)`
- `nextstat.gym.make_histfactory_env(workspace_json, *, channel, sample, reward_metric="nll", **kwargs) -> HistFactoryEnv`

Reward metrics: `"nll"`, `"q0"`, `"z0"`, `"qmu"`, `"zmu"`. Action modes: `"add"`, `"logmul"`.

## Arrow / Polars Integration

IPC-based interchange between NextStat and the Arrow columnar ecosystem (PyArrow, Polars, DuckDB, Spark). Backed by Rust `arrow` 57.3 + `parquet` 57.3 crates; the Python ↔ Rust bridge uses Arrow IPC.

See also: `docs/references/arrow-parquet-io.md` (schema contract + manifest for reproducible pipelines).

For Parquet/Arrow *authoring* from Python (schema validation + manifest writing): see `nextstat.arrow_io` (requires `pyarrow`).

### High-level API

- `nextstat.from_arrow(table, *, poi="mu", observations=None) -> HistFactoryModel` — create a model from a **PyArrow Table** or **RecordBatch** (requires `pyarrow`). The table must have columns `channel` (Utf8), `sample` (Utf8), `yields` (List\<Float64\>), optionally `stat_error` (List\<Float64\>). Works with any Arrow-compatible source (Polars, DuckDB, Spark).
- `nextstat.to_arrow(model, *, params=None, what="yields") -> pyarrow.Table` — export model data as a PyArrow table (requires `pyarrow`). `what="yields"` returns expected yields per channel; `what="params"` returns parameter metadata (name, index, value, bounds, init).
- `nextstat.from_parquet(path, poi="mu", observations=None) -> HistFactoryModel` — read a Parquet file directly (Snappy; Zstd support depends on build features), same schema as `from_arrow`.
- `nextstat.from_parquet_with_modifiers(yields_path, modifiers_path, poi="mu", observations=None) -> HistFactoryModel` — read **two** Parquet files: yields table + modifiers table (binned Parquet v2).

### Low-level IPC API

- `nextstat.from_arrow_ipc(ipc_bytes, poi="mu", observations=None) -> HistFactoryModel` — ingest raw Arrow IPC stream bytes.
- `nextstat.to_arrow_yields_ipc(model, params=None) -> bytes` — export yields as IPC bytes.
- `nextstat.to_arrow_params_ipc(model, params=None) -> bytes` — export parameters as IPC bytes.

```python
import pyarrow as pa
import nextstat

# From PyArrow
table = pa.table({
    "channel": ["SR", "SR", "CR"],
    "sample":  ["signal", "background", "background"],
    "yields":  [[5., 10., 15.], [100., 200., 150.], [500., 600.]],
})
model = nextstat.from_arrow(table, poi="mu")
result = nextstat.fit(model)

# From Polars
import polars as pl
df = pl.read_parquet("histograms.parquet")
model = nextstat.from_arrow(df.to_arrow())

# Export to Arrow
yields = nextstat.to_arrow(model, what="yields")
params = nextstat.to_arrow(model, what="params")
```

## Remote Server Client (`nextstat.remote`)

Pure-Python HTTP client for a remote `nextstat-server` instance. Zero native dependencies — only requires `httpx`.

### Core

- `nextstat.remote.connect(url, *, timeout=300.0) -> NextStatClient` — create a client.
- `client.fit(workspace=None, *, model_id=None, gpu=True) -> FitResult` — remote MLE fit. Pass `workspace` (dict or JSON string) or `model_id` from the model cache.
- `client.ranking(workspace=None, *, model_id=None, gpu=True) -> RankingResult` — remote nuisance-parameter ranking. For simplified-likelihood workspaces, entries rank reduced nuisance coordinates from the compiled model rather than source-level systematics.
- `client.health() -> HealthResult` — server health check (status, version, uptime, device, counters, cached_models).
- `client.close()` — close the connection. Also supports context manager (`with`).

### Batch API

- `client.batch_fit(workspaces, *, gpu=True) -> BatchFitResult` — fit multiple workspaces in one request. Returns `.results` (list of `FitResult | None`) and `.errors`.
- `client.batch_toys(workspace, *, params=None, n_toys=1000, seed=42, gpu=True) -> BatchToysResult` — GPU-accelerated toy fitting. Returns `.results` (list of `ToyFitItem`), `.n_converged`, `.n_failed`.

### Model Cache

- `client.upload_model(workspace, *, name=None) -> str` — upload a workspace to the server's model cache. Returns a `model_id` (SHA-256 hash).
- `client.list_models() -> list[ModelInfo]` — list cached models (model_id, name, n_params, n_channels, age_s, last_used_s, hit_count).
- `client.delete_model(model_id) -> bool` — evict a model from the cache.

Result types are typed dataclasses: `FitResult`, `RankingResult`, `RankingEntry`, `HealthResult`, `BatchFitResult`, `BatchToysResult`, `ToyFitItem`, `ModelInfo`.

Raises `NextStatServerError(status_code, detail)` on non-2xx HTTP responses.

```python
import nextstat.remote as remote

client = remote.connect("http://gpu-server:3742")

# Single fit
result = client.fit(workspace_json)
print(result.bestfit, result.nll, result.converged)

# Model cache — skip re-parsing on repeated calls
model_id = client.upload_model(workspace_json, name="my-analysis")
result = client.fit(model_id=model_id)  # ~4x faster

# Batch fit
batch = client.batch_fit([ws1, ws2, ws3])
for r in batch.results:
    print(r.nll if r else "failed")

# Batch toys (GPU-accelerated)
toys = client.batch_toys(workspace_json, n_toys=10_000, seed=42)
print(f"{toys.n_converged}/{toys.n_toys} converged in {toys.wall_time_s:.1f}s")

# Ranking
ranking = client.ranking(workspace_json)
for e in ranking.entries:
    print(f"{e.name}: Δμ = {e.delta_mu_up:+.3f} / {e.delta_mu_down:+.3f}")
```

## Survival analysis (non-parametric)

### `nextstat.kaplan_meier(times, events, *, conf_level=0.95) -> dict`

Kaplan-Meier survival estimator with Greenwood variance and log-log confidence intervals.

**Arguments:**
- `times` — list of observation times (≥ 0).
- `events` — list of booleans (`True` = event, `False` = right-censored).
- `conf_level` — confidence level for pointwise CIs (default 0.95).

**Returns** a dict with keys:
- `n`, `n_events`, `conf_level`, `median` (or `None`)
- `time`, `n_risk`, `n_event`, `n_censored` — per-step lists
- `survival`, `variance`, `ci_lower`, `ci_upper` — per-step lists

### `nextstat.log_rank_test(times, events, groups) -> dict`

Log-rank (Mantel-Cox) test comparing survival distributions of 2+ groups.

**Arguments:**
- `times` — list of observation times.
- `events` — list of booleans.
- `groups` — list of integer group labels (same length as `times`).

**Returns** a dict with keys:
- `n`, `chi_squared`, `df`, `p_value`
- `group_ids`, `observed`, `expected` — per-group lists

## Churn / Subscription vertical

### `nextstat.churn_generate_data(*, n_customers=2000, n_cohorts=6, max_time=24.0, treatment_fraction=0.3, seed=42) -> dict`

Generate a synthetic SaaS churn dataset (deterministic, seeded).

**Returns** a dict with keys: `n`, `n_events`, `times`, `events`, `groups`, `treated`, `covariates`, `covariate_names`, `plan`, `region`, `cohort`, `usage_score`.

### `nextstat.churn_retention(times, events, groups, *, conf_level=0.95) -> dict`

Cohort retention analysis: stratified KM + log-rank comparison.

**Returns** a dict with keys: `overall` (KM summary), `by_group` (list of per-group KM), `log_rank` (chi_squared, df, p_value).

### `nextstat.churn_risk_model(times, events, covariates, names, *, conf_level=0.95) -> dict`

Cox PH churn risk model with hazard ratios and CIs.

**Returns** a dict with keys: `n`, `n_events`, `nll`, `names`, `coefficients`, `se`, `hazard_ratios`, `hr_ci_lower`, `hr_ci_upper`.

### `nextstat.churn_uplift(times, events, treated, covariates, *, horizon=12.0) -> dict`

AIPW causal uplift estimate of intervention impact on churn.

**Returns** a dict with keys: `ate`, `se`, `ci_lower`, `ci_upper`, `n_treated`, `n_control`, `gamma_critical`, `horizon`.

### `nextstat.churn_diagnostics(times, events, groups, *, treated=[], covariates=[], covariate_names=[], trim=0.01) -> dict`

Data quality diagnostics and trust gate for churn analysis. Checks censoring rates per segment, covariate balance (SMD), propensity overlap, and sample-size adequacy.

**Returns** a dict with keys:
- `n`, `n_events`, `overall_censoring_frac`, `trust_gate_passed`
- `censoring_by_segment` — list of `{group, n, n_events, n_censored, frac_censored}`
- `covariate_balance` — list of `{name, smd_raw, mean_treated, mean_control}`
- `propensity_overlap` — `{quantiles, mean, n_trimmed_low, n_trimmed_high, trim}` or `None`
- `warnings` — list of `{category, severity, message}`

### `nextstat.churn_cohort_matrix(times, events, groups, period_boundaries) -> dict`

Life-table cohort retention matrix. For each cohort and period, computes at-risk count, events, censored, period retention rate, and cumulative retention.

**Arguments:**
- `period_boundaries` — sorted time points defining period ends (e.g. `[1, 3, 6, 12, 24]`).

**Returns** a dict with keys:
- `period_boundaries` — echo of input boundaries
- `cohorts` — list of `{cohort, n_total, n_events, periods}` where each period is `{n_at_risk, n_events, n_censored, retention_rate, cumulative_retention}`
- `overall` — same structure, all cohorts combined

### `nextstat.churn_compare(times, events, groups, *, conf_level=0.95, correction="benjamini_hochberg", alpha=0.05) -> dict`

Pairwise segment comparison with log-rank tests, hazard ratio proxies, and multiple comparisons correction.

**Arguments:**
- `correction` — `"benjamini_hochberg"` (alias `"bh"`) or `"bonferroni"`.

**Returns** a dict with keys:
- `overall_chi_squared`, `overall_p_value`, `overall_df`, `alpha`, `n`, `n_events`, `correction_method`
- `segments` — list of `{group, n, n_events, median, observed, expected}`
- `pairwise` — list of `{group_a, group_b, chi_squared, p_value, p_adjusted, hazard_ratio_proxy, median_diff, significant}`

### `nextstat.churn_uplift_survival(times, events, treated, *, covariates=[], horizon=12.0, eval_horizons=[3,6,12,24], trim=0.01) -> dict`

Survival-native causal uplift: RMST, IPW-weighted Kaplan-Meier, and ΔS(t) at specified horizons.

**Returns** a dict with keys:
- `rmst_treated`, `rmst_control`, `delta_rmst`, `horizon`, `ipw_applied`
- `arms` — list of `{arm, n, n_events, rmst, median}`
- `survival_diffs` — list of `{horizon, survival_treated, survival_control, delta_survival}`
- `overlap` — `{n_total, n_after_trim, n_trimmed, mean_propensity, min_propensity, max_propensity, ess_treated, ess_control}`

### `nextstat.churn_bootstrap_hr(times, events, covariates, names, *, n_bootstrap=1000, seed=42, conf_level=0.95, ci_method="percentile", n_jackknife=200) -> dict`

Bootstrap hazard ratios via parallel Cox PH refitting.

- `ci_method`: `"percentile"` (default) or `"bca"`.
- `n_jackknife`: number of leave-one-out fits for BCa acceleration (used when `ci_method="bca"`).

**Returns** a dict with keys:
- `names`, `hr_point`, `hr_ci_lower`, `hr_ci_upper`
- `n_bootstrap`, `n_jackknife_requested`, `n_jackknife_attempted`, `n_converged`, `elapsed_s`
- `ci_method_requested`, `ci_method_effective`
- `ci_diagnostics` (per coefficient): `requested_method`, `effective_method`, `z0`, `acceleration`, alpha fields, counts, and `fallback_reason`.

### `nextstat.churn_ingest(times, events, *, groups=None, treated=None, covariates=[], covariate_names=[], observation_end=None) -> dict`

Validate and ingest raw customer arrays into a clean churn dataset. Drops rows with invalid/missing times, applies observation-end censoring cap.

**Returns** a dict with keys: `n`, `n_events`, `times`, `events`, `groups`, `treated`, `covariates`, `covariate_names`, `n_dropped`, `warnings`.

## CLI parity

The CLI mirrors the core workflows for HEP (fit/hypotest/scan/limits), time series, survival, and churn analysis.
See `docs/references/cli.md`.
