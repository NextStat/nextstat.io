---
title: "RFC: Research-Grade Measurement Combination Mode (GVM)"
status: proposed
date: 2026-03-05
owners:
  - ns-inference maintainers
  - HEP maintainers
---

# RFC: Research-Grade Measurement Combination Mode (GVM)

## Status

Proposed.

Related user-facing docs:

- `docs/tutorials/hep-gvm-measurement-combinations.md`
- `docs/references/cli.md`
- `docs/references/python-api.md`

This RFC defines the product and implementation contract for a new research-grade
measurement-combination workflow in NextStat. It is intentionally narrower than a
general meta-analysis feature and intentionally separate from HistFactory workspace
fits.

## 1. Context and motivation

NextStat already supports:

- HistFactory-style binned likelihood inference
- pyhf-compatible workspace combination via `nextstat combine`
- profile likelihood fitting, toys, and HEP-focused visualization/reporting

What it does not yet support is a dedicated workflow for combining already reduced
scalar measurements when:

- the measurements have correlated statistical/systematic uncertainties
- the systematic assignments themselves are uncertain
- the analyst wants a likelihood-based combination rather than a purely covariance-only
  BLUE-style summary

The target use case is the Gamma Variance Model (GVM) setting discussed in:

- Enzo Canonero, Glen Cowan, "Correlated systematic uncertainties and errors-on-errors
  in measurement combinations with an application to the 7-8 TeV ATLAS-CMS top quark
  mass combination", Eur. Phys. J. C 85, 156 (2025)

This is a good fit for NextStat because:

- it is HEP-specific and likelihood-native
- it complements, rather than replaces, existing workspace-level workflows
- it benefits from NextStat's optimizer, profiling, deterministic JSON contracts, and
  Python/CLI wrappers

This is **not** a stable/common-path feature in v1. It is a research-grade mode with:

- explicit opt-in
- HEP-only scope
- literature-backed numeric validation
- no claim of production stability until parity gates pass

## 2. Decision

We will ship a new **research-grade measurement combination mode** with the following
product boundaries:

1. Scope is limited to **scalar measurement combinations**.
2. The canonical engine lives in **Rust first** under a new `ns-inference`
   module named `measurement_combine`.
3. User-facing access is provided through:
   - CLI: `nextstat combine-measurements`
   - CLI calibration: `nextstat combine-measurements-calibrate`
   - CLI calibration study: `nextstat combine-measurements-calibrate-study`
   - CLI scenario study: `nextstat combine-measurements-scenario-study`
   - CLI calibration campaign: `nextstat combine-measurements-calibration-campaign`
   - CLI solver-parity scenario study:
     `nextstat combine-measurements-solver-parity-scenario-study`
   - CLI solver-parity calibration campaign:
     `nextstat combine-measurements-solver-parity-calibration-campaign`
   - CLI cached solver-parity scenario study:
     `nextstat combine-measurements-solver-parity-scenario-study-from-reports`
   - CLI cached solver-parity calibration campaign:
     `nextstat combine-measurements-solver-parity-calibration-campaign-from-reports`
   - CLI solver-parity scenario-study digest:
     `nextstat combine-measurements-solver-parity-scenario-study-summarize`
   - CLI solver-parity calibration-campaign digest:
     `nextstat combine-measurements-solver-parity-calibration-campaign-summarize`
   - CLI calibration campaign digest:
     `nextstat combine-measurements-calibration-campaign-summarize`
   - CLI calibration campaign brief:
     `nextstat combine-measurements-calibration-campaign-brief`
   - CLI calibration campaign family report:
     `nextstat combine-measurements-calibration-campaign-family-report`
   - CLI calibration campaign family matrix:
     `nextstat combine-measurements-calibration-campaign-family-matrix`
   - CLI calibration campaign portfolio:
     `nextstat combine-measurements-calibration-campaign-portfolio`
   - CLI calibration campaign portfolio stability:
     `nextstat combine-measurements-calibration-campaign-portfolio-stability`
   - Python: `nextstat.hep.combine_measurements(...)`
   - Python calibration: `nextstat.hep.calibrate_measurements(...)`
   - Python calibration study: `nextstat.hep.calibrate_measurements_study(...)`
   - Python scenario study: `nextstat.hep.study_measurement_combination_scenarios(...)`
   - Python calibration campaign:
     `nextstat.hep.calibrate_measurement_combination_scenarios(...)`
   - Python solver-parity scenario study:
     `nextstat.hep.compare_measurement_combination_scenario_study_solvers(...)`
   - Python solver-parity calibration campaign:
     `nextstat.hep.compare_measurement_combination_calibration_campaign_solvers(...)`
   - Python cached solver-parity scenario study:
     `nextstat.hep.compare_measurement_combination_scenario_study_solver_reports(...)`
   - Python cached solver-parity calibration campaign:
     `nextstat.hep.compare_measurement_combination_calibration_campaign_solver_reports(...)`
   - Python solver-parity scenario-study renderer:
     `nextstat.hep.render_measurement_combination_scenario_study_solver_parity(...)`
   - Python solver-parity calibration-campaign renderer:
     `nextstat.hep.render_measurement_combination_calibration_campaign_solver_parity(...)`
   - Python solver-parity scenario-study digest:
     `nextstat.hep.summarize_measurement_combination_scenario_study_solver_parity(...)`
   - Python solver-parity scenario-study digest renderer:
     `nextstat.hep.render_measurement_combination_scenario_study_solver_parity_summary(...)`
   - Python solver-parity calibration-campaign digest:
     `nextstat.hep.summarize_measurement_combination_calibration_campaign_solver_parity(...)`
   - Python solver-parity calibration-campaign digest renderer:
     `nextstat.hep.render_measurement_combination_calibration_campaign_solver_parity_summary(...)`
   - Python calibration campaign digest:
     `nextstat.hep.summarize_measurement_combination_calibration_campaign(...)`
   - Python calibration campaign digest renderer:
     `nextstat.hep.render_measurement_combination_calibration_campaign_summary(...)`
   - Python calibration campaign brief builder:
     `nextstat.hep.build_measurement_combination_calibration_campaign_brief(...)`
   - Python calibration campaign brief renderer:
     `nextstat.hep.render_measurement_combination_calibration_campaign_brief(...)`
   - Python calibration campaign family report builder:
     `nextstat.hep.build_measurement_combination_calibration_campaign_family_report(...)`
   - Python calibration campaign family report renderer:
     `nextstat.hep.render_measurement_combination_calibration_campaign_family_report(...)`
   - Python calibration campaign family matrix builder:
     `nextstat.hep.build_measurement_combination_calibration_campaign_family_matrix(...)`
   - Python calibration campaign family matrix renderer:
     `nextstat.hep.render_measurement_combination_calibration_campaign_family_matrix(...)`
   - Python calibration campaign portfolio builder:
     `nextstat.hep.build_measurement_combination_calibration_campaign_portfolio(...)`
   - Python calibration campaign portfolio renderer:
     `nextstat.hep.render_measurement_combination_calibration_campaign_portfolio(...)`
   - Python calibration campaign portfolio stability builder:
     `nextstat.hep.build_measurement_combination_calibration_campaign_portfolio_stability(...)`
   - Python calibration campaign portfolio stability renderer:
     `nextstat.hep.render_measurement_combination_calibration_campaign_portfolio_stability(...)`
4. Existing `nextstat combine` semantics remain unchanged and continue to mean
   pyhf workspace merge.
5. v1 uses a **numerical likelihood-based reference path**. Research-grade
   Lawley/Bartlett diagnostics and toy-calibration reports are supported, while
   analytical acceleration and symbolic perturbative helpers remain deferred.
6. Delivery is governed by **strict TDD**. No implementation slice starts without a
   failing test.

This naming is chosen deliberately:

- `combine-measurements` is explicit and avoids overloading `nextstat combine`
- `nextstat.hep` keeps the Python surface domain-scoped and out of the top-level
  stable namespace

## 3. Public interfaces

### 3.1 Rust core module

Create a new module in `ns-inference`:

- `ns_inference::measurement_combine`

This module is the canonical engine and owns the public Rust types for the feature.

The RFC fixes the v1 public types to the following names:

- `MeasurementCombinationSpec`
- `MeasurementInput`
- `SystematicSource`
- `MeasurementCombinationResult`
- `ConfidenceInterval`
- `GoodnessOfFit`
- `ResearchDiagnostics`

Recommended semantic shape:

```rust
pub struct MeasurementCombinationSpec {
    pub schema_version: String,
    pub poi: String,
    pub measurements: Vec<MeasurementInput>,
    pub stat_covariance: Vec<Vec<f64>>,
    pub systematics: Vec<SystematicSource>,
}

pub struct MeasurementInput {
    pub name: String,
    pub value: f64,
}

pub struct SystematicSource {
    pub name: String,
    pub magnitudes: Vec<f64>,
    pub corr: Vec<Vec<f64>>,
    pub error_on_error: f64,
    pub aux_mean: f64,
}

pub struct MeasurementCombinationResult {
    pub poi: String,
    pub mu_hat: f64,
    pub confidence_interval: ConfidenceInterval,
    pub goodness_of_fit: GoodnessOfFit,
    pub converged: bool,
    pub stability: String,
    pub diagnostics: ResearchDiagnostics,
}
```

The exact field ordering may change, but these names and semantics are fixed by this RFC.

### 3.2 JSON spec contract

The canonical user artifact is a **dedicated measurement-combination spec**, not a
pyhf workspace.

Required top-level fields:

- `schema_version`
- `poi`
- `measurements[]`
- `stat_covariance`
- `systematics[]`

Each systematic source contains:

- `name`
- `magnitudes`
- `corr`
- `error_on_error`
- `aux_mean`

Defaults:

- `error_on_error` defaults to `0.0`
- `aux_mean` defaults to `0.0`

Schema rules that must be enforced in v1:

- `schema_version` must equal `nextstat_measurement_combination_v0`
- `measurements.len() >= 1`
- `stat_covariance` must be square with dimension `measurements.len()`
- `stat_covariance` must be symmetric
- every `systematics[k].magnitudes.len()` must equal `measurements.len()`
- every `systematics[k].corr` must be square with dimension `measurements.len()`
- every `systematics[k].corr` must be symmetric
- every diagonal of `systematics[k].corr` must equal `1` within numeric tolerance
- `error_on_error >= 0`

Research-grade published-data rule:

- raw published per-source correlation matrices may be non-PSD
- fixed-variance combinations may use such matrices directly when the aggregate
  covariance is positive semidefinite
- nuisance/GVM profiling must apply a documented regularization step before
  inverting a non-PSD per-source correlation matrix
- the applied per-source regularization shifts must be exposed in diagnostics

Canonical example:

```json
{
  "schema_version": "nextstat_measurement_combination_v0",
  "poi": "m_top",
  "measurements": [
    { "name": "ATLAS_ljets", "value": 172.08 },
    { "name": "CMS_dilepton", "value": 172.52 }
  ],
  "stat_covariance": [
    [0.04, 0.00],
    [0.00, 0.09]
  ],
  "systematics": [
    {
      "name": "bJES",
      "magnitudes": [0.32, 0.28],
      "corr": [
        [1.0, 0.72],
        [0.72, 1.0]
      ],
      "error_on_error": 0.30,
      "aux_mean": 0.0
    }
  ]
}
```

### 3.3 CLI

Add a new top-level command to the existing `nextstat` binary:

```bash
nextstat combine-measurements --input spec.json --output result.json
```

Add a companion research-grade calibration command:

```bash
nextstat combine-measurements-calibrate --input spec.json --output calibration.json --solver numerical|numerical-paper|analytic-perturbative|auto
```

For repeated-seed reproducibility studies, add an aggregate companion command:

```bash
nextstat combine-measurements-calibrate-study --input spec.json --output study.json --solver numerical|numerical-paper|analytic-perturbative|auto --seeds 42,43,44
```

For baseline-relative multi-scenario comparisons, add a scenario-study command:

```bash
nextstat combine-measurements-scenario-study --input spec.json --scenarios scenarios.json --output scenario-study.json --solver numerical|numerical-paper|analytic-perturbative|auto
```

Required v1 flags for `combine-measurements`:

- `--input <PATH>`: required
- `--output <PATH>`: optional, stdout when omitted
- `--ci-level <FLOAT>`: optional, default `0.68`
- `--solver <MODE>`: optional, default `auto`; supports `numerical`,
  `numerical-paper`, `analytic-perturbative`, and `auto`
- `--json-metrics <PATH>`: optional diagnostics output
- `--threads <N>`: optional, default `1`

Required CLI behavior:

- JSON in / JSON out only
- deterministic parity path supported with `--threads 1`
- command help and docs must mark it as `experimental` and `research-grade`
- no behavior change to current `nextstat combine`

Required result JSON fields:

- `poi`
- `mu_hat`
- `confidence_interval`
- `goodness_of_fit`
- `converged`
- `stability`
- optimizer diagnostics
- profiled nuisance/systematic diagnostics

The result payload must set:

- `stability: "research-grade"`
- `diagnostics.bartlett` when applicable
- `diagnostics.perturbative_validity`

Required calibration payload behavior:

- separate schema/version from the main result payload
- includes the fitted reference result used to generate toys
- records `n_toys` and `seed`
- reports empirical `q` / `q_star` moments and whether Bartlett improves the
  mean GOF agreement with the nominal chi-squared expectation

Required calibration-study payload behavior:

- separate schema/version from both the main result and one-seed calibration payloads
- includes the fitted reference result shared across the seed sweep
- records the full `seeds` list and `n_toys` per seed
- exposes `per_seed` calibration summaries
- exposes aggregate stability diagnostics for CI inflation and Bartlett behavior
- sets `stability: "research-grade"`

Required scenario-study payload behavior:

- separate schema/version from the main result and calibration payloads
- includes the baseline fit result for the unmodified input spec
- includes one result block per named scenario
- includes baseline-relative comparison diagnostics for each scenario
- includes an aggregate section that captures ordering and convergence signals
- sets `stability: "research-grade"`

### 3.4 Python

Add a new lazy-loaded Python submodule:

- `nextstat.hep`

The public entry point is:

```python
nextstat.hep.combine_measurements(spec_or_path, *, ci_level=0.68, solver="auto")
```

Companion calibration entry point:

```python
nextstat.hep.calibrate_measurements(
    spec_or_path, *, ci_level=0.68, solver="auto", n_toys=128, seed=42
)
```

Companion repeated-seed calibration entry point:

```python
nextstat.hep.calibrate_measurements_study(
    spec_or_path,
    *,
    ci_level=0.68,
    solver="auto",
    n_toys=128,
    seeds=[42, 43, 44],
)
```

Companion scenario-study entry point:

```python
nextstat.hep.study_measurement_combination_scenarios(
    spec_or_path,
    scenarios_or_path,
    *,
    ci_level=0.68,
    solver="auto",
)
```

Python v1 behavior:

- accepts either a parsed spec `dict`, a JSON string, or a filesystem path
- normalizes the input into the Rust-core JSON contract
- calls the same Rust engine used by the CLI
- returns a plain `dict` matching the CLI result schema
- mirrors the calibration report schema for the toy-calibration path
- mirrors the calibration-study schema for repeated deterministic seed sweeps
- mirrors the scenario-study schema for baseline-relative multi-scenario comparisons

Python namespace rules:

- do **not** add `combine_measurements` to top-level `nextstat`
- preserve the lazy submodule conventions in `nextstat.__init__`
- the submodule is domain-scoped because this is HEP-only and research-grade

## 4. Statistical model

### 4.1 Scope of the model

v1 solves the following problem:

- combine `N` scalar measurements of a common quantity `mu`
- include an arbitrary statistical covariance matrix
- include `M` systematic sources with per-source magnitudes and non-trivial
  cross-measurement correlations
- optionally model uncertainty in the assigned magnitude of each systematic source
  through an `error_on_error` parameter

This mode is not a HistFactory model and not a general-purpose random-effects
meta-analysis engine. It is a dedicated measurement-combination likelihood.

### 4.2 Public normalization convention

The public spec uses the normalized convention:

- the auxiliary variance estimate for each systematic source is fixed to `1`
- `magnitudes[i]` are the assigned systematic errors directly
- no public `v_s` field is exposed
- no alternative variance parameterizations are exposed in v1

This keeps the public surface small and matches the user mental model:

- statistical covariance comes from `stat_covariance`
- systematic size comes from `magnitudes`
- correlation shape comes from `corr`
- uncertainty in the systematic assignment comes from `error_on_error`

### 4.3 Canonical likelihood family

The v1 engine uses one unified likelihood family:

- when `error_on_error = 0`, it reduces to the ordinary nuisance-parameter
  combination limit
- when `error_on_error > 0`, it promotes each systematic source to a GVM-style
  uncertain-variance treatment

For a measurement vector `y`:

- `mu` is the parameter of interest
- `V_stat` is the user-provided statistical covariance matrix
- each systematic source `s` has:
  - magnitude vector `Gamma_s`
  - correlation matrix `R_s`
  - error-on-error parameter `epsilon_s`
  - auxiliary mean `u_s`, default `0`

Internal representation for each source `s`:

- one nuisance vector `theta_s` with one component per measurement
- covariance induced by the source is `diag(Gamma_s) * R_s * diag(Gamma_s)`
- a positive scalar variance-scale latent is introduced only when `epsilon_s > 0`

Implementation decision for v1:

- the Rust engine is allowed to evaluate and optimize the exact numerical reference
  likelihood directly
- no analytical profiling formulas are required in v1
- no perturbative Bartlett-corrected approximation is required in v1

This is the reference path because it is easier to validate, easier to debug under
TDD, and less likely to hide algebraic mistakes in a research-grade first release.

### 4.4 Reduction rules that must hold

The implementation must satisfy these exact reduction rules:

1. Single-measurement input returns that measurement value as `mu_hat`.
2. Purely statistical independent Gaussian inputs reduce to inverse-variance weighting.
3. `error_on_error = 0` reduces to the standard nuisance-parameter combination limit.
4. A zero-magnitude systematic source has no effect on the result.
5. Non-trivial per-source correlations affect only the source they belong to and
   do not implicitly create cross-source coupling.

### 4.5 Goodness-of-fit and interval contract

v1 result semantics:

- `mu_hat` is the MLE for the combined quantity
- `confidence_interval` is profile-likelihood based
- `goodness_of_fit` is reported for the combined fit using the same reference
  numerical path

Required confidence-interval behavior:

- support arbitrary `ci_level`
- default to `0.68`
- report lower bound, upper bound, and the `ci_level` used

Required goodness-of-fit behavior:

- report the test statistic
- report degrees-of-freedom convention used
- report p-value when available
- record whether the GOF is asymptotic or numeric in diagnostics

## 5. Non-goals

The following are explicitly out of scope for v1:

- generic cross-domain meta-analysis
- random-effects epidemiology/biostatistics feature work
- full HistFactory integration
- adding GVM semantics to pyhf modifiers
- automatic import from TRExFitter / HistFactory / RooFit combination artifacts
- symbolic BLUE/GVM algebra helpers as public API
- analytical Bartlett-corrected fast path
- perturbative closed-form profile approximations
- claims of long-term stable API status

## 6. TDD implementation plan

TDD policy is strict:

- no implementation slice starts without a failing test
- no wrapper work starts before Rust core tests pass
- no docs/API promotion starts before literature regression tests pass

The delivery order is fixed.

### Phase A: schema and validation

Write failing tests first for:

- malformed covariance shapes
- non-symmetric `stat_covariance`
- non-symmetric `corr`
- inconsistent measurement/systematic lengths
- invalid `error_on_error < 0`
- invalid non-unit correlation diagonal
- invalid non-PSD correlation matrix

Only after these tests exist may schema parsing/validation be implemented.

### Phase B: reduction and sanity behavior

Write failing tests first for:

- one-measurement pass-through
- independent Gaussian inverse-variance weighted pooling
- `error_on_error = 0` reproducing the standard nuisance-parameter limit

Only then implement the minimum numerical reference likelihood.

### Phase C: correlation behavior

Write failing tests first for:

- trivial correlations `0`, `+1`, `-1`
- one-source non-trivial correlation affecting the result in the expected direction
- rejection of non-symmetric and non-unit-diagonal per-source correlation matrices
- acceptance of raw published non-PSD per-source matrices when the aggregate
  covariance is valid, with surfaced regularization diagnostics for nuisance/GVM
  paths

Only then wire non-trivial correlations into the Rust engine.

### Phase D: GVM behavior

Write failing tests first for:

- increasing `error_on_error` weakens constraints relative to the standard case
- profiled systematic variance scales remain positive and finite
- outlier fixture reduces central-value sensitivity and widens the interval as
  `error_on_error` increases

Only then expose research diagnostics in the result object.

### Phase E: interfaces

Write failing interface tests first for:

- CLI golden JSON for `combine-measurements`
- Python smoke test for `nextstat.hep.combine_measurements(...)`
- CLI/Python parity on the same fixture

Only then wire the command and Python wrapper.

## 7. Validation and acceptance gates

This feature is not considered ready until all gates below pass.

### 7.1 Unit and contract gates

Must pass:

- Rust unit tests for validation, reduction, correlation, and GVM behavior
- CLI JSON golden test
- Python smoke/parity tests

### 7.2 Literature regression gate

Add a literature-backed fixture derived from the 2025 GVM paper:

- preferred: full ATLAS-CMS top-mass combination transcribed into the new
  measurement-combination spec from published tables and supplementary material
- fallback: a reduced published subset from the same paper if a full transcription
  is blocked during early slices

The fixture must be committed as a stable JSON input under `tests/fixtures/`.
For research-grade toy calibration, at least one full published correlated case
must also be committed as a stable calibration report artifact under
`tests/fixtures/`.

Acceptance thresholds for the literature regression:

- `mu_hat` absolute error <= `1e-3` in the paper's measurement units
- confidence-interval endpoint absolute error <= `2e-3`
- monotonic trend checks must pass for increasing `error_on_error`
- when Bartlett diagnostics are implemented, `diagnostics.bartlett` must remain
  finite on the published correlated fixtures and `diagnostics.perturbative_validity`
  must be exposed alongside them

For reduced published subsets built from rounded summary values rather than the
full unrounded combination inputs:

- the fixture must record the derivation source in the test name or companion
  note
- `mu_hat` tolerance may be relaxed up to `2e-2`
- confidence-interval half-width tolerance may be relaxed up to `1e-2`
- monotonic trend checks remain mandatory

If the published reference is only available as a trend/curve rather than a single
printed number, acceptance must be encoded as:

- digitized target points committed alongside the fixture
- absolute deviation <= `2e-3` on the digitized central-value/intervallized outputs

For full published toy-calibration artifacts:

- the committed report must include fixed `n_toys` and `seed`
- `mean_sigma_star > mean_sigma` must hold
- single-systematic calibration fixtures may require `sigma_star_ge_sigma_fraction >= 0.99`
- multi-scenario campaign fixtures must instead lock explicit per-scenario lower bounds,
  because published grouped-systematic studies can show near-neutral Bartlett inflation
  for some scenarios while still remaining deterministic and scientifically consistent
- published calibration campaigns must commit both the scenario config and the resulting
  campaign report as versioned research artifacts
- repeated-seed slow gates must lock the acceptable drift on
  `mean_sigma_star_to_sigma_ratio` and related CI-inflation summaries

### 7.3 Determinism gate

With `--threads 1`, repeated runs on the same fixture must produce byte-identical
JSON apart from approved timestamp/metrics fields.

### 7.4 Research-grade labeling gate

Before any user-facing docs mention this feature outside the RFC:

- CLI help must label it experimental/research-grade
- Python docstrings must label it research-grade
- output payload must carry `stability: "research-grade"`

## 8. Follow-up phase: analytical acceleration and Bartlett corrections

Once the reference path is validated, phase 2 may add:

- analytical profiling for nuisance and variance-scale terms
- optional fast-path selection when the approximation is numerically validated on
  the v1 fixtures

Bartlett status after the current research-grade slices:

- Lawley/Bartlett diagnostics are implemented for both trivial and non-trivial
  correlated GVM cases
- calibration campaigns also support a pure post-processing comparison layer:
  single-artifact digests from `combine-measurements-calibration-campaign-summarize`
  can be merged into a versioned multi-artifact brief for cross-fixture review
- briefs can in turn be merged into a versioned family-level report for
  cross-family review without rerunning any fit or toy-calibration workload
- family-level reports can be transformed into a deterministic dominance matrix
  for machine-readable cross-family rankings and pairwise comparison gates
- multiple family matrices can be merged into a portfolio-level report for
  deterministic cross-campaign comparison across scenario sets or seed grids
- multiple portfolio artifacts can be merged into a stability report for
  deterministic cross-run ordering checks across seed-grid or campaign variants
- perturbative-validity diagnostics are exposed to help assess when those
  approximations are on solid footing
- a slow toy/MC calibration gate is required to demonstrate that the corrected
  GOF statistic behaves more chi-squared-like than the raw proxy on at least one
  representative GVM fixture
- for full published correlated fixtures, slow toy/MC gates must also verify
  seed-stable CI inflation behavior even when GOF-side Bartlett improvement is
  not used as the primary acceptance signal

Phase 2 rules:

- the numerical reference path remains the source of truth
- fast paths are optional and must be parity-checked against the reference path
- no fast path becomes default until its tolerances are locked by tests

## Consequences

### Positive

- gives NextStat a dedicated HEP measurement-combination workflow
- extends likelihood-native inference into an area where BLUE-only tooling is common
- keeps risky research semantics out of the stable top-level API surface
- creates a disciplined TDD path toward future analytical acceleration

### Negative

- adds a specialized HEP feature that many NextStat users will never touch
- introduces a new spec and validation surface
- requires literature transcription and numeric acceptance maintenance
- creates another research-grade workflow that must be clearly labeled to avoid
  over-promising stability

## Related artifacts

- CLI reference: `docs/references/cli.md`
- Python API reference: `docs/references/python-api.md`
- Rust API reference: `docs/references/rust-api.md`
- Existing workspace combination command: `crates/ns-cli/src/main.rs`
- Existing Python workspace combine binding: `bindings/ns-py/python/nextstat/_core.pyi`
- Related paper: EPJC 2025 GVM combinations paper cited above
