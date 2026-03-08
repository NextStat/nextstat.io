---
title: "RFC: Simplified Likelihoods for HEP Reinterpretation"
status: proposed
date: 2026-03-06
owners:
  - ns-translate maintainers
  - ns-inference maintainers
  - HEP maintainers
---

# RFC: Simplified Likelihoods for HEP Reinterpretation

## Status

Proposed.

Related local docs:

- `docs/pyhf-parity.md`
- `docs/trexfitter-parity.md`
- `docs/rfcs/research-grade-measurement-combinations.md`
- `docs/tutorials/hep-full-workflow.md`
- `docs/references/cli.md`
- `docs/references/python-api.md`

External context used for scope selection (current as of March 2026):

- ATLAS full-likelihood publication for reinterpretation:
  https://cds.cern.ch/record/2684863
- ATLAS simplified-likelihood implementation note:
  http://cds.cern.ch/record/2782654
- Simplified-likelihood framework paper:
  https://arxiv.org/abs/2301.05676
- CMS public results that publish covariance matrices or simplified-binning
  reinterpretation material:
  https://cms-results.web.cern.ch/cms-results/public-results/publications/SUS-15-005/
  https://cms-results.web.cern.ch/cms-results/public-results/publications/SUS-21-002/

This RFC intentionally treats the March 2026 ecosystem as already mature enough
that a simplified-likelihood exchange surface is a product feature, not a
research toy.

## 1. Context and motivation

NextStat already has the major building blocks needed for reinterpretation:

- full pyhf JSON support
- native HS3 support
- HEPData-style patchset support
- asymptotic and toy-based CLs
- full HistFactory fitting, profiling, and visualization
- covariance-oriented HEP math in `measurement_combine`

What it does not yet have is a dedicated path for reduced public likelihoods
that sits between:

- a full HistFactory workspace with O(10^2) nuisance parameters
- and a pure covariance-only recast that has no likelihood structure left

That gap matters because full reinterpretation workflows are often dominated by:

1. repeated model loading and translation
2. repeated NLL/gradient evaluation in a workspace whose nuisance basis is much
   larger than the public reinterpretation task actually needs
3. publication formats that differ between ATLAS-style HistFactory/pyhf releases
   and CMS-style covariance/super-signal-region releases

The practical March 2026 opportunity is:

- experiments already publish or discuss simplified likelihoods as a useful
  reinterpretation artifact
- ATLAS explicitly frames simplified likelihoods as HistFactory-compatible
  reduced models
- CMS regularly publishes covariance matrices and reduced reinterpretation
  material that can be lifted into the same user-facing surface
- NextStat already has enough translation and inference infrastructure that
  the feature should be Low-Med effort, not a multi-quarter rewrite

## 2. Problem statement

We need a feature that lets users and experiments do all of the following with
one coherent contract:

1. ingest a published simplified likelihood
2. run `fit`, `scan`, `hypotest`, and `upper-limit` on it
3. derive a simplified likelihood from a full workspace for public release or
   internal recasting campaigns
4. preserve deterministic diagnostics about what fidelity was lost
5. obtain a real speedup for reinterpretation workloads

The success metric is not exact equivalence to the parent full workspace.
The success metric is:

- materially smaller artifact
- materially fewer effective nuisance parameters
- materially faster limit setting
- with explicit fidelity diagnostics and bounded error

For product planning, the target is:

- default `>= 5x` end-to-end speedup on medium public reinterpretation fixtures
- realistic `~10x` on reduced-basis workloads
- no silent degradation of statistical meaning

The `~10x` number is a delivery target for NextStat's own path. It is an
inference from the existing codebase and published simplified-likelihood use
cases, not a claim copied from a single source.

## 3. Decision

We will add a new HEP feature family named Simplified Likelihoods with the
following v1 design:

1. The canonical artifact is a new JSON schema:
   `nextstat_simplified_likelihood_v0`.
2. The canonical implementation lives in `ns-translate`, not `ns-inference`.
3. In v1, simplified likelihoods are compiled into a reduced
   `HistFactoryModel` and then flow through the existing inference stack.
4. The reduced model uses linearized shape modifiers (`histosys` semantics with
   `Code0`) to preserve the simplified-likelihood linear-order contract.
5. The feature supports two public input forms:
   - basis form: already reduced to explicit nuisance components
   - covariance form: only a covariance matrix is given, so NextStat
     factorizes it into a nuisance basis with diagnostics
6. v1 supports consume and derive/export workflows.
7. v1 explicitly does not promise a universal standalone
   `SimplifiedLikelihoodModel` in core inference.

This is the key architecture choice:

- a standalone model type would be cleaner mathematically
- but it would duplicate large parts of the current CLs/profile/viz plumbing
- and that pushes the feature out of the intended Low-Med effort band

By compiling to a reduced HistFactory model we reuse:

- `MaximumLikelihoodEstimator`
- profile likelihood scans
- asymptotic CLs and upper limits
- toy-based workflows
- existing CLI output contracts
- most HEP visualization/reporting

The standalone model path is deferred until one of these becomes true:

- the conversion layer itself becomes the performance bottleneck
- linearized `histosys` encoding is shown to be insufficient on real public
  releases
- we need functionality that cannot be expressed as reduced HistFactory at all

## 4. Product boundaries

### 4.1 What v1 will support

- load simplified-likelihood JSON directly in model-based CLI commands
- validate and audit simplified-likelihood artifacts
- convert simplified-likelihood JSON to a materialized reduced pyhf workspace
- derive a simplified-likelihood artifact from a full workspace plus fit result
- run standard frequentist workflows on the reduced model
- expose factorization and fidelity diagnostics as first-class JSON

### 4.2 What v1 will not support

- automatic publication-ready signal patchsets for simplified likelihoods
- non-Gaussian reduced nuisance priors beyond the current Gaussian linearized
  contract
- exact reproduction of every full-workspace ranking/correlation artifact
- arbitrary multi-POI reduced likelihoods
- a new generic asymptotic CLs trait hierarchy in `ns-core`

### 4.3 Why this scope is correct

This scope captures the main reinterpretation value now while staying inside
the engineering budget implied by "Low-Med":

- the hard problem is exchange format plus reduction diagnostics
- the fit/test-stat machinery already exists
- signal-patch publication can follow once the base artifact is stable

## 5. Public interfaces

### 5.1 New translation module

Create a new module family:

- `crates/ns-translate/src/simplified/mod.rs`
- `crates/ns-translate/src/simplified/schema.rs`
- `crates/ns-translate/src/simplified/validate.rs`
- `crates/ns-translate/src/simplified/factorize.rs`
- `crates/ns-translate/src/simplified/convert.rs`
- `crates/ns-translate/src/simplified/export.rs`

Update top-level exports:

- `crates/ns-translate/src/lib.rs`
- JSON format auto-detection in `hs3/detect.rs` or a renamed shared detector

### 5.2 Canonical JSON contract

The public artifact name is:

- `nextstat_simplified_likelihood_v0`

Recommended top-level shape:

```json
{
  "schema_version": "nextstat_simplified_likelihood_v0",
  "metadata": {
    "experiment": "ATLAS|CMS|external",
    "analysis_id": "analysis-name",
    "source_format": "basis|covariance|derived_from_workspace",
    "reference": "doi|cds|hepdata|internal",
    "description": "optional free text"
  },
  "poi": {
    "name": "mu",
    "init": 1.0,
    "bounds": [0.0, 10.0]
  },
  "bins": [
    { "channel": "SR", "name": "bin0" },
    { "channel": "SR", "name": "bin1" }
  ],
  "observed": [12.0, 8.0],
  "background_nominal": [10.5, 7.8],
  "signal_nominal": [1.8, 0.9],
  "uncertainty_model": {
    "kind": "basis",
    "components": [
      {
        "name": "sl_np_000",
        "hi": [11.1, 8.2],
        "lo": [9.9, 7.4]
      }
    ]
  },
  "diagnostics": {
    "factorization": null,
    "fidelity": null
  }
}
```

Notes:

- `signal_nominal` is optional.
- v1 is intentionally single-signal-template per artifact.
- base-artifact plus patchset publication is deferred.
- bins are globally flattened, but channel labels are preserved for later
  reconstruction into per-channel HistFactory layout.

### 5.3 Uncertainty model variants

#### Basis form

Used when the publisher already provides a reduced nuisance basis.

```json
"uncertainty_model": {
  "kind": "basis",
  "components": [
    { "name": "np0", "hi": [...], "lo": [...] },
    { "name": "np1", "hi": [...], "lo": [...] }
  ]
}
```

Semantics:

- each component is a Gaussian-constrained nuisance parameter
- `hi` and `lo` are the `+1 sigma` and `-1 sigma` background templates
- conversion must use `histosys` with `Code0`

#### Covariance form

Used for CMS-style public releases that provide only reduced background
covariance information.

```json
"uncertainty_model": {
  "kind": "covariance",
  "total_covariance": [
    [0.50, 0.12],
    [0.12, 0.30]
  ],
  "stat_covariance": [
    [0.10, 0.00],
    [0.00, 0.08]
  ]
}
```

Semantics:

- `total_covariance` is required
- `stat_covariance` is optional
- if both are present, the reducible shared systematic covariance is
  `total_covariance - stat_covariance`
- the conversion layer factorizes the covariance into a basis of Gaussian
  nuisance components and records the residual

### 5.4 Validation rules

The validator must enforce:

- `schema_version == "nextstat_simplified_likelihood_v0"`
- `bins.len() >= 1`
- `observed.len() == bins.len()`
- `background_nominal.len() == bins.len()`
- if `signal_nominal` is present, `signal_nominal.len() == bins.len()`
- all template vectors in basis form have the same length as `bins`
- covariance matrices are square with dimension `bins.len()`
- covariance matrices are symmetric within tolerance
- diagonal entries are non-negative
- basis templates do not produce negative expectations below the configured
  floor after conversion

Non-PSD handling:

- tiny negative eigenvalues from numeric noise may be clipped
- materially non-PSD covariance must fail validation unless the user opts into
  explicit regularization
- all regularization actions must be surfaced in diagnostics

## 6. Conversion contract

### 6.1 Simplified likelihood -> reduced HistFactory

The v1 reduced model is constructed as:

1. one HistFactory channel per unique `bins[].channel`
2. one background sample named `total_background`
3. optional one signal sample named `signal`
4. one unconstrained `normfactor` POI named from `poi.name`
5. one `histosys` modifier per reduced nuisance component

Important implementation rule:

- simplified-likelihood modifiers must always use `HistoSysInterpCode::Code0`
  regardless of the global `--interp-defaults`

Reason:

- simplified likelihoods are linearized by construction
- `Code0` preserves the intended piecewise-linear interpolation exactly
- `Code4p` would silently change the public contract

### 6.2 Covariance factorization

Default factorization algorithm:

1. symmetrize the covariance numerically:
   `C <- 0.5 * (C + C^T)`
2. if needed, regularize to nearest PSD within configured tolerance
3. perform eigen-decomposition
4. drop eigenmodes below absolute and relative thresholds
5. optionally truncate by explained variance
6. construct basis shifts from `sqrt(lambda_k) * v_k`

Public diagnostics must include:

- factorization method
- original rank
- retained rank
- explained variance fraction
- Frobenius residual norm
- any eigenvalue clipping or PSD repair

This is where existing covariance-heavy HEP code in `measurement_combine` can
be reused for linear algebra utilities and diagnostics conventions.

### 6.3 Full workspace -> simplified likelihood export

Add a new CLI/export path that derives a simplified artifact from a full
HistFactory workspace:

```bash
nextstat simplify workspace \
  --input workspace.json \
  --fit fit.json \
  --output simplified.json \
  --regions SR1,SR2 \
  --basis eigen \
  --explained-variance 0.995
```

The export algorithm is:

1. choose bins/regions to preserve in the reduced artifact
2. evaluate postfit background nominal `b_hat` in those bins
3. compute the nuisance covariance `Sigma_theta` from the fit result
4. compute the Jacobian of background yields wrt nuisance parameters:
   `J_{i,p} = d b_i / d theta_p`
5. build reduced covariance:
   `C_b = J Sigma_theta J^T`
6. optionally split `stat_covariance` from the rest when identifiable
7. factorize to the requested basis
8. emit diagnostics comparing full vs reduced behavior

v1 implementation detail:

- `J` may be finite-difference initially
- exact AD-based Jacobians are a follow-up optimization, not a blocker

### 6.4 Fidelity diagnostics

Every derived artifact must carry a fidelity report with at least:

- `nuisance_count_full`
- `nuisance_count_reduced`
- `bins_count`
- `relative_background_cov_residual`
- `max_abs_expected_delta_at_nominal`
- `max_abs_expected_delta_random_draws`
- `qmu_delta_smoke`
- `upper_limit_ratio_smoke`

These diagnostics are not optional. Reduced likelihoods without fidelity
metadata are not acceptable for publication or internal benchmarking.

## 7. CLI and Python surface

### 7.1 CLI

Add:

- `nextstat simplify workspace --input ... --fit ... --output simplified.json`
- `nextstat import simplified-likelihood --input simplified.json --output workspace.json`
- `nextstat audit --input simplified.json`

Extend existing model-based commands so they accept simplified-likelihood JSON:

- `nextstat fit --input simplified.json`
- `nextstat scan --input simplified.json`
- `nextstat hypotest --input simplified.json`
- `nextstat upper-limit --input simplified.json`

Auto-detection must distinguish among:

- pyhf JSON
- HS3 JSON
- simplified-likelihood JSON

### 7.2 Python

Add HEP-scoped helpers:

- `nextstat.hep.simplify_workspace(...) -> dict`
- `nextstat.hep.simplified_to_workspace(spec_or_path) -> dict`

Optional thin loader:

- `nextstat.hep.from_simplified(spec_or_path) -> HistFactoryModel`

The Python scope stays under `nextstat.hep` because this is a HEP-specific
exchange format, not a cross-domain core API.

## 8. TDD delivery plan

No implementation slice starts without a failing test. The intended order is:

### Slice 0: Schema and detection

Tests first:

- format detector recognizes simplified-likelihood JSON
- validator rejects wrong lengths, malformed covariance, and unknown kinds
- minimal basis-form fixture converts to internal spec struct

Implementation after red:

- schema types
- detection
- validation

### Slice 1: Basis-form conversion

Tests first:

- `sl_basis_two_bin.json` converts to a `HistFactoryModel`
- POI and reduced nuisance names are preserved
- `expected_data()` on the reduced model matches the simplified templates at
  `alpha = -1, 0, +1`

Implementation after red:

- basis-form converter
- forced `Code0` interpolation

### Slice 2: Covariance-form factorization

Tests first:

- exact PSD covariance round-trips into a basis with residual below tolerance
- tiny negative eigenvalues are clipped and logged
- materially non-PSD covariance fails without explicit regularization
- explained-variance truncation reduces retained components deterministically

Implementation after red:

- covariance factorizer
- diagnostics structs

### Slice 3: CLI consume path

Tests first:

- `nextstat fit --input simplified.json` succeeds on a small fixture
- `nextstat upper-limit --input simplified.json` succeeds and returns JSON
- `nextstat import simplified-likelihood` materializes a reduced workspace

Implementation after red:

- CLI plumbing
- input dispatch

### Slice 4: Workspace simplification export

Tests first:

- simplifying `tests/fixtures/simple_workspace.json` emits valid simplified JSON
- derived nominal yields match full workspace postfit yields within tolerance
- fidelity diagnostics are present and numerically stable

Implementation after red:

- exporter
- Jacobian/covariance construction

### Slice 5: Statistical fidelity gates

Tests first:

- `mu_hat` reduced/full agreement on selected fixtures
- `q_mu` agreement on selected scan points
- observed upper-limit ratio stays within configured tolerance

Recommended initial gates:

- `|mu_hat_sl - mu_hat_full| <= 0.05 sigma_full`
- `max |q_mu_sl - q_mu_full| <= 0.1`
- `upper_limit_sl / upper_limit_full` in `[0.95, 1.05]`

These are starting gates, not immutable truths. They must be calibrated on
public fixtures before being frozen.

### Slice 6: Performance gates

Tests first:

- medium workspace simplification reduces nuisance count materially
- medium reinterpretation benchmark shows wall-time speedup

Recommended initial gates:

- retained reduced components `<= 0.25 * full nuisance count`
- simplified JSON bytes `<= 0.35 * full workspace bytes`
- end-to-end upper-limit wall-time speedup `>= 3x` in CI smoke

Non-CI benchmark target:

- `~10x` on a public reinterpretation-style benchmark with O(50-200) original
  nuisance parameters and O(10-30) retained components

## 9. Test fixture plan

Add fixtures under `tests/fixtures/`:

- `sl_basis_two_bin.json`
- `sl_covariance_three_bin.json`
- `sl_covariance_non_psd.json`
- `sl_expected_roundtrip_workspace.json`

Add integration coverage:

- `crates/ns-translate/src/simplified/tests.rs`
- `crates/ns-cli/tests/cli_simplified_likelihood.rs`

Add a benchmark/script path:

- `tests/apex2_simplified_likelihood_report.py`

That Apex2 report should summarize:

- schema validity
- factorization residuals
- full-vs-simplified fidelity
- speedup summary

## 10. Risks and mitigations

### Risk: linearized model silently leaves its validity region

Mitigation:

- require fidelity diagnostics on export
- keep basis size and explained variance explicit
- document that simplified artifacts are local approximations around the chosen
  fit point

### Risk: covariance-only releases lose nuisance semantics

Mitigation:

- treat covariance-form ranking and per-source diagnostics as unsupported or
  approximate
- expose this clearly in the audit output

### Risk: non-PSD public covariance matrices

Mitigation:

- fail fast by default
- allow explicit `--regularize nearest-psd`
- record every repair in diagnostics

### Risk: `Code4p` or ROOT-style interpolation accidentally changes the contract

Mitigation:

- hard-force `Code0` for simplified-likelihood modifiers
- add explicit regression tests for `alpha = -1, 0, +1`

### Risk: trying to solve signal patch publication in v1 blows up scope

Mitigation:

- keep v1 on fully materialized S+B simplified artifacts
- defer simplified patchsets to a follow-up RFC

## 11. Acceptance criteria

This RFC is considered implemented when all of the following are true:

1. simplified-likelihood JSON is a first-class accepted input for relevant CLI
   workflows
2. both basis-form and covariance-form artifacts are supported
3. a full workspace can be reduced into a simplified artifact with explicit
   fidelity diagnostics
4. CI has deterministic coverage for schema, conversion, and smoke fidelity
5. a benchmark note demonstrates real reinterpretation speedup

## 12. Follow-up work

Not in v1, but natural next steps:

- simplified base artifact plus signal patchset publication flow
- multi-POI reduced likelihoods
- dedicated `SimplifiedLikelihoodModel` if reduction-to-HistFactory becomes
  limiting
- richer visualization for covariance-only artifacts
- HEPData fetch/materialization helpers for published simplified-likelihood
  datasets

## 13. Summary

The pragmatic March 2026 decision is:

- treat simplified likelihoods as a translation/export feature first
- compile them into reduced HistFactory models
- force linear interpolation semantics
- ship with hard TDD and fidelity gates

That keeps the feature in the intended effort band, preserves compatibility with
ATLAS/pyhf-style workflows, gives CMS-style covariance releases a clean entry
point, and creates a credible path to the `~10x` reinterpretation speedup goal.
