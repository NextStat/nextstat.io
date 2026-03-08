---
title: "HEP GVM Measurement Combinations"
status: stable-first
---

# HEP GVM Measurement Combinations

This tutorial is the practical entry point for NextStat's stable-first scalar
measurement-combination engine, with the wider advanced reporting stack kept as
research-grade, for correlated scalar measurements with `error_on_error`
support.

Stable-first status now applies to the foundational inference subset:

- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- `nextstat.hep.combine_measurements(...)`
- `nextstat.hep.calibrate_measurements(...)`
- `nextstat.hep.calibrate_measurements_study(...)`

Scenario, campaign, parity, and higher reporting layers documented below remain
research-grade.

Use this workflow when you already have reduced measurements, a statistical
covariance matrix, and systematic sources with per-source correlation matrices.

Do **not** use this workflow for HistFactory or pyhf workspace fits. That path
stays under `nextstat fit`, `nextstat hypotest`, and related workspace commands.

## Recommended stable-first input path

If your source of truth is a spreadsheet or analysis note, do **not** start by
hand-writing the JSON spec. The stable-first golden path is:

1. prepare CSV/TSV tables
2. build the canonical JSON spec once
3. run fit, calibration, and repeated-seed study on that generated spec

Stable-first tabular entry points:

- CLI: `nextstat combine-measurements-build-spec`
- Python: `nextstat.hep.build_measurement_combination_spec(...)`
- Python manifest wrapper: `nextstat.hep.build_measurement_combination_spec_from_manifest(...)`

Canonical runnable bundle in this repo:

- `docs/examples/gvm-stable-first/`

Stable-first shortest path inside that bundle:

- `docs/examples/gvm-stable-first/manifest.yaml`
- `make gvm-stable-first-example`

If you are validating adoption with an external physics user, do not improvise a
handoff. Use the committed maintainer kit:

- `docs/guides/gvm-external-validation-kit.md`
- `docs/guides/gvm-external-validator-outreach-pack.md`
- `docs/guides/gvm-external-validation-tracker-template.md`
- `docs/examples/gvm-stable-first/external-validator-invite-template.md`
- `docs/examples/gvm-stable-first/external-validation-report-template.md`

This keeps the canonical JSON schema as the runtime contract while making the
user-facing input path much closer to how HEP combinations are usually tracked
in tables.

## Paper reference

L. Canonero and G. Cowan, "Combination of measurements and the BLUE method
generalized by allowing for errors in the error assignments," *Eur. Phys. J. C*
**85**, 156 (2025).

NextStat is the **only known public implementation** of the GVM likelihood. No
other HEP package (pyhf, ROOT/RooFit, or otherwise) currently provides it.

## What you get

The GVM workflow gives you:

- a likelihood-based scalar measurement combination
- optional `error_on_error` handling through the Gamma Variance Model (GVM)
- three independent solver paths for cross-validation
- Lawley/Bartlett O(ε²) correction with fast (Woodbury) and reference paths
- profile likelihood confidence intervals with exponential bracketing and bisection
- deterministic toy calibration reports
- scenario studies over multiple `error_on_error` assignments
- repeated-seed calibration campaigns
- an 11-level calibration pyramid from single fit to portfolio stability
- paper-facing solver-parity studies between:
  - `numerical-paper`
  - `analytic-perturbative`

## Default solver contract

The default solver is `auto` for the stable-first core and for the wider
research-grade extensions.

`auto` means:

1. try the perturbative paper path first
2. if the Eq. `(29)/(60)` validity gate fails, fall back automatically to the
   paper-faithful numerical path in the original correlated `theta_s^i` basis

That default applies consistently across:

- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- `nextstat combine-measurements-scenario-study`
- `nextstat combine-measurements-calibration-campaign`
- `nextstat.hep.*` wrappers

For deterministic parity or fixture generation, keep `--threads 1`.

## Minimal input spec

The canonical input artifact is `nextstat_measurement_combination_v0`.

Example:

```json
{
  "schema_version": "nextstat_measurement_combination_v0",
  "poi": "mu",
  "measurements": [
    { "name": "atlas_ljets", "value": 172.40 },
    { "name": "cms_ljets", "value": 172.62 }
  ],
  "stat_covariance": [
    [0.04, 0.00],
    [0.00, 0.05]
  ],
  "systematics": [
    {
      "name": "b-JES",
      "magnitudes": [0.30, 0.28],
      "corr": [
        [1.0, 0.8],
        [0.8, 1.0]
      ],
      "error_on_error": 0.10,
      "aux_mean": 0.0
    },
    {
      "name": "hadronization",
      "magnitudes": [0.20, 0.18],
      "corr": [
        [1.0, 1.0],
        [1.0, 1.0]
      ],
      "error_on_error": 0.00,
      "aux_mean": 0.0
    }
  ]
}
```

Field semantics:

- `measurements[].value`: scalar observed measurement values
- `stat_covariance`: measurement-by-measurement statistical covariance
- `systematics[].magnitudes`: one assigned systematic magnitude per measurement
- `systematics[].corr`: per-source correlation matrix across measurements
- `systematics[].error_on_error`: relative uncertainty on that systematic size
- `systematics[].aux_mean`: optional auxiliary-mean offset, usually `0.0`

Important v1 simplification:

- the public spec uses the normalized convention from the RFC
- you do not provide raw `v_s`
- `magnitudes` are the assigned systematic errors directly

## Stable-first tabular bundle

If you start from tables instead of JSON, the minimal bundle is:

- `measurements.csv`
  - columns: `name,value`
- `stat_covariance.csv`
  - named square matrix with row/column measurement names
- optional `systematics.csv`
  - columns: `systematic,measurement,magnitude,error_on_error,aux_mean`
- optional `correlations.csv`
  - columns: `systematic,row_measurement,col_measurement,corr`

If `correlations.csv` is omitted, each systematic defaults to identity
correlation.

CLI example:

```bash
nextstat combine-measurements-build-spec \
  --manifest docs/examples/gvm-stable-first/manifest.yaml \
  --output spec.json
```

Direct table flags remain supported when you do not want a manifest wrapper.

Python example:

```python
from nextstat import hep

spec = hep.build_measurement_combination_spec(
    "docs/examples/gvm-stable-first/measurements.csv",
    "docs/examples/gvm-stable-first/stat_covariance.csv",
    poi="mu",
    systematics_table="docs/examples/gvm-stable-first/systematics.csv",
    correlations_table="docs/examples/gvm-stable-first/correlations.csv",
)

manifest_spec = hep.build_measurement_combination_spec_from_manifest(
    "docs/examples/gvm-stable-first/manifest.yaml"
)
```

## Minimal scenario config

Scenario studies and calibration campaigns use a second JSON artifact:

```json
{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p1",
      "error_on_error": [
        { "systematic": "b-JES", "value": 0.10 }
      ]
    },
    {
      "name": "bjes_0p3",
      "error_on_error": [
        { "systematic": "b-JES", "value": 0.30 }
      ]
    },
    {
      "name": "theory_core_0p2",
      "error_on_error": [
        { "systematic": "hadronization", "value": 0.20 }
      ]
    }
  ]
}
```

## CLI workflow

### 1. Build one spec from tables

```bash
nextstat combine-measurements-build-spec \
  --manifest docs/examples/gvm-stable-first/manifest.yaml \
  --output spec.json
```

### 2. Combine one spec

```bash
nextstat combine-measurements \
  --input spec.json \
  --output result.json \
  --ci-level 0.68 \
  --solver auto \
  --threads 1
```

What to inspect in `result.json`:

- `mu_hat`
- `confidence_interval`
- `goodness_of_fit`
- `optimizer.method`
- `diagnostics.requested_solver`
- `diagnostics.effective_solver`
- `diagnostics.perturbative_validity`
- `diagnostics.bartlett`

Typical solver outcomes:

- `analytic_perturbative_order_eps2`
- `numerical_profile_gvm_original_theta`
- `closed_form_blue` when all `error_on_error == 0`

If you request `--solver auto`, use `diagnostics.requested_solver` and
`diagnostics.effective_solver` to see whether runtime dispatch stayed on the
perturbative fast path or fell back to the paper-faithful numerical path.

### 3. Run toy calibration

```bash
nextstat combine-measurements-calibrate \
  --input spec.json \
  --output calibration.json \
  --ci-level 0.68 \
  --solver auto \
  --n-toys 256 \
  --seed 42 \
  --threads 1
```

Calibration reports tell you whether Bartlett-corrected summaries behave better
than the raw proxy across deterministic toys.

Key fields:

- `reference`
- `summary.mean_q`
- `summary.mean_q_star`
- `summary.mean_sigma`
- `summary.mean_sigma_star`
- `summary.mean_sigma_star_to_sigma_ratio`
- `summary.bartlett_improves_mean_q`

### 4. Run a repeated-seed calibration study

```bash
nextstat combine-measurements-calibrate-study \
  --input spec.json \
  --output calibration_study.json \
  --ci-level 0.68 \
  --solver auto \
  --n-toys 256 \
  --seeds 42,43,44 \
  --threads 1
```

Use this when one seed is not enough and you want a stability artifact instead
of one toy report.

### 5. Run a scenario study

```bash
nextstat combine-measurements-scenario-study \
  --input spec.json \
  --scenarios scenarios.json \
  --output scenario_study.json \
  --ci-level 0.68 \
  --solver auto \
  --threads 1
```

Use this to compare named `error_on_error` assignments against one baseline.

### 6. Run a calibration campaign

```bash
nextstat combine-measurements-calibration-campaign \
  --input spec.json \
  --scenarios scenarios.json \
  --output campaign.json \
  --ci-level 0.68 \
  --solver auto \
  --n-toys 128 \
  --seeds 42,43,44 \
  --threads 1
```

This is the main research artifact for comparing fit-side and calibration-side
behavior across multiple scenarios and seeds.

### 6. Summarize a campaign

```bash
nextstat combine-measurements-calibration-campaign-summarize \
  --input campaign.json \
  --output campaign_summary.json

nextstat combine-measurements-calibration-campaign-summarize \
  --input campaign.json \
  --format markdown \
  --output campaign_summary.md
```

Use JSON for downstream tooling and Markdown for review.

### 7. Compare solvers directly

Scenario-study parity:

```bash
nextstat combine-measurements-solver-parity-scenario-study \
  --input spec.json \
  --scenarios scenarios.json \
  --output parity_scenarios.json \
  --lhs-solver numerical-paper \
  --rhs-solver analytic-perturbative \
  --threads 1
```

Calibration-campaign parity:

```bash
nextstat combine-measurements-solver-parity-calibration-campaign \
  --input spec.json \
  --scenarios scenarios.json \
  --output parity_campaign.json \
  --lhs-solver numerical-paper \
  --rhs-solver analytic-perturbative \
  --n-toys 128 \
  --seeds 42,43,44 \
  --threads 1
```

Use these when you want a Fig. 5 style paper-facing comparison between the
numerical paper path and the perturbative approximation.

## Python workflow

The same engine is available under `nextstat.hep`.

### 1. Combine one spec

```python
import json
import nextstat

spec = json.load(open("spec.json"))
result = nextstat.hep.combine_measurements(
    spec,
    ci_level=0.68,
    solver="auto",
)

print(result["mu_hat"])
print(result["optimizer"]["method"])
print(result["diagnostics"]["bartlett"]["supported"])
```

### 2. Run toy calibration

```python
report = nextstat.hep.calibrate_measurements(
    spec,
    ci_level=0.68,
    solver="auto",
    n_toys=256,
    seed=42,
)

print(report["summary"]["mean_q_star"])
print(report["summary"]["mean_sigma_star_to_sigma_ratio"])
```

### 3. Run scenario study and campaign

```python
scenarios = json.load(open("scenarios.json"))

scenario_report = nextstat.hep.study_measurement_combination_scenarios(
    spec,
    scenarios,
    ci_level=0.68,
    solver="auto",
)

campaign = nextstat.hep.calibrate_measurement_combination_scenarios(
    spec,
    scenarios,
    ci_level=0.68,
    solver="auto",
    n_toys=128,
    seeds=[42, 43, 44],
)
```

### 4. Summarize or render existing artifacts

```python
summary = nextstat.hep.summarize_measurement_combination_calibration_campaign(campaign)
markdown = nextstat.hep.render_measurement_combination_calibration_campaign_summary(summary)
```

### 5. Compare solver modes

```python
parity = nextstat.hep.compare_measurement_combination_scenario_study_solvers(
    spec,
    scenarios,
    ci_level=0.68,
    lhs_solver="numerical-paper",
    rhs_solver="analytic-perturbative",
)
```

## How to choose a solver

Use:

- `auto`
  - default
  - best choice for most users
  - tries perturbative first, then falls back safely
- `numerical-paper`
  - use when you need the paper-faithful reference path in the original
    correlated `theta_s^i` basis
  - preferred for parity studies and paper comparisons
- `analytic-perturbative`
  - use when you want the Eq. `(21)-(28)` / Appendix B approximation directly
  - rejects invalid cases instead of falling back
- `numerical`
  - compatibility path for the reduced-basis numerical solver
  - keep explicit; do not assume it is paper-faithful

## Recommended workflow

For most HEP combination studies:

1. Start with `combine-measurements --solver auto --threads 1`
2. Inspect `diagnostics.perturbative_validity`
3. If the perturbative gate fails often, compare against `numerical-paper`
4. Run `...-calibrate` for one deterministic seed
5. Run `...-calibrate-study` for repeated-seed stability
6. Run `...-scenario-study` for named `error_on_error` scans
7. Run `...-calibration-campaign` only when you need one artifact spanning
   both scenario and seed dimensions

## Reproducibility rules

For research-grade parity:

- use `--threads 1`
- set `--seed` or `--seeds`
- keep committed JSON artifacts under version control
- use the cached post-processing commands when you do not need to rerun fits or
  toys

For published evidence beyond solver-parity and performance numbers, use the
committed robustness snapshots:

- [GVM Benchmark Snapshot](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md)
- [GVM NumericalPaper Robustness Snapshot](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md)
- [GVM Stable-Surface Readiness Memo](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md)
- [GVM Stable-Surface Support Policy](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md)
- [GVM Stable-First Promotion Decision](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-decision-2026-03-07.md)
- [GVM Stable-First Support Matrix](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)
- [GVM Stable-First Release Notes](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md)

Cached parity helpers:

- `...-solver-parity-scenario-study-from-reports`
- `...-solver-parity-calibration-campaign-from-reports`
- `...-solver-parity-*-summarize`
- `...-calibration-campaign-summarize`
- `...-calibration-campaign-brief`
- `...-calibration-campaign-family-report`
- `...-calibration-campaign-family-matrix`
- `...-calibration-campaign-portfolio`
- `...-calibration-campaign-portfolio-stability`

## Troubleshooting

### "analytic perturbative path is outside the Eq. (29)/(60) validity radius"

This means the perturbative approximation is not trusted for that case.

What to do:

- use `solver="auto"` or `--solver auto`
- or explicitly rerun with `numerical-paper`

### Non-PSD source correlation matrices

Published HEP inputs can contain raw per-source matrices that are not strictly
PSD. NextStat keeps those raw inputs in the BLUE covariance construction and
applies minimal regularization only where inversion is required. Inspect:

- `diagnostics.corr_regularization_deltas`

### Which command should I use?

Use:

- `combine-measurements`
  - one result
- `combine-measurements-calibrate`
  - one toy calibration run
- `combine-measurements-calibrate-study`
  - repeated-seed toy study
- `combine-measurements-scenario-study`
  - multiple named `error_on_error` scenarios
- `combine-measurements-calibration-campaign`
  - scenario study + repeated-seed calibration together

## Three independent solvers

NextStat provides three solver paths that cross-validate each other:

| Solver | Method | Best for |
|--------|--------|----------|
| `analytic-perturbative` | Eq. (21)-(28) / Appendix B perturbative expansion to O(ε²) | Fast path, small-to-moderate error_on_error |
| `numerical-paper` | Paper-faithful numerical profiling in original correlated θ basis | Reference path, parity studies, large error_on_error |
| `numerical` | Reduced-basis numerical path (QR-decomposed) | Legacy compatibility, fastest numerical |
| `auto` (default) | Tries perturbative first; falls back to `numerical-paper` outside validity radius | Best choice for most users |

When all `error_on_error` values are zero, the GVM reduces to the standard BLUE
and the result is computed in closed form — no iteration needed.

## Bartlett-Lawley correction

The GVM profile likelihood ratio can deviate from chi-squared when error_on_error
is non-trivial. NextStat computes the Bartlett-Lawley O(ε²) correction factor:

- **Fast (Woodbury)** — Sherman-Morrison-Woodbury identity, O(N·K) complexity
- **Reference** — direct dense-matrix computation for cross-validation

The corrected test statistic `q*` and its confidence interval `σ*` are available
in `diagnostics.bartlett` of every fit result.

## 11-level calibration pyramid

Each level adds a dimension of evidence on top of the previous one:

| Level | Artifact | What it adds |
|-------|----------|--------------|
| 1 | Fit | Single combination: μ̂, CI, GoF, Bartlett |
| 2 | Calibrate | Single-seed toy calibration: σ*/σ ratio |
| 3 | Calibrate Study | Multi-seed stability of calibration |
| 4 | Scenario Study | Sweep over named error_on_error assignments |
| 5 | Campaign | Scenarios × seeds in one artifact |
| 6 | Digest / Summary | Compact campaign metrics for downstream |
| 7 | Brief | Cross-campaign aggregation |
| 8 | Family Report | Cross-brief comparison |
| 9 | Family Matrix | Pairwise dominance matrix |
| 10 | Portfolio | Cross-matrix portfolio view |
| 11 | Portfolio Stability | Stability of portfolio conclusions across seed grids |

Every artifact is a versioned JSON document with `schema_version`, and every
level can also emit Markdown for human review. There are 17 CLI commands and
28 Python API functions covering all levels.

## Performance

Criterion benchmarks (single-threaded, release build):

| Fixture | Solver | Time |
|---------|--------|------|
| Paper top-mass (15 × 22) | auto | 101 µs |
| Synthetic (32 × 24) | analytic-perturbative | 13.4 ms |
| Synthetic (32 × 24) | numerical-paper | 44.6 ms |
| Synthetic (64 × 48) | analytic-perturbative | 124.7 ms |

## Test coverage

- **122 Rust unit tests** covering all solver paths, edge cases, and calibration levels
- **37 golden-file fixtures** (JSON + Markdown) for regression testing
- **CLI integration tests** for all 17 commands
- **Python API tests** via `test_hep_module_api.py`
- Paper-faithful top-mass fixture transcribed from Canonero & Cowan Table 1

## Related docs

- CLI reference: `docs/references/cli.md`
- Python API: `docs/references/python-api.md`
- RFC: `docs/rfcs/research-grade-measurement-combinations.md`
- Stable-surface memo: `docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md`
- Stable-surface policy: `docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md`
- Stable-first decision: `docs/benchmarks/gvm-stable-first-decision-2026-03-07.md`
- Stable-first support matrix: `docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md`
- Stable-first release notes: `docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md`
- HEP full workflow tutorial: `docs/tutorials/hep-full-workflow.md`
