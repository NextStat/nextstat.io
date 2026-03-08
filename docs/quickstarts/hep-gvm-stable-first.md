---
title: "HEP GVM Stable-First Quickstart"
status: stable-first
---

# HEP GVM Stable-First Quickstart

Goal: get a real scalar GVM measurement-combination result in about 5 minutes
from the committed example bundle, without hand-writing either the JSON spec or
the long table CLI arguments.

This is the shortest supported path for the stable-first GVM subset:

- `nextstat combine-measurements-build-spec`
- `nextstat combine-measurements`
- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`

The source bundle lives in:

- [docs/examples/gvm-stable-first](/Users/andresvlc/WebDev/nextstat.io/docs/examples/gvm-stable-first)

One-command golden path:

```bash
make gvm-stable-first-example
```

That command writes the spec, fit, calibration, and calibration-study outputs
to `tmp/gvm-stable-first-example/`.

## 1. Build the canonical spec from the manifest bundle

From the repo root:

```bash
nextstat combine-measurements-build-spec \
  --manifest docs/examples/gvm-stable-first/manifest.yaml \
  --output /tmp/gvm-spec.json
```

If you need lower-level control, the same command still accepts the raw
`--measurements`, `--stat-covariance`, `--systematics`, and `--correlations`
table flags directly.

## 2. Run the stable fit

```bash
nextstat combine-measurements \
  --input /tmp/gvm-spec.json \
  --output /tmp/gvm-result.json \
  --solver auto \
  --threads 1
```

Inspect:

- `mu_hat`
- `confidence_interval`
- `goodness_of_fit`
- `diagnostics.requested_solver`
- `diagnostics.effective_solver`

## 3. Run deterministic toy calibration

```bash
nextstat combine-measurements-calibrate \
  --input /tmp/gvm-spec.json \
  --output /tmp/gvm-calibration.json \
  --solver auto \
  --n-toys 32 \
  --seed 42 \
  --threads 1
```

## 4. Run repeated-seed stability

```bash
nextstat combine-measurements-calibrate-study \
  --input /tmp/gvm-spec.json \
  --output /tmp/gvm-study.json \
  --solver auto \
  --n-toys 32 \
  --seeds 42,43 \
  --threads 1
```

## 5. Equivalent Python path

```python
from nextstat import hep

spec = hep.build_measurement_combination_spec(
    # Raw table path is still supported.
    "docs/examples/gvm-stable-first/measurements.csv",
    "docs/examples/gvm-stable-first/stat_covariance.csv",
    poi="mu",
    systematics_table="docs/examples/gvm-stable-first/systematics.csv",
    correlations_table="docs/examples/gvm-stable-first/correlations.csv",
)

manifest_spec = hep.build_measurement_combination_spec_from_manifest(
    "docs/examples/gvm-stable-first/manifest.yaml"
)

result = hep.combine_measurements(spec, solver="auto")
calibration = hep.calibrate_measurements(spec, solver="auto", n_toys=32, seed=42)
study = hep.calibrate_measurements_study(spec, solver="auto", n_toys=32, seeds=[42, 43])
```

## Stable scope

Stable-first subset:

- `build-spec --manifest`
- `build-spec`
- `combine`
- `calibrate`
- `calibrate-study`

Still research-grade:

- `scenario-study`
- `calibration-campaign`
- solver parity
- cached reporting / brief / family / portfolio layers

For the wider context, see:

- [HEP GVM Measurement Combinations Tutorial](/Users/andresvlc/WebDev/nextstat.io/docs/tutorials/hep-gvm-measurement-combinations.md)
- [GVM Stable-First Support Matrix](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)

For maintainer-driven first external rollout, use:

- [GVM External Validation Kit](/Users/andresvlc/WebDev/nextstat.io/docs/guides/gvm-external-validation-kit.md)
- [GVM External Validator Outreach Pack](/Users/andresvlc/WebDev/nextstat.io/docs/guides/gvm-external-validator-outreach-pack.md)
- [GVM External Validation Tracker Template](/Users/andresvlc/WebDev/nextstat.io/docs/guides/gvm-external-validation-tracker-template.md)

For release execution on top of that rollout, use:

- [GVM Stable-First Release Candidate: v0.10.0](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md)
- [GVM Stable-First Release PR Checklist](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-release-pr-checklist-2026-03-07.md)
- [GVM Stable-First Launch Checklist](/Users/andresvlc/WebDev/nextstat.io/docs/benchmarks/gvm-stable-first-launch-checklist-2026-03-07.md)
