# Stable-First GVM Example Bundle

This directory is the canonical runnable example for the stable-first scalar
GVM measurement-combination path.

Files:

- `manifest.yaml`
- `measurements.csv`
- `stat_covariance.csv`
- `systematics.csv`
- `correlations.csv`
- `spec.json`

One-command stable-first run:

```bash
make gvm-stable-first-example
```

This writes `spec.json`, `result.json`, `calibration.json`, and
`calibration_study.json` to `tmp/gvm-stable-first-example/`.

Typical workflow:

```bash
nextstat combine-measurements-build-spec \
  --manifest docs/examples/gvm-stable-first/manifest.yaml \
  --output spec.json

nextstat combine-measurements \
  --input spec.json \
  --output result.json \
  --solver auto \
  --threads 1

nextstat combine-measurements-calibrate \
  --input spec.json \
  --output calibration.json \
  --solver auto \
  --n-toys 32 \
  --seed 42 \
  --threads 1

nextstat combine-measurements-calibrate-study \
  --input spec.json \
  --output calibration_study.json \
  --solver auto \
  --n-toys 32 \
  --seeds 42,43 \
  --threads 1
```

The committed `spec.json` is the canonical JSON generated from the tabular
bundle. It exists both as a user example and as a regression anchor for the
stable-first tabular ingress.

The committed `manifest.yaml` is the shortest stable-first wrapper around that
bundle. It keeps the source-of-truth tables together and resolves relative
paths from the example directory.

If you are sending this bundle to an external physics user for first validation,
also include:

- `docs/guides/gvm-external-validation-kit.md`
- `docs/guides/gvm-external-validator-outreach-pack.md`
- `docs/guides/gvm-external-validation-tracker-template.md`
- `docs/examples/gvm-stable-first/external-validator-invite-template.md`
- `docs/examples/gvm-stable-first/external-validation-report-template.md`
