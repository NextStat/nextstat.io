# Stable-First GVM External Validation Report Template

Use this template when validating the stable-first scalar GVM path on a real
machine outside the maintainer environment.

## Validator

- Name / team:
- Date:
- Machine / OS:
- Python version:
- Rust / cargo version:
- NextStat version or commit:

## Pass 1: committed example bundle

Command used:

```bash
make gvm-stable-first-example
```

Status:

- [ ] passed without edits
- [ ] passed with local edits
- [ ] failed

Generated files:

- [ ] `spec.json`
- [ ] `result.json`
- [ ] `calibration.json`
- [ ] `calibration_study.json`

Key observations:

- `diagnostics.requested_solver`:
- `diagnostics.effective_solver`:
- Anything unclear in the output:

## Pass 2: one small real case

Input description:

- number of measurements:
- number of systematic sources:
- source format:

Commands used:

```bash
nextstat combine-measurements-build-spec ...
nextstat combine-measurements ...
nextstat combine-measurements-calibrate ...
nextstat combine-measurements-calibrate-study ...
```

Status:

- [ ] passed without maintainer help
- [ ] passed with maintainer help
- [ ] failed

Returned outputs:

- [ ] spec
- [ ] fit result
- [ ] calibration
- [ ] calibration study

Key result fields:

- `mu_hat`:
- `confidence_interval`:
- `goodness_of_fit`:
- `diagnostics.requested_solver`:
- `diagnostics.effective_solver`:

## Friction log

List anything that was confusing or slow:

- input formatting:
- manifest/table mapping:
- command naming:
- diagnostics wording:
- runtime / performance:
- packaging / environment:

## Overall verdict

- [ ] ready for real analysis notes / small combinations
- [ ] needs docs cleanup
- [ ] needs packaging cleanup
- [ ] needs solver/runtime clarification

Short summary:
