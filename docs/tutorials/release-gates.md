# Release Gates (Apex2, no cluster)

This document describes the minimal **pre-release gate** based on the Apex2 baseline workflow.

## What it checks

- **pyhf parity** vs the pyhf reference implementation
- **P6 GLM** end-to-end fit/predict stability vs a recorded baseline
- **Performance regressions** vs the latest baseline manifest on the same machine

Artifacts:
- baselines: `tmp/baselines/`
- compare report: `tmp/baseline_compare_report.json`

## Step-by-step (recommended)

1) Record baselines once on a reference machine:

```bash
make apex2-baseline-record
```

2) Before cutting a release, run the pre-release gate:

```bash
make apex2-baseline-compare COMPARE_ARGS="--require-same-host --p6-attempts 2"
```

or:

```bash
make apex2-pre-release-gate
```

This runs:
- `cargo build --workspace --release`
- `cargo test --workspace --all-features`
- `maturin develop --release` (Python bindings)
- `pytest -m "not slow" tests/python`
- `tests/compare_with_latest_baseline.py --require-same-host --p6-attempts 2` (P6 retried up to N times; whole compare retried once if it fails with `rc=2`)

Exit codes:
- `0`: OK (parity OK and within slowdown thresholds)
- `2`: FAIL (parity failure or slowdown threshold exceeded)
- `3`: baseline manifest missing/invalid
- `4`: runner error (missing deps, crash, etc.)
- `5`: git working tree dirty (set `APEX2_ALLOW_DIRTY=1` to override)

## If it fails

- Open `tmp/baseline_compare_report.json` and check:
  - `pyhf.compare.cases[*].ok` for perf regressions
  - `p6_glm.attempts[*].status` to see retry outcomes
  - `p6_glm.compare.compare.cases[*].ok` for fit/predict regressions (selected attempt)
- If the baseline is stale (e.g. after a known perf improvement), record a new baseline and re-run the gate.

## Cluster notes (ROOT/TRExFitter)

ROOT/HistFactory parity baselines are recorded separately on a cluster environment (e.g. lxplus) via:
- `tests/record_baseline.py --only root ...`

See `docs/tutorials/root-trexfitter-parity.md` for the HTCondor job-array workflow and aggregation.
