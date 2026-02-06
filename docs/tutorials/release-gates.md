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
make apex2-baseline-compare COMPARE_ARGS="--require-same-host"
```

or:

```bash
bash scripts/apex2/pre_release_gate.sh
```

Exit codes:
- `0`: OK (parity OK and within slowdown thresholds)
- `2`: FAIL (parity failure or slowdown threshold exceeded)
- `3`: baseline manifest missing/invalid
- `4`: runner error (missing deps, crash, etc.)

## If it fails

- Open `tmp/baseline_compare_report.json` and check:
  - `pyhf.compare.cases[*].ok` for perf regressions
  - `p6_glm.compare.compare.cases[*].ok` for fit/predict regressions
- If the baseline is stale (e.g. after a known perf improvement), record a new baseline and re-run the gate.

## Cluster notes (ROOT/TRExFitter)

ROOT/HistFactory parity baselines are recorded separately on a cluster environment (e.g. lxplus) via:
- `tests/record_baseline.py --only root ...`

See `docs/tutorials/root-trexfitter-parity.md` for the HTCondor job-array workflow and aggregation.

