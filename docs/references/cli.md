---
title: "CLI Reference (nextstat)"
status: stable
---

# CLI Reference (nextstat)

The `nextstat` CLI is implemented in `crates/ns-cli` and focuses on:
- deterministic parity mode (`--threads 1`)
- JSON in / JSON out contracts for reproducible workflows

## Commands (high level)

HEP / HistFactory:
- `nextstat fit --input workspace.json`
- `nextstat hypotest --input workspace.json --mu 1.0 [--expected-set]`
- `nextstat upper-limit --input workspace.json [--expected] [--scan-start ... --scan-stop ... --scan-points ...]`
- `nextstat scan --input workspace.json --start 0 --stop 5 --points 21`
- `nextstat viz profile --input workspace.json ...`
- `nextstat viz cls --input workspace.json ...`
- `nextstat viz ranking --input workspace.json`
- `nextstat viz pulls --input workspace.json --fit fit.json`
- `nextstat viz corr --input workspace.json --fit fit.json`
- `nextstat viz distributions --input workspace.json --histfactory-xml combination.xml [--fit fit.json]`
- `nextstat report --input workspace.json --histfactory-xml combination.xml --fit fit.json --out-dir report/ [--render]`

Time series (Phase 8):
- `nextstat timeseries kalman-filter --input kalman_1d.json`
- `nextstat timeseries kalman-smooth --input kalman_1d.json`
- `nextstat timeseries kalman-em --input kalman_1d.json ...`
- `nextstat timeseries kalman-fit --input kalman_1d.json ...`
- `nextstat timeseries kalman-forecast --input kalman_1d.json ...`
- `nextstat timeseries kalman-simulate --input kalman_1d.json ...`

## Determinism and parity

Use `--threads 1` for deterministic parity comparisons (pyhf or baselines).

## Upper limit: bisection vs scan mode

`upper-limit` supports two modes:

1. Bisection (root finding): default.
2. Scan mode: provide `--scan-start`, `--scan-stop`, `--scan-points` to compute limits from a dense CLs curve.

Scan mode is useful for:
- storing a full curve for plotting
- avoiding repeated root-finding (including expected-set curves)

## JSON contracts

The CLI outputs pretty JSON to stdout by default, or to `--output`.

`nextstat report` writes multiple JSON artifacts into `--out-dir` (currently: `distributions.json`, `pulls.json`, `corr.json`, `yields.json`). When `--render` is enabled it calls `python -m nextstat.report render ...` to produce a multi-page PDF and per-plot SVGs (requires `matplotlib`, see `nextstat[viz]` extra).

For time series input formats, see:
- `docs/tutorials/phase-8-timeseries.md`

For the frequentist (CLs) workflow, see:
- `docs/tutorials/phase-3.1-frequentist.md`
