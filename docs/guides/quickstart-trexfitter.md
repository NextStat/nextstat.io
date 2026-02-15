---
title: "Route B Quickstart (TRExFitter config)"
status: draft
---

# Route B: TRExFitter Config Quickstart

Goal: start from a TREx-style config and run one-command replacement steps in NextStat.

Route fixture config:

- `docs/guides/fixtures/route_b/minimal_ntup_quickstart.config`

## 0) Build CLI

```bash
CARGO_TARGET_DIR=target cargo build -p ns-cli
```

## 1) Import config -> workspace (CLI)

```bash
mkdir -p tmp/guides/route_b
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- import trex-config \
  --config docs/guides/fixtures/route_b/minimal_ntup_quickstart.config \
  --base-dir . \
  --output tmp/guides/route_b/workspace_from_import_trex_config.json
```

## 2) Build histograms from ntuples -> workspace (CLI)

```bash
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- build-hists \
  --config docs/guides/fixtures/route_b/minimal_ntup_quickstart.config \
  --base-dir . \
  --out-dir tmp/guides/route_b/build_hists \
  --overwrite
```

Workspace output:

- `tmp/guides/route_b/build_hists/workspace.json`

## 3) Fit + diagnostics artifacts (CLI)

```bash
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- fit \
  --input tmp/guides/route_b/build_hists/workspace.json \
  --threads 1 \
  --output tmp/guides/route_b/fit_result.json

CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- scan \
  --input tmp/guides/route_b/build_hists/workspace.json \
  --start 0 --stop 3 --points 31 --threads 1 \
  --output tmp/guides/route_b/scan_points.json

CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- viz cls \
  --input tmp/guides/route_b/build_hists/workspace.json \
  --scan-start 0 --scan-stop 3 --scan-points 81 --threads 1 \
  --output tmp/guides/route_b/cls_curve.json

CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- viz pulls \
  --input tmp/guides/route_b/build_hists/workspace.json \
  --fit tmp/guides/route_b/fit_result.json --threads 1 \
  --output tmp/guides/route_b/pulls.json

CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- viz corr \
  --input tmp/guides/route_b/build_hists/workspace.json \
  --fit tmp/guides/route_b/fit_result.json --threads 1 \
  --output tmp/guides/route_b/corr.json
```

## 4) One-command workflow (CLI)

```bash
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- run --help
```

Use `nextstat run` when your analysis spec YAML is ready; for this quickstart we keep explicit step-by-step commands for easier debugging.

## Expected outputs (reference)

Compare with:

- `docs/guides/fixtures/route_b/workspace_from_trex_config.json`
- `docs/guides/fixtures/route_b/workspace_from_import_trex_config.json`
- `docs/guides/fixtures/route_b/fit_result.json`
- `docs/guides/fixtures/route_b/upper_limit_scan.json`
- `docs/guides/fixtures/route_b/scan_points.json`
- `docs/guides/fixtures/route_b/scan_points.csv`
- `docs/guides/fixtures/route_b/cls_curve.json`
- `docs/guides/fixtures/route_b/pulls.json`
- `docs/guides/fixtures/route_b/corr.json`
- `docs/guides/fixtures/route_b/reference_plot.png`
- `docs/guides/fixtures/route_b/validation_report_snippet.json`
