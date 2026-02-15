---
title: "Route A Quickstart (pyhf/HS3 workspace)"
status: draft
---

# Route A: pyhf/HS3 Workspace Quickstart

Goal: copy-paste path from workspace JSON to fit, hypotest, profile scan, and Brazil band.

Fixture workspace:

- `tests/fixtures/simple_workspace.json`

## 0) Build CLI

```bash
CARGO_TARGET_DIR=target cargo build -p ns-cli
```

## 1) Fit (CLI)

```bash
mkdir -p tmp/guides/route_a
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- fit \
  --input tests/fixtures/simple_workspace.json \
  --parity --threads 1 \
  --output tmp/guides/route_a/fit_result.json
```

## 2) Hypotest (CLI)

```bash
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- hypotest \
  --input tests/fixtures/simple_workspace.json \
  --mu 1.0 --expected-set --threads 1 \
  --output tmp/guides/route_a/hypotest_mu1.json
```

## 3) Upper-limit scan (CLI)

```bash
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- upper-limit \
  --input tests/fixtures/simple_workspace.json \
  --expected --scan-start 0 --scan-stop 5 --scan-points 101 --threads 1 \
  --output tmp/guides/route_a/upper_limit_scan.json
```

## 4) Profile scan + Brazil artifact (CLI)

```bash
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- scan \
  --input tests/fixtures/simple_workspace.json \
  --start 0 --stop 5 --points 41 --threads 1 \
  --output tmp/guides/route_a/scan_points.json

CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- viz cls \
  --input tests/fixtures/simple_workspace.json \
  --scan-start 0 --scan-stop 5 --scan-points 101 --threads 1 \
  --output tmp/guides/route_a/cls_curve.json
```

## 5) Toys CLs (CLI)

```bash
CARGO_TARGET_DIR=target cargo run -q -p ns-cli -- hypotest-toys \
  --input tests/fixtures/simple_workspace.json \
  --mu 1.0 --n-toys 200 --seed 42 --expected-set --threads 1 \
  --output tmp/guides/route_a/hypotest_toys_mu1.json
```

## Python API (same route)

```python
import json
import nextstat

with open("tests/fixtures/simple_workspace.json", "r", encoding="utf-8") as f:
    ws = json.load(f)

model = nextstat.from_pyhf(ws)
fit = nextstat.fit(model)
scan = nextstat.profile_scan(model, start=0.0, stop=5.0, points=41)
ul = nextstat.upper_limit(model, expected=True)
```

## Expected outputs (reference)

Compare with:

- `docs/guides/fixtures/route_a/fit_result.json`
- `docs/guides/fixtures/route_a/hypotest_mu1.json`
- `docs/guides/fixtures/route_a/upper_limit_scan.json`
- `docs/guides/fixtures/route_a/scan_points.json`
- `docs/guides/fixtures/route_a/scan_points.csv`
- `docs/guides/fixtures/route_a/cls_curve.json`
- `docs/guides/fixtures/route_a/pulls.json`
- `docs/guides/fixtures/route_a/corr.json`
- `docs/guides/fixtures/route_a/reference_plot.png`
- `docs/guides/fixtures/route_a/validation_report_snippet.json`
