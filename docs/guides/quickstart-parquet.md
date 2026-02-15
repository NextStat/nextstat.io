---
title: "Route C Quickstart (Parquet/Arrow data-lake)"
status: draft
---

# Route C: Parquet/Arrow Data-Lake Quickstart

Goal: run HistFactory-style fit workflow from Parquet/Arrow without ROOT in the user flow.

This route uses:

- `docs/guides/fixtures/route_c/histograms_table.example.csv`
- `docs/guides/fixtures/route_c/build_histograms_parquet_example.py`

## 0) Optional Python deps for this route

```bash
python3 -m pip install pyarrow polars duckdb
```

## 1) Build example Parquet from CSV table

```bash
python3 docs/guides/fixtures/route_c/build_histograms_parquet_example.py
```

This writes:

- `docs/guides/fixtures/route_c/histograms.parquet`

## 2) from_parquet -> fit (Python API)

```python
import nextstat

observations = {"SR": [55.0, 65.0]}

model = nextstat.from_parquet(
    "docs/guides/fixtures/route_c/histograms.parquet",
    poi="mu",
    observations=observations,
)

fit = nextstat.fit(model)
scan = nextstat.profile_scan(model, start=0.0, stop=5.0, points=41)
```

## 3) Polars/DuckDB -> Arrow -> fit (Python API)

```python
import duckdb
import nextstat
import polars as pl

# Polars -> Arrow
pl_df = pl.read_parquet("docs/guides/fixtures/route_c/histograms.parquet")
model_a = nextstat.from_arrow(pl_df.to_arrow(), poi="mu", observations={"SR": [55.0, 65.0]})

# DuckDB -> Arrow
con = duckdb.connect()
reader = con.execute("SELECT * FROM 'docs/guides/fixtures/route_c/histograms.parquet'").arrow()
model_b = nextstat.from_arrow(reader.read_all(), poi="mu", observations={"SR": [55.0, 65.0]})

fit_a = nextstat.fit(model_a)
fit_b = nextstat.fit(model_b)
```

## 4) Export results back to Parquet (scan/pulls/correlation)

```python
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

Path("tmp/guides").mkdir(parents=True, exist_ok=True)

with open("docs/guides/fixtures/route_c/scan_points.json", "r", encoding="utf-8") as f:
    scan = json.load(f)
with open("docs/guides/fixtures/route_c/pulls.json", "r", encoding="utf-8") as f:
    pulls = json.load(f)
with open("docs/guides/fixtures/route_c/corr.json", "r", encoding="utf-8") as f:
    corr = json.load(f)

rows = scan["points"]
scan_tbl = pa.table({
    "mu": [r["mu"] for r in rows],
    "q_mu": [r["q_mu"] for r in rows],
    "nll_mu": [r["nll_mu"] for r in rows],
    "converged": [r["converged"] for r in rows],
})

pq.write_table(scan_tbl, "tmp/guides/route_c_scan_points.parquet", compression="zstd")

pulls_tbl = pa.table({
    "name": [r["name"] for r in pulls["entries"]],
    "pull": [r["pull"] for r in pulls["entries"]],
    "postfit_sigma": [r.get("postfit_sigma") for r in pulls["entries"]],
})
pq.write_table(pulls_tbl, "tmp/guides/route_c_pulls.parquet", compression="zstd")

corr_rows = []
for i, row in enumerate(corr["corr"]):
    for j, value in enumerate(row):
        corr_rows.append({"i": i, "j": j, "value": value})
corr_tbl = pa.table({
    "i": [r["i"] for r in corr_rows],
    "j": [r["j"] for r in corr_rows],
    "value": [r["value"] for r in corr_rows],
})
pq.write_table(corr_tbl, "tmp/guides/route_c_corr.parquet", compression="zstd")
```

## Expected outputs (reference)

Compare with:

- `docs/guides/fixtures/route_c/fit_result.json`
- `docs/guides/fixtures/route_c/upper_limit_scan.json`
- `docs/guides/fixtures/route_c/scan_points.json`
- `docs/guides/fixtures/route_c/scan_points.csv`
- `docs/guides/fixtures/route_c/cls_curve.json`
- `docs/guides/fixtures/route_c/pulls.json`
- `docs/guides/fixtures/route_c/corr.json`
- `docs/guides/fixtures/route_c/reference_plot.png`
- `docs/guides/fixtures/route_c/validation_report_snippet.json`
