# Adoption Playbook Fixtures

Reference outputs used by the 3 adoption routes:

- `route_a/` — pyhf/HS3 workspace quickstart outputs
- `route_b/` — TREx config quickstart outputs
- `route_c/` — Parquet/Arrow quickstart outputs

Each route directory includes:

- fit output (`fit_result.json`)
- profile scan output (`scan_points.json`, `scan_points.csv`)
- CLs / upper-limit outputs (`cls_curve.json`, `upper_limit_scan.json`)
- nuisance diagnostics (`pulls.json`, `corr.json`)
- plot preview (`reference_plot.png`)
- validation summary (`validation_report_snippet.json`)

Route C additionally includes a table fixture:

- `histograms_table.example.csv`
- `build_histograms_parquet_example.py`

These fixtures are deterministic references for documentation workflows and manual regression checks.
