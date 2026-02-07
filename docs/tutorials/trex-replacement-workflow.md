---
title: "TREx Replacement Workflow (config → run → report)"
status: stable
---

# TREx Replacement Workflow (config → run → report)

This tutorial shows the “one command” path to replace TRExFitter runs using NextStat:

- **Input**: a HistFactory/TREx export directory (contains `combination.xml` + ROOT histograms)
- **Output**: deterministic, numbers-first report artifacts (JSON) + optional publication-ready **PDF/SVG**

For parity/baselines across ROOT/TREx, see `docs/tutorials/root-trexfitter-parity.md`.

## 1) Determinism and Parity Mode (non-negotiable for parity)

NextStat separates "specification correctness" from "speed" via two evaluation modes:

| Mode | Summation | Backend | Threads | Use Case |
|------|-----------|---------|---------|----------|
| **Parity** | Kahan compensated | SIMD (Accelerate disabled) | 1 (forced) | Validation vs pyhf / TREx baselines |
| **Fast** | Naive | SIMD / Accelerate / CUDA | Rayon (auto) | Production inference |

### Recommended: `--parity` flag

For reproducible parity comparisons, use:

```yaml
# analysis.yaml
execution:
  determinism: { threads: 1, parity: true }
```

Or via CLI:

```bash
nextstat fit --input workspace.json --parity
```

Or via Python:

```python
import nextstat
nextstat.set_eval_mode("parity")
```

When Parity mode is active:
1. **EvalMode::Parity** is set process-wide (atomic flag)
2. **Kahan compensated summation** replaces naive `+=` in Poisson NLL
3. **Apple Accelerate** is automatically disabled
4. **Thread count forced to 1** (sequential Rayon)
5. Results are **bit-exact reproducible** across runs

### Legacy: `threads: 1` only

Setting `execution.determinism.threads: 1` without `parity: true` gives stable ordering but uses naive summation. Use `--parity` instead for full determinism.

### Tolerance contract (Parity mode vs pyhf)

| Tier | Metric | Tolerance |
|------|--------|-----------|
| 1 | Per-bin expected data | 1e-12 |
| 3 | NLL value | 1e-8 atol |
| 5 | Best-fit params | 2e-4 |
| 6 | Uncertainties | 5e-4 |

Full 7-tier hierarchy: `docs/pyhf-parity-contract.md`.

### Kahan overhead

Confirmed **<5%** overhead vs naive summation at the same thread count across all tested workspaces (simple, complex, tHu 184-param, tttt 249-param).

## 2) (Optional) IDE schema for `analysis.yaml`

You can materialize the JSON schema locally and reference it from the YAML for IDE auto-complete:

```bash
nextstat config schema > analysis_spec_v0.schema.json
```

Then in your `analysis.yaml`:

```yaml
$schema: ./analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0
```

## 3) Create `analysis.yaml` for a HistFactory export directory

Minimal “import-only” example:

```yaml
schema_version: trex_analysis_spec_v0

analysis:
  name: "my-analysis"
  description: "HistFactory export → workspace"
  tags: ["trex-replacement"]

inputs:
  mode: histfactory_xml
  histfactory:
    export_dir: /path/to/export_dir
    combination_xml: null   # auto-discover `combination.xml` under export_dir
    measurement: NominalMeasurement

execution:
  determinism: { threads: 1, parity: true }

  import:
    enabled: true
    output_json: workspace.json

  fit: { enabled: false, output_json: fit.json }
  profile_scan: { enabled: false, start: 0.0, stop: 5.0, points: 21, output_json: scan.json }

  report:
    enabled: false
    out_dir: report/
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render: { enabled: false, pdf: null, svg_dir: null, python: null }
    skip_uncertainty: true
    uncertainty_grouping: prefix_1

gates:
  baseline_compare: { enabled: false, baseline_dir: tmp/baselines, require_same_host: true, max_slowdown: 1.3 }
```

Notes:
- `combination_xml: null` triggers bounded auto-discovery under `export_dir` with deterministic ordering.
- Bin edges for distributions come from the ROOT histograms referenced by `combination.xml`, so the plot geometry matches the source histograms exactly.

## 4) Validate (no execution)

```bash
nextstat validate --config analysis.yaml
```

This prints a JSON summary (and exits non-zero on validation errors).

## 5) Run (import → fit → report)

Enable `fit` and `report` to produce TREx-like artifacts:

```yaml
execution:
  determinism: { threads: 1, parity: true }
  import: { enabled: true, output_json: workspace.json }
  fit: { enabled: true, output_json: fit.json }
  profile_scan: { enabled: false, start: 0.0, stop: 5.0, points: 21, output_json: scan.json }
  report:
    enabled: true
    out_dir: report/
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render: { enabled: false, pdf: null, svg_dir: null, python: null }
    skip_uncertainty: true
    uncertainty_grouping: prefix_1
```

Run:

```bash
nextstat run --config analysis.yaml
```

Expected outputs (paths depend on your config):
- `workspace.json` (pyhf-compatible workspace)
- `fit.json` (MLE result)
- `report/` numeric artifacts like `distributions.json`, `yields.json`, `pulls.json`, `corr.json`

## 6) Repro bundle (inputs + hashes + outputs)

To capture a self-contained bundle with hashes and intermediate provenance:

```bash
nextstat --bundle run_bundle/ run --config analysis.yaml
```

The bundle contains:
- `meta.json` (high-level provenance)
- `provenance.json` (per-step output hashes)
- `manifest.json` (sha256 + bytes for every bundled file)
- `inputs/` (config + HistFactory XML/ROOT inputs best-effort)
- `outputs/` (workspace/fit/scan/report copied with stable names)

## 7) Publication-ready PDF/SVG rendering

Rendering is driven by numeric artifacts (JSON); it does **not** recompute numbers.

Enable rendering in `analysis.yaml`:

```yaml
execution:
  report:
    render:
      enabled: true
      pdf: report/report.pdf
      svg_dir: report/svg
      python: null
```

Requirements:
- a Python environment with `matplotlib` available (see project packaging notes; local `.venv` is recommended)

## 8) Next steps (parity baselines)

To enforce "identical numbers" against ROOT/TREx baselines, follow:
- `docs/tutorials/root-trexfitter-parity.md`
- `docs/pyhf-parity-contract.md` — 7-tier tolerance hierarchy (per-bin 1e-12 → toys 0.05)
- `docs/references/trex_replacement_parity_contract.md` — TREx replacement parity contract

Realistic TREx export dirs (with `combination.xml` + ROOT histograms) will be used to record baselines via `tests/record_baseline.py` (tracked separately in BMCP).

