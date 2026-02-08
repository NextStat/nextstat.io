---
title: "Validation Report Artifacts"
status: partial
---

# Validation Report Artifacts

NextStat produces two categories of validation artifacts: **Apex2 JSON reports** (shipped) and a planned **unified validation report pack** (JSON + PDF).

## Apex2 JSON Reports (Shipped)

### Master Report

The master runner (`tests/apex2_master_report.py`) aggregates all validation suites into a single JSON:

```bash
PYTHONPATH=bindings/ns-py/python \
  python tests/apex2_master_report.py \
    --deterministic \
    --out tmp/apex2_master_report.json
```

**Top-level schema:**

```json
{
  "meta": {
    "timestamp": 1738944000,
    "python": "3.12.0",
    "platform": "macOS-26.2-arm64-arm-64bit",
    "wall_s": 12.3
  },
  "pyhf":                 { "status": "ok|fail|skipped", ... },
  "histfactory_golden":   { "status": "ok|fail|skipped", ... },
  "survival":             { "status": "ok|fail|skipped", ... },
  "survival_statsmodels": { "status": "ok|fail|skipped", ... },
  "regression_golden":    { "status": "ok|fail|skipped", ... },
  "nuts_quality":         { "status": "ok|fail|skipped", ... },
  "nuts_quality_report":  { "status": "ok|fail|skipped", ... },
  "p6_glm_bench":         { "status": "ok|fail|skipped", ... },
  "bias_pulls":           { "status": "ok|fail|skipped", ... },
  "sbc":                  { "status": "ok|fail|skipped", ... },
  "root":                 { "status": "ok|fail|skipped", ... }
}
```

Each section includes environment metadata (Python version, platform, package versions, seeds) and per-case pass/fail with numeric deltas.

### Deterministic Mode

The `--deterministic` flag strips non-reproducible keys (timestamps, wall-clock timings, subprocess output) and applies stable JSON key ordering via `_apex2_json.py`. This ensures bit-exact JSON output across runs on the same platform.

### Individual Runners

| Runner | Output | Purpose |
|--------|--------|---------|
| `apex2_pyhf_validation_report.py` | `tmp/apex2_pyhf_report.json` | NLL + expected_data parity vs pyhf |
| `apex2_root_suite_report.py` | `tmp/apex2_root_suite_report.json` | Profile scan parity vs ROOT/RooFit |
| `apex2_bias_pulls_report.py` | `tmp/apex2_bias_pulls_report.json` | Toy-based μ̂ bias and pull widths |
| `apex2_sbc_report.py` | `tmp/apex2_sbc_report.json` | SBC rank uniformity (NUTS) |
| `apex2_nuts_quality_report.py` | `tmp/apex2_nuts_quality_report.json` | Divergence rate, R-hat, ESS, E-BFMI |
| `apex2_p6_glm_benchmark_report.py` | `tmp/apex2_p6_glm_bench_report.json` | GLM fit/predict performance |
| `apex2_survival_statsmodels_report.py` | `tmp/apex2_survival_statsmodels_report.json` | Cox PH parity vs statsmodels |
| `aggregate_apex2_root_suite_reports.py` | `tmp/apex2_root_suite_aggregate.json` | Merge HTCondor per-case reports |

### HTCondor / Job Array Support

For large ROOT suite runs, each case can be dispatched as an independent job:

```bash
# Per-case (HTCondor array element):
python tests/apex2_root_suite_report.py \
  --case-index $PROCESS --cases cases.json --out tmp/apex2_root_case_${PROCESS}.json

# Aggregate:
python tests/aggregate_apex2_root_suite_reports.py \
  --in-dir tmp/ --glob 'apex2_root_case_*.json' --out tmp/apex2_root_suite_aggregate.json
```

## Analysis Report (`nextstat report`, Shipped)

The `nextstat report` CLI generates per-workspace analysis artifacts:

```bash
nextstat report \
  --input workspace.json \
  --histfactory-xml combination.xml \
  --out-dir report/ \
  [--fit fit.json] \
  [--render] \
  [--blind] \
  [--deterministic] \
  [--skip-uncertainty]
```

**Outputs:**

| File | Content |
|------|---------|
| `distributions.json` | Expected yields per channel/sample |
| `pulls.json` | Nuisance parameter pulls (θ̂ ± σ) |
| `corr.json` | Parameter correlation matrix |
| `yields.json` / `.csv` / `.tex` | Yield tables |
| `uncertainty.json` | NP ranking (systematic impact) |
| `fit.json` | MLE result (if `--fit` omitted) |

When `--render` is enabled, calls `python -m nextstat.report render` to produce:
- Multi-page PDF with all plots
- Per-plot SVGs

Requires `nextstat[viz]` extra (matplotlib).

## Validation Report Pack (Planned)

The planned `nextstat validation-report` command will produce a unified document combining Apex2 results with workspace-level metadata for regulated review workflows.

### Planned CLI

```bash
nextstat validation-report \
  --apex2 tmp/apex2_master_report.json \
  --workspace workspace.json \
  --out validation_report.json \
  [--pdf validation_report.pdf] \
  [--deterministic]
```

### Planned Schema (`validation_report.json`)

```json
{
  "schema_version": "validation_report_v1",
  "generated_at": "2026-02-08T17:00:00Z",
  "deterministic": true,

  "dataset_fingerprint": {
    "workspace_sha256": "a1b2c3...",
    "workspace_bytes": 12345,
    "channels": ["SR", "CR1", "CR2"],
    "n_channels": 3,
    "n_bins_per_channel": [10, 5, 5],
    "n_samples_total": 12,
    "n_parameters": 47,
    "observation_summary": {
      "total_observed": 1234.0,
      "min_bin": 0.0,
      "max_bin": 312.0
    }
  },

  "model_spec": {
    "poi": "mu",
    "parameters": [
      {"name": "mu", "init": 1.0, "bounds": [0.0, 10.0], "constraint": "none"},
      {"name": "alpha_JES", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": "normal"}
    ],
    "interpolation": {"normsys": "code4", "histosys": "code4p"},
    "objective": "poisson_nll_with_constraints",
    "optimizer": "lbfgsb",
    "eval_mode": "fast"
  },

  "environment": {
    "nextstat_version": "0.2.0",
    "nextstat_git_commit": "4844ed0",
    "python_version": "3.12.0",
    "platform": "macOS-26.2-arm64",
    "rust_toolchain": "1.85.0",
    "pyhf_version": "0.7.6",
    "determinism_settings": {"eval_mode": "fast", "threads": 8}
  },

  "apex2_summary": {
    "master_report_sha256": "d4e5f6...",
    "suites": {
      "pyhf":              {"status": "ok", "n_cases": 9, "worst_delta_nll": 1.2e-10},
      "histfactory_golden": {"status": "ok"},
      "regression_golden":  {"status": "ok", "n_cases": 8, "n_ok": 8},
      "survival":           {"status": "ok"},
      "nuts_quality":       {"status": "ok"},
      "root":               {"status": "ok", "n_cases": 3}
    },
    "overall": "pass"
  }
}
```

### Target Consumers

- **HEP**: analysis preservation, reinterpretation readiness (RECAST/pyhf ecosystem).
- **Pharma**: IQ/OQ/PQ validation packs for regulatory submission support (21 CFR Part 11 awareness).
- **FinTech**: model risk management documentation (SR 11-7 / SS1/23 model inventory).

### Open-Core Boundary

The OSS baseline provides:
- `validation_report.json` (machine-readable, deterministic)
- `validation_report.pdf` (matplotlib-rendered, basic layout)

Enterprise extensions (commercial license) may add:
- Branded PDF templates with corporate identity
- Digital signatures and timestamping
- Integration with audit trail / document management systems
- Automated scheduling and alerting
