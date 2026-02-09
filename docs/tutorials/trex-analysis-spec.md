---
title: "TREx Analysis Spec (YAML + JSON Schema)"
status: draft
created: 2026-02-07
---

# TREx Analysis Spec (YAML + JSON Schema)

Goal: give TREx-style workflows an **IDE-friendly** config:
- YAML (human-editable)
- JSON Schema (autocomplete + validation)
- one-command runner (import/fit/scan/report)

Files:
- Schema: `docs/schemas/trex/analysis_spec_v0.schema.json`
- Main example: `docs/specs/trex/analysis_spec_v0.yaml`
- Additional examples: `docs/specs/trex/examples/*.yaml`
- CLI validator/runner: `nextstat validate --config ...` / `nextstat run --config ...`
- Schema validator (optional): `scripts/trex/validate_analysis_spec.py`
- Schema-validated runner wrapper (optional): `scripts/trex/run_analysis_spec.py`
- Baseline recorder: `tests/record_trex_analysis_spec_baseline.py`
- Baseline compare: `tests/compare_trex_analysis_spec_with_latest_baseline.py`

## IDE autocomplete

Every spec should start with:

```yaml
$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0
```

Many YAML-capable IDEs (e.g. VS Code with a YAML extension) will then show:
- key suggestions
- enum dropdowns
- inline docs
- red squiggles on invalid configs

If your IDE cannot fetch schemas from the network, map the schema URL to the local file.
For VS Code (YAML extension), add a workspace setting similar to:

```json
{
  "yaml.schemas": {
    "docs/schemas/trex/analysis_spec_v0.schema.json": [
      "docs/specs/trex/**/*.yaml"
    ]
  }
}
```

## Path resolution

In v0, all relative paths in the spec are resolved relative to the spec file directory (not the current working directory).
This is why the repo examples under `docs/specs/trex/examples/` use `../../../../` prefixes to point back to the repo root.

## Validate a spec

Native (semantic) validation:

```sh
target/release/nextstat validate --config docs/specs/trex/analysis_spec_v0.yaml
```

Schema validation (JSON Schema; best for IDE + CI):

```sh
./.venv/bin/python scripts/trex/validate_analysis_spec.py \
  --spec docs/specs/trex/analysis_spec_v0.yaml
```

Validation errors are printed as `$`-paths (e.g. `$.execution.report.out_dir`).

Make target:

```sh
make trex-spec-validate TREX_SPEC=docs/specs/trex/analysis_spec_v0.yaml
```

## Run a spec (one command)

Native:

```sh
target/release/nextstat run --config docs/specs/trex/analysis_spec_v0.yaml
```

Wrapper (validates schema, then runs `nextstat run`; supports `--dry-run`):

```sh
./.venv/bin/python scripts/trex/run_analysis_spec.py \
  --spec docs/specs/trex/analysis_spec_v0.yaml
```

Make target:

```sh
make trex-spec-run TREX_SPEC=docs/specs/trex/analysis_spec_v0.yaml
```

Optional: pick an explicit `nextstat` binary:

```sh
./.venv/bin/python scripts/trex/run_analysis_spec.py \
  --spec docs/specs/trex/analysis_spec_v0.yaml \
  --nextstat target/release/nextstat
```

## Input modes (v0)

`inputs.mode` selects how the workspace is obtained:

1) `histfactory_xml` (TREx export directory)
- Uses `inputs.histfactory.export_dir` and `inputs.histfactory.combination_xml` (or auto-discovery).
- Supports full pipeline including `nextstat report` (needs `combination.xml` for bin edges).

2) `workspace_json`
- Uses an existing `pyhf` JSON workspace at `inputs.workspace_json.path`.
- In v0, `execution.import.enabled` must be `false`.
- If you enable `execution.report`, you must set `execution.report.histfactory_xml` explicitly.

3) `trex_config_txt`
- Wraps the existing line-based config consumed by `nextstat import trex-config` (subset; `ReadFrom=NTUP`).
- Useful as a bridge while migrating away from the text format.
- Note: report generation requires HistFactory XML; for ntuple mode use fit/scan first.
  - Tip: `nextstat import trex-config --analysis-yaml analysis.yaml --coverage-json coverage.json ...` can generate:
    - an analysis spec wrapper (`inputs.mode=trex_config_txt`) for `nextstat run`, and
    - a coverage report highlighting unknown keys/attrs in legacy configs.

4) `trex_config_yaml`
- IDE-friendly YAML representation of the same subset as `trex_config_txt`.
- Supported natively by `nextstat run --config ...` (it generates an equivalent internal text config and reuses the same importer).
- Migration tip: generate a starter spec from a TRExFitter `.config` via:
  - `nextstat trex import-config --config analysis.config --out analysis.yaml --report analysis.mapping.json`

## Examples

- HistFactory report-only: `docs/specs/trex/examples/histfactory_report_only.yaml`
- HistFactory fit + scan + report: `docs/specs/trex/examples/histfactory_fit_scan_report.yaml`
- Repo fixture smoke (local, no cluster): `docs/specs/trex/examples/histfactory_fixture_smoke.yaml`
- Existing workspace + report: `docs/specs/trex/examples/workspace_json_report.yaml`
- TREx text config + fit: `docs/specs/trex/examples/trex_config_txt_fit.yaml`
- TREx YAML config + fit: `docs/specs/trex/examples/trex_config_yaml_fit.yaml`

## Baselines (pre-release gate)

Baseline workflow mirrors Apex2: record once on a reference machine, then compare repeatedly.

Record:

```sh
make trex-spec-baseline-record TREX_SPEC=docs/specs/trex/analysis_spec_v0.yaml
```

Compare vs latest recorded baseline:

```sh
make trex-spec-baseline-compare TREX_COMPARE_ARGS="--require-same-host"
```

Outputs:
- Latest manifest pointer: `tmp/baselines/latest_trex_analysis_spec_manifest.json`
- Compare report: `tmp/trex_analysis_spec_compare_report.json`
