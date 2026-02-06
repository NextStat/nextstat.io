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
- Validator: `scripts/trex/validate_analysis_spec.py`
- Runner: `scripts/trex/run_analysis_spec.py`

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

## Validate a spec

```sh
./.venv/bin/python scripts/trex/validate_analysis_spec.py \
  --spec docs/specs/trex/analysis_spec_v0.yaml
```

Validation errors are printed as `$`-paths (e.g. `$.execution.report.out_dir`).

## Run a spec (one command)

Dry-run (prints commands):

```sh
./.venv/bin/python scripts/trex/run_analysis_spec.py \
  --spec docs/specs/trex/analysis_spec_v0.yaml \
  --dry-run
```

Execute:

```sh
./.venv/bin/python scripts/trex/run_analysis_spec.py \
  --spec docs/specs/trex/analysis_spec_v0.yaml
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

4) `trex_config_yaml`
- IDE-friendly YAML representation of the same subset as `trex_config_txt`.
- The runner generates a temporary text config and calls `nextstat import trex-config`.

## Examples

- HistFactory report-only: `docs/specs/trex/examples/histfactory_report_only.yaml`
- HistFactory fit + scan + report: `docs/specs/trex/examples/histfactory_fit_scan_report.yaml`
- Existing workspace + report: `docs/specs/trex/examples/workspace_json_report.yaml`
- TREx text config + fit: `docs/specs/trex/examples/trex_config_txt_fit.yaml`
- TREx YAML config + fit: `docs/specs/trex/examples/trex_config_yaml_fit.yaml`

