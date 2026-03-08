---
title: "ICH M15 Reporting Artifacts"
status: draft
---

# ICH M15 Reporting Artifacts

This repo defines versioned JSON contracts for an `ICH M15` reporting layer that sits on top of the existing deterministic validation stack.

Current schema surfaces:

- `m15_config_v1` — planning/config input for M15 artifact generation
- `m15_assessment_table_v1` — per-question assessment table artifact
- `m15_map_v1` — Model Analysis Plan contract
- `m15_mar_v1` — Model Analysis Report contract
- `m15_profile_diff_report_v1` — operator-facing jurisdiction profile diff contract
- `m15_bundle_manifest_v1` — bundle integrity/packaging contract

Supported jurisdiction profiles:

- `ich_core`
- `ema_step5_2026`
- `fda_draft_2024`

Canonical examples in `docs/specs/m15_*.example.json` are deterministic CLI-generated outputs for the shipped example inputs. Changes to the runtime M15 JSON surface should keep these examples byte-stable or explicitly update them in the same change.

All of these schemas are available from the CLI:

```bash
nextstat config schema --name m15_config_v1
nextstat config schema --name m15_assessment_table_v1
nextstat config schema --name m15_map_v1
nextstat config schema --name m15_mar_v1
nextstat config schema --name m15_profile_diff_report_v1
nextstat config schema --name m15_bundle_manifest_v1
```

Benchmark gate for runtime-affecting changes:

- `docs/benchmarks/m15-reporting-runtime-gate.md`
- `scripts/benchmarks/bench_m15_reporting.py`
- `scripts/benchmarks/bench_m15_reporting_remote.sh`
- `scripts/benchmarks/m15_reporting_stable_surface_gate.sh`
- `docs/specs/pharma/m15_reporting_benchmark_result_v1.example.json`
- `scripts/benchmarks/compare_m15_reporting_benchmark.py`
- `benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json`
- `docs/specs/pharma/m15_reporting_benchmark_compare_report_v1.example.json`
- `.github/workflows/m15-reporting-stable-surface.yml`

Current shipped generator surface:

```bash
nextstat m15 assessment-table \
  --config docs/specs/m15_config_v1.example.json \
  --validation-report docs/specs/validation_report_v1.example.json \
  --pharma-validation tests/fixtures/pharma_validation_ok.json \
  --deterministic

nextstat m15 map \
  --config docs/specs/m15_config_v1.example.json \
  --assessment-table artifacts/m15_assessment_table.json \
  --deterministic

nextstat m15 mar \
  --map artifacts/m15_map.json \
  --assessment-table artifacts/m15_assessment_table.json \
  --validation-report artifacts/validation_report.json \
  --pharma-validation artifacts/pharma_validation.json \
  --deterministic

nextstat m15 profile-diff \
  --config docs/specs/m15_config_v1.example.json \
  --deterministic

nextstat m15 bundle \
  --config docs/specs/m15_config_v1.example.json \
  --assessment-table artifacts/m15_assessment_table.json \
  --map artifacts/m15_map.json \
  --mar artifacts/m15_mar.json \
  --validation-report artifacts/validation_report.json \
  --pharma-validation artifacts/pharma_validation.json \
  --deterministic

python -m nextstat.m15_report render \
  --assessment-table artifacts/m15_assessment_table.json \
  --map artifacts/m15_map.json \
  --mar artifacts/m15_mar.json \
  --profile-diff artifacts/m15_profile_diff_report.json \
  --bundle artifacts/m15_bundle_manifest.json \
  --markdown artifacts/m15_report.md \
  --pdf artifacts/m15_report.pdf \
  --docx artifacts/m15_report.docx

bash validation-pack/render_validation_pack.sh \
  --out-dir tmp/validation_pack_m15 \
  --workspace tests/fixtures/simple_workspace.json \
  --apex2-master tests/fixtures/apex2_master_min_plus.json \
  --m15-config docs/specs/m15_config_v1.example.json \
  --skip-pharma-validation \
  --json-only \
  --deterministic
```

Example payloads live in:

- `docs/specs/m15_config_v1.example.json`
- `docs/specs/m15_assessment_table_v1.example.json`
- `docs/specs/m15_map_v1.example.json`
- `docs/specs/m15_mar_v1.example.json`
- `docs/specs/m15_profile_diff_report_v1.example.json`
- `docs/specs/m15_bundle_manifest_v1.example.json`

Design constraints for this layer:

- Human judgment fields are explicit, never inferred from validation pass/fail alone.
- Artifacts are versioned and deterministic-friendly (`generated_at` may be `null` in deterministic mode).
- `assessment-table`, `MAP`, and `MAR` carry explicit `profile_requirements` payloads so profile-specific wording and mandatory sections stay aligned across JSON, Markdown, examples, and release assets.
- `profile-diff` is a separate operator artifact: it compares the shipped profile matrix across `assessment-table`, `MAP`, and `MAR` without mutating bundle integrity semantics.
- M15 reporting reuses the existing validation stack instead of rerunning model execution inside rendering/bundling steps.
- `nextstat m15 bundle` verifies deterministic re-render of `assessment-table`, `map`, and `mar` against the supplied artifacts before marking a bundle `complete`.
- `m15_config_v1.review_plan` requires explicit `primary_author`, `qa_reviewer`, and `approver` roles, and these roles must be assigned to distinct people.
- `assessment-table.review_status`, `map.signoff`, and `mar.document_status` are driven by explicit signoff state, but `MAR` remains bounded by technical evidence: unmet criteria or open deviations force `draft`.
- `validation-pack/render_validation_pack.sh --m15-config ...` copies the supplied config into the output directory and emits a self-contained M15 artifact set (`m15_config.json`, assessment-table, MAP, MAR, profile diff report, bundle manifest).
- `python -m nextstat.m15_report render ...` renders a publishable combined report from frozen M15 artifacts only and writes deterministic `m15_report.md`, `m15_report.pdf`, and `m15_report.docx`.
- `.github/workflows/python-tests.yml` and `.github/workflows/release.yml` publish M15 integrity sidecars too: `m15_bundle_manifest.sha256`, `m15_bundle_manifest.sha256.bin`, and `m15_snapshot_index.json`.

## PR and Release Gates

Any PR that changes the public M15 surface should treat the following as required merge gates:

- Keep schemas, examples, and `docs/references/m15-reporting.md` in sync.
- Preserve deterministic rerender for `assessment-table`, `map`, `mar`, `profile-diff`, and `bundle`.
- Do not introduce hidden model execution into the M15 render/bundle path.
- Keep CI and release workflow parity for the M15 validation-pack path.
- Keep the M15 benchmark promotion gate wired through `.github/workflows/m15-reporting-stable-surface.yml`, `.github/workflows/python-tests.yml`, and `.github/workflows/release.yml`.
- Preserve the author-reviewer-approver flow: distinct signoff roles, explicit signoff status in MAP/MAR, and bundle completion only for `reviewed` or `approved` MAR artifacts with complete signoff metadata.

Recommended targeted checks:

```bash
pytest -q \
  tests/python/test_m15_artifact_schema_smoke.py \
  tests/python/test_m15_reporting_benchmark_smoke.py \
  tests/python/test_python_tests_workflow_m15_smoke.py \
  tests/python/test_release_workflow_m15_smoke.py \
  tests/python/test_validation_pack_script_smoke.py

cargo test -p ns-cli \
  --test cli_m15_assessment_table \
  --test cli_m15_map \
  --test cli_m15_mar \
  --test cli_m15_profile_diff \
  --test cli_m15_bundle \
  --test cli_m15_profiles

python3 scripts/benchmarks/compare_m15_reporting_benchmark.py \
  --current benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json

make m15-reporting-stable-surface-gate
```

Maintainer release gate for M15-affecting releases:

- Confirm release workflow still renders the M15 validation-pack path.
- Confirm M15 schema validation and determinism checks are green.
- Confirm the M15 benchmark promotion gate stays `passed` against the accepted release baseline on `nextstat-bench`.
- Confirm the dedicated workflow `.github/workflows/m15-reporting-stable-surface.yml` remains aligned with `python-tests.yml` and `release.yml`.
- Confirm release assets include `m15_profile_diff_report.json`, `m15_profile_diff_report_v1.schema.json`, `m15_report.md`, `m15_report.pdf`, `m15_report.docx`, `m15_bundle_manifest.json`, `m15_bundle_manifest_v1.schema.json`, `m15_bundle_manifest.sha256`, `m15_bundle_manifest.sha256.bin`, and `m15_snapshot_index.json`.
