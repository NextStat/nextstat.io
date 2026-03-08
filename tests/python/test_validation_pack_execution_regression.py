from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_nextstat_stub(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = _repo_root()
    log_path = tmp_path / "nextstat_calls.jsonl"
    stub_path = tmp_path / "nextstat_stub.py"
    _write_executable(
        stub_path,
        f"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

repo_root = Path({str(repo_root)!r})
log_path = Path({str(log_path)!r})
args = sys.argv[1:]
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")

def arg_value(flag: str) -> str:
    idx = args.index(flag)
    return args[idx + 1]

def write_json(path: str, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

def copy_example(path: str, name: str) -> None:
    Path(path).write_text(
        (repo_root / "docs" / "specs" / name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

if args[:1] == ["validation-report"]:
    payload = {{"schema_version": "validation_report_v1", "status": "ok"}}
    write_json(arg_value("--out"), payload)
    if "--pdf" in args:
        Path(arg_value("--pdf")).write_bytes(b"%PDF-1.4\\n% nextstat-stub\\n")
elif args[:2] == ["m15", "assessment-table"]:
    copy_example(arg_value("--output"), "m15_assessment_table_v1.example.json")
elif args[:2] == ["m15", "profile-diff"]:
    copy_example(arg_value("--output"), "m15_profile_diff_report_v1.example.json")
elif args[:2] == ["m15", "map"]:
    copy_example(arg_value("--output"), "m15_map_v1.example.json")
elif args[:2] == ["m15", "mar"]:
    copy_example(arg_value("--output"), "m15_mar_v1.example.json")
elif args[:2] == ["m15", "bundle"]:
    copy_example(arg_value("--output"), "m15_bundle_manifest_v1.example.json")
else:
    raise SystemExit(f"unexpected nextstat args: {{args}}")
""",
    )
    return stub_path, log_path


def _make_guard_python(tmp_path: Path) -> Path:
    stub_path = tmp_path / "guard_python.sh"
    _write_executable(
        stub_path,
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "-c" && "${{2:-}}" == "import matplotlib" ]]; then
  PYTHONPATH="" exec {sys.executable!r} "$@"
fi
if [[ "${{1:-}}" == *"tests/apex2_master_report.py" ]]; then
  echo "unexpected apex2 execution: $1" >&2
  exit 99
fi
if [[ "${{1:-}}" == *"tests/pharma_validation/runner.py" ]]; then
  echo "unexpected pharma execution: $1" >&2
  exit 98
fi
exec {sys.executable!r} "$@"
""",
    )
    return stub_path


def _make_bayes_design_sitecustomize(tmp_path: Path) -> Path:
    py_path = tmp_path / "pyguard"
    py_path.mkdir(parents=True, exist_ok=True)
    (py_path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import importlib
            import json
            from pathlib import Path

            bayes_design = importlib.import_module("nextstat.bayes_design")

            def _coerce_report(report_or_path):
                if isinstance(report_or_path, dict):
                    return report_or_path
                text = str(report_or_path)
                path = Path(text)
                if path.exists():
                    return json.loads(path.read_text(encoding="utf-8"))
                return json.loads(text)

            def _appendix(report_or_path, *, expected_family, expected_schema_version):
                report = _coerce_report(report_or_path)
                if report.get("design_family") != expected_family:
                    raise AssertionError(f"unexpected design family: {report.get('design_family')}")
                if report.get("schema_version") != expected_schema_version:
                    raise AssertionError(
                        f"unexpected report schema version: {report.get('schema_version')}"
                    )
                design_spec = report["design_spec"]
                return {
                    "schema_version": "nextstat_bayesian_design_regulatory_appendix_v0",
                    "appendix_id": f"{design_spec['design_id']}_appendix_guard",
                    "design_family": expected_family,
                    "design_id": design_spec["design_id"],
                    "source_report_schema_version": expected_schema_version,
                    "generated_from_frozen_report": True,
                    "required_sections": ["design_summary", "provenance"],
                    "section_order": ["design_summary", "provenance"],
                    "sections": {
                        "design_summary": {
                            "design_id": design_spec["design_id"],
                            "current_look_id": report["current_analysis"]["look"]["id"],
                        },
                        "provenance": {
                            "generator": "sitecustomize-guard",
                            "source_report_seed": report["provenance"]["simulation_seed"],
                        },
                    },
                }

            def _build_beta(report_or_path):
                return _appendix(
                    report_or_path,
                    expected_family="beta_binomial",
                    expected_schema_version="nextstat_beta_binomial_design_report_v0",
                )

            def _build_normal(report_or_path):
                return _appendix(
                    report_or_path,
                    expected_family="normal_normal",
                    expected_schema_version="nextstat_normal_normal_design_report_v0",
                )

            def _forbidden(*args, **kwargs):
                raise AssertionError("unexpected hidden execution")

            bayes_design.build_beta_binomial_regulatory_appendix = _build_beta
            bayes_design.build_normal_normal_regulatory_appendix = _build_normal
            bayes_design.render_bayesian_regulatory_appendix_markdown = (
                lambda appendix_or_path: "# Frozen appendix markdown guard\\n"
            )
            bayes_design.write_bayesian_regulatory_appendix_pdf = (
                lambda pdf_path, appendix_or_path: Path(pdf_path).write_bytes(b"%PDF-1.4\\n% appendix-guard\\n")
            )

            for name in (
                "analyze_beta_binomial_design",
                "simulate_beta_binomial_design",
                "forecast_beta_binomial_design",
                "analyze_beta_binomial_prior_sensitivity",
                "build_beta_binomial_design_report",
                "render_beta_binomial_design_report",
                "write_beta_binomial_design_report_bundle",
                "analyze_normal_normal_design",
                "simulate_normal_normal_design",
                "forecast_normal_normal_design",
                "analyze_normal_normal_prior_sensitivity",
                "build_normal_normal_design_report",
                "render_normal_normal_design_report",
                "write_normal_normal_design_report_bundle",
            ):
                setattr(bayes_design, name, _forbidden)
            """
        ),
        encoding="utf-8",
    )
    return py_path


def _bayes_design_pythonpath(sitecustomize_dir: Path) -> str:
    repo_root = _repo_root()
    return os.pathsep.join(
        [
            str(sitecustomize_dir),
            str(repo_root / "bindings" / "ns-py" / "python"),
            *( [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [] ),
        ]
    )


def _read_nextstat_calls(log_path: Path) -> list[list[str]]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_validation_pack_non_m15_path_remains_backward_compatible(tmp_path: Path) -> None:
    repo_root = _repo_root()
    out_dir = tmp_path / "artifacts"
    nextstat_stub, log_path = _make_nextstat_stub(tmp_path)

    subprocess.run(
        [
            "bash",
            "validation-pack/render_validation_pack.sh",
            "--out-dir",
            str(out_dir),
            "--workspace",
            "tests/fixtures/simple_workspace.json",
            "--apex2-master",
            "tests/fixtures/apex2_master_min_plus.json",
            "--python",
            sys.executable,
            "--nextstat-bin",
            str(nextstat_stub),
            "--deterministic",
            "--json-only",
            "--skip-pharma-validation",
        ],
        check=True,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads((out_dir / "validation_pack_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = [entry["path"] for entry in manifest["files"]]

    assert manifest_paths == [
        "apex2_master_report.json",
        "validation_report.json",
        "validation_report_v1.schema.json",
    ]
    assert not any(path.startswith("m15_") for path in manifest_paths)
    assert not (out_dir / "pharma_validation.json").exists()
    assert _read_nextstat_calls(log_path) == [
        [
            "validation-report",
            "--apex2",
            str(out_dir / "apex2_master_report.json"),
            "--workspace",
            str(repo_root / "tests/fixtures/simple_workspace.json"),
            "--out",
            str(out_dir / "validation_report.json"),
            "--deterministic",
        ]
    ]


def test_validation_pack_bayesian_report_pdf_integration_uses_frozen_renderers_only(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    out_dir = tmp_path / "artifacts"
    nextstat_stub, log_path = _make_nextstat_stub(tmp_path)
    guard_python = _make_guard_python(tmp_path)
    sitecustomize_dir = _make_bayes_design_sitecustomize(tmp_path)
    report_path = repo_root / "docs/specs/pharma/beta_binomial_design_report_v0.example.json"

    subprocess.run(
        [
            "bash",
            "validation-pack/render_validation_pack.sh",
            "--out-dir",
            str(out_dir),
            "--workspace",
            "tests/fixtures/simple_workspace.json",
            "--apex2-master",
            "tests/fixtures/apex2_master_min_plus.json",
            "--python",
            str(guard_python),
            "--nextstat-bin",
            str(nextstat_stub),
            "--deterministic",
            "--skip-pharma-validation",
            "--bayesian-design-report",
            str(report_path),
        ],
        check=True,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": _bayes_design_pythonpath(sitecustomize_dir)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads((out_dir / "validation_pack_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = [entry["path"] for entry in manifest["files"]]

    assert manifest_paths == [
        "apex2_master_report.json",
        "bayesian_design_regulatory_appendix.json",
        "bayesian_design_regulatory_appendix.md",
        "bayesian_design_regulatory_appendix.pdf",
        "bayesian_design_regulatory_appendix_v0.schema.json",
        "bayesian_design_report.json",
        "beta_binomial_design_report_v0.schema.json",
        "validation_report.json",
        "validation_report.pdf",
        "validation_report_v1.schema.json",
    ]
    assert (out_dir / "bayesian_design_regulatory_appendix.md").read_text(encoding="utf-8") == (
        "# Frozen appendix markdown guard\n"
    )
    assert (out_dir / "bayesian_design_regulatory_appendix.pdf").read_bytes().startswith(b"%PDF-")

    assert _read_nextstat_calls(log_path) == [
        [
            "validation-report",
            "--apex2",
            str(out_dir / "apex2_master_report.json"),
            "--workspace",
            str(repo_root / "tests/fixtures/simple_workspace.json"),
            "--out",
            str(out_dir / "validation_report.json"),
            "--pdf",
            str(out_dir / "validation_report.pdf"),
            "--python",
            str(guard_python),
            "--deterministic",
        ]
    ]


def test_validation_pack_m15_requires_preseeded_pharma_when_skipped(tmp_path: Path) -> None:
    repo_root = _repo_root()
    out_dir = tmp_path / "artifacts"
    nextstat_stub, log_path = _make_nextstat_stub(tmp_path)

    proc = subprocess.run(
        [
            "bash",
            "validation-pack/render_validation_pack.sh",
            "--out-dir",
            str(out_dir),
            "--workspace",
            "tests/fixtures/simple_workspace.json",
            "--apex2-master",
            "tests/fixtures/apex2_master_min_plus.json",
            "--python",
            sys.executable,
            "--nextstat-bin",
            str(nextstat_stub),
            "--deterministic",
            "--json-only",
            "--skip-pharma-validation",
            "--m15-config",
            "docs/specs/m15_config_v1.example.json",
        ],
        check=False,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.returncode == 2
    assert "M15 artifact generation requires" in proc.stderr
    assert "Either omit --skip-pharma-validation or pre-seed that file" in proc.stderr
    assert _read_nextstat_calls(log_path) == [
        [
            "validation-report",
            "--apex2",
            str(out_dir / "apex2_master_report.json"),
            "--workspace",
            str(repo_root / "tests/fixtures/simple_workspace.json"),
            "--out",
            str(out_dir / "validation_report.json"),
            "--deterministic",
        ]
    ]


def test_validation_pack_m15_uses_preseeded_artifacts_without_hidden_execution(tmp_path: Path) -> None:
    repo_root = _repo_root()
    out_dir = tmp_path / "artifacts"
    nextstat_stub, log_path = _make_nextstat_stub(tmp_path)
    guard_python = _make_guard_python(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pharma_validation.json").write_text(
        json.dumps(
            {
                "schema_version": "nextstat.pharma_validation.v1",
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "bash",
            "validation-pack/render_validation_pack.sh",
            "--out-dir",
            str(out_dir),
            "--workspace",
            "tests/fixtures/simple_workspace.json",
            "--apex2-master",
            "tests/fixtures/apex2_master_min_plus.json",
            "--python",
            str(guard_python),
            "--nextstat-bin",
            str(nextstat_stub),
            "--deterministic",
            "--json-only",
            "--skip-pharma-validation",
            "--m15-config",
            "docs/specs/m15_config_v1.example.json",
        ],
        check=True,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads((out_dir / "validation_pack_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = [entry["path"] for entry in manifest["files"]]

    assert manifest_paths == [
        "apex2_master_report.json",
        "m15_assessment_table.json",
        "m15_bundle_manifest.json",
        "m15_bundle_manifest_v1.schema.json",
        "m15_config.json",
        "m15_map.json",
        "m15_mar.json",
        "m15_profile_diff_report.json",
        "m15_profile_diff_report_v1.schema.json",
        "pharma_validation.json",
        "validation_report.json",
        "validation_report_v1.schema.json",
    ]
    assert _read_nextstat_calls(log_path) == [
        [
            "validation-report",
            "--apex2",
            str(out_dir / "apex2_master_report.json"),
            "--workspace",
            str(repo_root / "tests/fixtures/simple_workspace.json"),
            "--out",
            str(out_dir / "validation_report.json"),
            "--deterministic",
        ],
        [
            "m15",
            "profile-diff",
            "--config",
            str(out_dir / "m15_config.json"),
            "--output",
            str(out_dir / "m15_profile_diff_report.json"),
            "--deterministic",
        ],
        [
            "m15",
            "assessment-table",
            "--config",
            str(out_dir / "m15_config.json"),
            "--validation-report",
            str(out_dir / "validation_report.json"),
            "--pharma-validation",
            str(out_dir / "pharma_validation.json"),
            "--output",
            str(out_dir / "m15_assessment_table.json"),
            "--deterministic",
        ],
        [
            "m15",
            "map",
            "--config",
            str(out_dir / "m15_config.json"),
            "--assessment-table",
            str(out_dir / "m15_assessment_table.json"),
            "--output",
            str(out_dir / "m15_map.json"),
            "--deterministic",
        ],
        [
            "m15",
            "mar",
            "--map",
            str(out_dir / "m15_map.json"),
            "--assessment-table",
            str(out_dir / "m15_assessment_table.json"),
            "--validation-report",
            str(out_dir / "validation_report.json"),
            "--pharma-validation",
            str(out_dir / "pharma_validation.json"),
            "--output",
            str(out_dir / "m15_mar.json"),
            "--deterministic",
        ],
        [
            "m15",
            "bundle",
            "--config",
            str(out_dir / "m15_config.json"),
            "--assessment-table",
            str(out_dir / "m15_assessment_table.json"),
            "--map",
            str(out_dir / "m15_map.json"),
            "--mar",
            str(out_dir / "m15_mar.json"),
            "--validation-report",
            str(out_dir / "validation_report.json"),
            "--pharma-validation",
            str(out_dir / "pharma_validation.json"),
            "--output",
            str(out_dir / "m15_bundle_manifest.json"),
            "--deterministic",
        ],
    ]


def test_validation_pack_m15_publishable_artifacts_render_from_frozen_artifacts(tmp_path: Path) -> None:
    repo_root = _repo_root()
    out_dir = tmp_path / "artifacts"
    nextstat_stub, log_path = _make_nextstat_stub(tmp_path)
    guard_python = _make_guard_python(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pharma_validation.json").write_text(
        json.dumps(
            {
                "schema_version": "nextstat.pharma_validation.v1",
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "bash",
            "validation-pack/render_validation_pack.sh",
            "--out-dir",
            str(out_dir),
            "--workspace",
            "tests/fixtures/simple_workspace.json",
            "--apex2-master",
            "tests/fixtures/apex2_master_min_plus.json",
            "--python",
            str(guard_python),
            "--nextstat-bin",
            str(nextstat_stub),
            "--deterministic",
            "--skip-pharma-validation",
            "--m15-config",
            "docs/specs/m15_config_v1.example.json",
        ],
        check=True,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads((out_dir / "validation_pack_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = [entry["path"] for entry in manifest["files"]]

    assert manifest_paths == [
        "apex2_master_report.json",
        "m15_assessment_table.json",
        "m15_bundle_manifest.json",
        "m15_bundle_manifest_v1.schema.json",
        "m15_config.json",
        "m15_map.json",
        "m15_mar.json",
        "m15_profile_diff_report.json",
        "m15_profile_diff_report_v1.schema.json",
        "m15_report.docx",
        "m15_report.md",
        "m15_report.pdf",
        "pharma_validation.json",
        "validation_report.json",
        "validation_report.pdf",
        "validation_report_v1.schema.json",
    ]
    assert (out_dir / "m15_report.md").read_text(encoding="utf-8").startswith("# ICH M15 Publishable Report")
    assert (out_dir / "m15_report.pdf").read_bytes().startswith(b"%PDF-")
    assert (out_dir / "m15_report.docx").stat().st_size > 0
    assert _read_nextstat_calls(log_path) == [
        [
            "validation-report",
            "--apex2",
            str(out_dir / "apex2_master_report.json"),
            "--workspace",
            str(repo_root / "tests/fixtures/simple_workspace.json"),
            "--out",
            str(out_dir / "validation_report.json"),
            "--pdf",
            str(out_dir / "validation_report.pdf"),
            "--python",
            str(guard_python),
            "--deterministic",
        ],
        [
            "m15",
            "profile-diff",
            "--config",
            str(out_dir / "m15_config.json"),
            "--output",
            str(out_dir / "m15_profile_diff_report.json"),
            "--deterministic",
        ],
        [
            "m15",
            "assessment-table",
            "--config",
            str(out_dir / "m15_config.json"),
            "--validation-report",
            str(out_dir / "validation_report.json"),
            "--pharma-validation",
            str(out_dir / "pharma_validation.json"),
            "--output",
            str(out_dir / "m15_assessment_table.json"),
            "--deterministic",
        ],
        [
            "m15",
            "map",
            "--config",
            str(out_dir / "m15_config.json"),
            "--assessment-table",
            str(out_dir / "m15_assessment_table.json"),
            "--output",
            str(out_dir / "m15_map.json"),
            "--deterministic",
        ],
        [
            "m15",
            "mar",
            "--map",
            str(out_dir / "m15_map.json"),
            "--assessment-table",
            str(out_dir / "m15_assessment_table.json"),
            "--validation-report",
            str(out_dir / "validation_report.json"),
            "--pharma-validation",
            str(out_dir / "pharma_validation.json"),
            "--output",
            str(out_dir / "m15_mar.json"),
            "--deterministic",
        ],
        [
            "m15",
            "bundle",
            "--config",
            str(out_dir / "m15_config.json"),
            "--assessment-table",
            str(out_dir / "m15_assessment_table.json"),
            "--map",
            str(out_dir / "m15_map.json"),
            "--mar",
            str(out_dir / "m15_mar.json"),
            "--validation-report",
            str(out_dir / "validation_report.json"),
            "--pharma-validation",
            str(out_dir / "pharma_validation.json"),
            "--output",
            str(out_dir / "m15_bundle_manifest.json"),
            "--deterministic",
        ],
    ]


@pytest.mark.parametrize(
    ("report_relpath", "expected_family", "expected_schema_path"),
    [
        (
            "docs/specs/pharma/beta_binomial_design_report_v0.example.json",
            "beta_binomial",
            "beta_binomial_design_report_v0.schema.json",
        ),
        (
            "docs/specs/pharma/normal_normal_design_report_v0.example.json",
            "normal_normal",
            "normal_normal_design_report_v0.schema.json",
        ),
    ],
)
def test_validation_pack_bayesian_report_integration_uses_frozen_appendix_only(
    tmp_path: Path,
    report_relpath: str,
    expected_family: str,
    expected_schema_path: str,
) -> None:
    repo_root = _repo_root()
    out_dir = tmp_path / "artifacts"
    nextstat_stub, log_path = _make_nextstat_stub(tmp_path)
    guard_python = _make_guard_python(tmp_path)
    sitecustomize_dir = _make_bayes_design_sitecustomize(tmp_path)
    report_path = repo_root / report_relpath

    subprocess.run(
        [
            "bash",
            "validation-pack/render_validation_pack.sh",
            "--out-dir",
            str(out_dir),
            "--workspace",
            "tests/fixtures/simple_workspace.json",
            "--apex2-master",
            "tests/fixtures/apex2_master_min_plus.json",
            "--python",
            str(guard_python),
            "--nextstat-bin",
            str(nextstat_stub),
            "--deterministic",
            "--json-only",
            "--skip-pharma-validation",
            "--bayesian-design-report",
            str(report_path),
        ],
        check=True,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": _bayes_design_pythonpath(sitecustomize_dir)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads((out_dir / "validation_pack_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = [entry["path"] for entry in manifest["files"]]

    assert manifest_paths == [
        "apex2_master_report.json",
        "bayesian_design_regulatory_appendix.json",
        "bayesian_design_regulatory_appendix.md",
        "bayesian_design_regulatory_appendix_v0.schema.json",
        "bayesian_design_report.json",
        expected_schema_path,
        "validation_report.json",
        "validation_report_v1.schema.json",
    ]
    assert (out_dir / "bayesian_design_report.json").read_text(encoding="utf-8") == report_path.read_text(
        encoding="utf-8"
    )

    appendix = json.loads(
        (out_dir / "bayesian_design_regulatory_appendix.json").read_text(encoding="utf-8")
    )
    assert appendix["schema_version"] == "nextstat_bayesian_design_regulatory_appendix_v0"
    assert appendix["design_family"] == expected_family
    assert appendix["generated_from_frozen_report"] is True
    assert appendix["sections"]["provenance"]["generator"] == "sitecustomize-guard"

    assert _read_nextstat_calls(log_path) == [
        [
            "validation-report",
            "--apex2",
            str(out_dir / "apex2_master_report.json"),
            "--workspace",
            str(repo_root / "tests/fixtures/simple_workspace.json"),
            "--out",
            str(out_dir / "validation_report.json"),
            "--deterministic",
        ]
    ]
