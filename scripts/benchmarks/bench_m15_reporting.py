#!/usr/bin/env python3
"""Deterministic benchmark harness for the M15 reporting surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "bench_results" / "m15_reporting" / "summary.json"
DEFAULT_WORK_ROOT = DEFAULT_OUT.parent / "work"
DEFAULT_CONFIG = REPO_ROOT / "docs" / "specs" / "m15_config_v1.example.json"
DEFAULT_VALIDATION_REPORT = REPO_ROOT / "docs" / "specs" / "validation_report_v1.example.json"
DEFAULT_PHARMA_VALIDATION = REPO_ROOT / "tests" / "fixtures" / "pharma_validation_ok.json"
DEFAULT_APEX2_MASTER = REPO_ROOT / "tests" / "fixtures" / "apex2_master_min_plus.json"
DEFAULT_WORKSPACE = REPO_ROOT / "tests" / "fixtures" / "simple_workspace.json"
VALIDATION_PACK_SCRIPT = REPO_ROOT / "validation-pack" / "render_validation_pack.sh"
SUITE = "m15_reporting"
SCHEMA_VERSION = "nextstat.m15_reporting_benchmark_result.v1"
HOST_POLICY = "nextstat-bench"
CASE_ORDER = [
    "m15_assessment_table",
    "m15_map",
    "m15_mar",
    "m15_bundle",
    "validation_pack_base_json_only",
    "validation_pack_m15_json_only",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str:
    linux_cpuinfo = Path("/proc/cpuinfo")
    if linux_cpuinfo.exists():
        for line in linux_cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                return value.strip()
    machine = platform.processor().strip() or platform.machine().strip()
    return machine or "unknown"


def _git_commit() -> str | None:
    env_value = os.environ.get("NEXTSTAT_BENCH_GIT_COMMIT", "").strip()
    if env_value:
        return env_value
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _build_profile(binary: Path) -> str:
    parts = set(binary.parts)
    if "release" in parts:
        return "release"
    if "debug" in parts:
        return "debug"
    return "unknown"


def _relative_or_name(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def _timing_doc(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("expected at least one sample")
    return {
        "min_s": round(min(samples), 6),
        "median_s": round(statistics.median(samples), 6),
        "max_s": round(max(samples), 6),
        "samples_s": [round(sample, 6) for sample in samples],
    }


def _run(cmd: list[str], *, cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return elapsed, proc


def _validated_artifact(path: Path, expected_schema_version: str, case_root: Path) -> dict[str, Any]:
    doc = _load_json(path)
    observed = str(doc.get("schema_version", ""))
    return {
        "expected_schema_version": expected_schema_version,
        "observed_schema_version": observed,
        "path": _relative_or_name(path, case_root),
        "status": "ok" if observed == expected_schema_version else "fail",
    }


def _validation_plan(path: Path, expected_schema_version: str) -> tuple[Path, str]:
    return path, expected_schema_version


def _assert_validations(validations: list[dict[str, Any]]) -> None:
    failures = [item for item in validations if item["status"] != "ok"]
    if failures:
        raise RuntimeError(f"schema version validation failed: {json.dumps(failures, indent=2)}")


def _build_setup_artifacts(nextstat_bin: Path, work_root: Path, deterministic: bool) -> dict[str, Path]:
    setup_root = work_root / "setup"
    setup_root.mkdir(parents=True, exist_ok=True)
    assessment_path = setup_root / "m15_assessment_table.json"
    map_path = setup_root / "m15_map.json"
    mar_path = setup_root / "m15_mar.json"

    assessment_cmd = [
        str(nextstat_bin),
        "m15",
        "assessment-table",
        "--config",
        str(DEFAULT_CONFIG),
        "--validation-report",
        str(DEFAULT_VALIDATION_REPORT),
        "--pharma-validation",
        str(DEFAULT_PHARMA_VALIDATION),
        "--output",
        str(assessment_path),
    ]
    if deterministic:
        assessment_cmd.append("--deterministic")
    _run(assessment_cmd, cwd=REPO_ROOT)

    map_cmd = [
        str(nextstat_bin),
        "m15",
        "map",
        "--config",
        str(DEFAULT_CONFIG),
        "--assessment-table",
        str(assessment_path),
        "--output",
        str(map_path),
    ]
    if deterministic:
        map_cmd.append("--deterministic")
    _run(map_cmd, cwd=REPO_ROOT)

    mar_cmd = [
        str(nextstat_bin),
        "m15",
        "mar",
        "--map",
        str(map_path),
        "--assessment-table",
        str(assessment_path),
        "--validation-report",
        str(DEFAULT_VALIDATION_REPORT),
        "--pharma-validation",
        str(DEFAULT_PHARMA_VALIDATION),
        "--output",
        str(mar_path),
    ]
    if deterministic:
        mar_cmd.append("--deterministic")
    _run(mar_cmd, cwd=REPO_ROOT)

    return {
        "assessment_table": assessment_path,
        "map": map_path,
        "mar": mar_path,
    }


def _assessment_run(
    nextstat_bin: Path, case_root: Path, deterministic: bool
) -> tuple[list[str], list[tuple[Path, str]]]:
    out_path = case_root / "m15_assessment_table.json"
    cmd = [
        str(nextstat_bin),
        "m15",
        "assessment-table",
        "--config",
        str(DEFAULT_CONFIG),
        "--validation-report",
        str(DEFAULT_VALIDATION_REPORT),
        "--pharma-validation",
        str(DEFAULT_PHARMA_VALIDATION),
        "--output",
        str(out_path),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return cmd, [_validation_plan(out_path, "m15_assessment_table_v1")]


def _map_run(
    nextstat_bin: Path, case_root: Path, deterministic: bool, setup: dict[str, Path]
) -> tuple[list[str], list[tuple[Path, str]]]:
    out_path = case_root / "m15_map.json"
    cmd = [
        str(nextstat_bin),
        "m15",
        "map",
        "--config",
        str(DEFAULT_CONFIG),
        "--assessment-table",
        str(setup["assessment_table"]),
        "--output",
        str(out_path),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return cmd, [_validation_plan(out_path, "m15_map_v1")]


def _mar_run(
    nextstat_bin: Path, case_root: Path, deterministic: bool, setup: dict[str, Path]
) -> tuple[list[str], list[tuple[Path, str]]]:
    out_path = case_root / "m15_mar.json"
    cmd = [
        str(nextstat_bin),
        "m15",
        "mar",
        "--map",
        str(setup["map"]),
        "--assessment-table",
        str(setup["assessment_table"]),
        "--validation-report",
        str(DEFAULT_VALIDATION_REPORT),
        "--pharma-validation",
        str(DEFAULT_PHARMA_VALIDATION),
        "--output",
        str(out_path),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return cmd, [_validation_plan(out_path, "m15_mar_v1")]


def _bundle_run(
    nextstat_bin: Path, case_root: Path, deterministic: bool, setup: dict[str, Path]
) -> tuple[list[str], list[tuple[Path, str]]]:
    out_path = case_root / "m15_bundle_manifest.json"
    cmd = [
        str(nextstat_bin),
        "m15",
        "bundle",
        "--config",
        str(DEFAULT_CONFIG),
        "--assessment-table",
        str(setup["assessment_table"]),
        "--map",
        str(setup["map"]),
        "--mar",
        str(setup["mar"]),
        "--validation-report",
        str(DEFAULT_VALIDATION_REPORT),
        "--pharma-validation",
        str(DEFAULT_PHARMA_VALIDATION),
        "--output",
        str(out_path),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return cmd, [_validation_plan(out_path, "m15_bundle_manifest_v1")]


def _validation_pack_run(
    nextstat_bin: Path, case_root: Path, deterministic: bool, *, include_m15: bool
) -> tuple[list[str], list[tuple[Path, str]]]:
    out_dir = case_root / "pack"
    out_dir.mkdir(parents=True, exist_ok=True)
    if include_m15:
        shutil.copyfile(DEFAULT_PHARMA_VALIDATION, out_dir / "pharma_validation.json")
    cmd = [
        "bash",
        str(VALIDATION_PACK_SCRIPT),
        "--out-dir",
        str(out_dir),
        "--workspace",
        str(DEFAULT_WORKSPACE),
        "--apex2-master",
        str(DEFAULT_APEX2_MASTER),
        "--python",
        sys.executable,
        "--nextstat-bin",
        str(nextstat_bin),
        "--json-only",
        "--skip-pharma-validation",
    ]
    if include_m15:
        cmd.extend(["--m15-config", str(DEFAULT_CONFIG)])
    if deterministic:
        cmd.append("--deterministic")

    validations = [_validation_plan(out_dir / "validation_pack_manifest.json", "validation_pack_manifest_v1")]
    if include_m15:
        validations.append(_validation_plan(out_dir / "m15_bundle_manifest.json", "m15_bundle_manifest_v1"))
    return cmd, validations


def _measure_case(
    *,
    case_id: str,
    description: str,
    work_root: Path,
    warmups: int,
    runs: int,
    invoke: Any,
) -> dict[str, Any]:
    case_root = work_root / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    samples: list[float] = []
    first_validations: list[dict[str, Any]] | None = None
    first_command: list[str] | None = None
    first_artifact_root: str | None = None

    for idx in range(warmups + runs):
        sample_root = case_root / f"run_{idx + 1:02d}"
        sample_root.mkdir(parents=True, exist_ok=True)
        cmd, validation_plan = invoke(sample_root)
        elapsed, _ = _run(cmd, cwd=REPO_ROOT)
        validations = [
            _validated_artifact(path, expected_schema_version, sample_root)
            for path, expected_schema_version in validation_plan
        ]
        _assert_validations(validations)
        if idx >= warmups:
            samples.append(elapsed)
            if first_validations is None:
                first_validations = validations
                first_command = cmd
                first_artifact_root = _relative_or_name(sample_root, work_root)

    assert first_validations is not None
    assert first_command is not None
    assert first_artifact_root is not None
    timing = _timing_doc(samples)
    return {
        "artifact_root": first_artifact_root,
        "case_id": case_id,
        "command": first_command,
        "description": description,
        "max_s": timing["max_s"],
        "median_s": timing["median_s"],
        "min_s": timing["min_s"],
        "runs": runs,
        "samples_s": timing["samples_s"],
        "status": "ok",
        "validation": {"validated_artifacts": first_validations},
        "warmups": warmups,
    }


def _build_markdown(report: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# M15 Reporting Benchmark Baseline",
        "",
        f"- Host: {report['host']['hostname']}",
        f"- CPU: {report['host']['cpu_model']}",
        f"- Binary: `{report['binary']['path']}`",
        f"- Binary sha256: `{report['binary']['sha256']}`",
        f"- Version: `{report['binary']['version']}`",
        f"- Build profile: `{report['binary']['build_profile']}`",
        f"- Warmups: `{report['protocol']['warmups']}`",
        f"- Runs: `{report['protocol']['runs']}`",
        "",
        "| Case | Min (s) | Median (s) | Max (s) |",
        "|---|---:|---:|---:|",
    ]
    for case in report["results"]:
        lines.append(
            f"| {case['case_id']} | {case['min_s']:.4f} | {case['median_s']:.4f} | {case['max_s']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Validation-pack M15/base median ratio: `{report['derived']['validation_pack_m15_over_base_median_ratio']:.4f}x`",
            f"- Raw JSON report: `{json_path}`",
        ]
    )
    binary_note = report["binary"].get("note")
    if binary_note:
        lines.extend(["", "Note:", binary_note])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nextstat-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.runs = 2
        args.warmups = 0
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.warmups < 0:
        raise SystemExit("--warmups must be >= 0")

    nextstat_bin = args.nextstat_bin.resolve()
    markdown_out = args.markdown_out or args.out.with_suffix(".md")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    version_proc = subprocess.run(
        [str(nextstat_bin), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if version_proc.returncode != 0:
        raise SystemExit(f"failed to query nextstat version from {nextstat_bin}: {version_proc.stdout}")

    setup = _build_setup_artifacts(nextstat_bin, args.work_root, args.deterministic)

    results: list[dict[str, Any]] = []
    results.append(
        _measure_case(
            case_id="m15_assessment_table",
            description="Build assessment-table from frozen config, validation report, and pharma evidence.",
            work_root=args.work_root,
            warmups=args.warmups,
            runs=args.runs,
            invoke=lambda case_root: _assessment_run(nextstat_bin, case_root, args.deterministic),
        )
    )
    results.append(
        _measure_case(
            case_id="m15_map",
            description="Build MAP from frozen config and prebuilt assessment-table.",
            work_root=args.work_root,
            warmups=args.warmups,
            runs=args.runs,
            invoke=lambda case_root: _map_run(nextstat_bin, case_root, args.deterministic, setup),
        )
    )
    results.append(
        _measure_case(
            case_id="m15_mar",
            description="Build MAR from prebuilt MAP and assessment-table plus frozen evidence.",
            work_root=args.work_root,
            warmups=args.warmups,
            runs=args.runs,
            invoke=lambda case_root: _mar_run(nextstat_bin, case_root, args.deterministic, setup),
        )
    )
    results.append(
        _measure_case(
            case_id="m15_bundle",
            description="Build M15 bundle manifest from prebuilt artifact chain.",
            work_root=args.work_root,
            warmups=args.warmups,
            runs=args.runs,
            invoke=lambda case_root: _bundle_run(nextstat_bin, case_root, args.deterministic, setup),
        )
    )
    results.append(
        _measure_case(
            case_id="validation_pack_base_json_only",
            description="Render JSON-only validation pack without M15 artifacts.",
            work_root=args.work_root,
            warmups=args.warmups,
            runs=args.runs,
            invoke=lambda case_root: _validation_pack_run(
                nextstat_bin, case_root, args.deterministic, include_m15=False
            ),
        )
    )
    results.append(
        _measure_case(
            case_id="validation_pack_m15_json_only",
            description="Render JSON-only validation pack with M15 artifact chain using preseeded pharma evidence.",
            work_root=args.work_root,
            warmups=args.warmups,
            runs=args.runs,
            invoke=lambda case_root: _validation_pack_run(
                nextstat_bin, case_root, args.deterministic, include_m15=True
            ),
        )
    )

    by_case = {case["case_id"]: case for case in results}
    ratio = by_case["validation_pack_m15_json_only"]["median_s"] / by_case["validation_pack_base_json_only"]["median_s"]

    report = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "deterministic": bool(args.deterministic),
        "ok": all(case["status"] == "ok" for case in results),
        "meta": {
            "git_commit": _git_commit(),
            "host_policy": HOST_POLICY,
            "markdown_out": str(markdown_out),
            "nextstat_bin": str(nextstat_bin),
            "nextstat_command": ["python3", "scripts/benchmarks/bench_m15_reporting.py"],
            "out": str(args.out),
            "smoke": bool(args.smoke),
            "work_root": str(args.work_root),
        },
        "host": {
            "cpu_model": _cpu_model(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "binary": {
            "build_profile": _build_profile(nextstat_bin),
            "path": str(nextstat_bin),
            "sha256": _sha256_file(nextstat_bin),
            "version": version_proc.stdout.strip(),
        },
        "protocol": {
            "cases": CASE_ORDER,
            "metric": "wall_time_seconds",
            "runs": args.runs,
            "summary_policy": ["min", "median", "max"],
            "warmups": args.warmups,
            "workspace_root": str(args.work_root),
        },
        "results": results,
        "derived": {
            "validation_pack_m15_over_base_median_ratio": round(ratio, 6),
        },
    }

    _write_json(args.out, report)
    markdown_out.write_text(_build_markdown(report, args.out), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
