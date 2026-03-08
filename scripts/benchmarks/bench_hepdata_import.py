#!/usr/bin/env python3
"""Deterministic runtime gate for `nextstat import hepdata`.

This script benchmarks the supported public HEPData import modes on local,
checked-in fixtures so the gate can run without external network access.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BENCH_SCRIPTS_ROOT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(BENCH_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPTS_ROOT))

from bench_env import collect_environment  # type: ignore
from hepdata_example_helpers import (  # type: ignore
    CANONICAL_BKGONLY_FILENAME,
    CANONICAL_DATASET_ID,
    CANONICAL_DISPLAY_NAME,
    CANONICAL_DOI,
    CANONICAL_PATCHSET_FILENAME,
    fixture_archive,
    preferred_benchmark_patch_name,
    seed_cached_download,
    static_archive_server,
)


CURATED_DATASET_ID = "hepdata.90607.v3.r3"
DEFAULT_OUT = REPO_ROOT / "bench_results" / "hepdata_import_benchmark" / "summary.json"
DEFAULT_WORK_ROOT = DEFAULT_OUT.parent / "work"


def _cargo_target_dir() -> str:
    override = os.environ.get("NEXTSTAT_HEPDATA_BENCH_CARGO_TARGET_DIR")
    if override:
        return override
    return str(REPO_ROOT.parent / ".nextstat-cargo-target" / "hepdata-import-bench")


def _derive_cli_from_hepdata_override(override: str) -> list[str]:
    parts = shlex.split(override)
    if parts[-2:] == ["import", "hepdata"]:
        return parts[:-2]
    return parts


def _ensure_nextstat_cli_argv() -> list[str]:
    cli_override = os.environ.get("NEXTSTAT_CLI_CMD")
    if cli_override:
        return shlex.split(cli_override)

    hepdata_override = os.environ.get("NEXTSTAT_HEPDATA_CMD")
    if hepdata_override:
        return _derive_cli_from_hepdata_override(hepdata_override)

    bin_path = Path(_cargo_target_dir()) / "debug" / "nextstat"
    if not bin_path.exists():
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = _cargo_target_dir()
        subprocess.run(
            ["cargo", "build", "-q", "-p", "ns-cli"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
    return [str(bin_path)]


def _run_json(argv: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return json.loads(completed.stdout)


def _timing_doc(per_run_s: list[float]) -> dict[str, Any]:
    if not per_run_s:
        raise RuntimeError("expected at least one timing sample")
    return {
        "repeat": len(per_run_s),
        "policy": "min",
        "per_run_s": [round(value, 6) for value in per_run_s],
        "best_s": round(min(per_run_s), 6),
    }


def _summary_stage_timings(dataset: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(dataset, dict):
        return None
    timings = dataset.get("timings")
    if not isinstance(timings, dict):
        return None
    return {key: float(value) for key, value in timings.items()}


def _merge_summary_timings(
    primary: dict[str, float] | None,
    extra: dict[str, float] | None,
) -> dict[str, float] | None:
    if primary is None and extra is None:
        return None
    merged: dict[str, float] = {}
    for timings in (primary, extra):
        if timings is None:
            continue
        for key, value in timings.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def _benchmark_stages(
    *,
    summary_mode: str,
    command_best_s: float,
    dataset_timings: dict[str, float] | None,
    fit_best_s: float | None,
) -> dict[str, Any]:
    if dataset_timings is None:
        discovery_s = command_best_s if summary_mode == "catalog" else 0.0
        download_s = 0.0
        extract_s = 0.0
        materialize_s = 0.0
    else:
        discovery_s = float(dataset_timings.get("inspect_inputs_s", 0.0))
        download_s = float(dataset_timings.get("download_s", 0.0))
        extract_s = float(dataset_timings.get("extract_archive_s", 0.0)) + float(
            dataset_timings.get("extract_nested_archives_s", 0.0)
        )
        materialize_s = float(dataset_timings.get("materialize_total_s", 0.0))
    return {
        "discovery_s": round(discovery_s, 6),
        "download_s": round(download_s, 6),
        "extract_s": round(extract_s, 6),
        "materialize_s": round(materialize_s, 6),
        "fit_s": None if fit_best_s is None else round(fit_best_s, 6),
        "import_total_s": round(command_best_s, 6),
    }


def _relative_to(base: Path, target: Path) -> str:
    return str(target.resolve().relative_to(base.resolve()))


def _normalize_benchmark_dataset_doi(value: str) -> str:
    if not value:
        return value
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost"} and parsed.path == "/download":
        return CANONICAL_DOI
    return value


def _fit_workspace_path(materialized: list[dict[str, Any]]) -> Path | None:
    for item in materialized:
        if item.get("kind") == "patched":
            return Path(item["path"])
    for item in materialized:
        if item.get("kind") == "bkgonly":
            return Path(item["path"])
    return None


def _run_fit(cli_argv: list[str], workspace_path: Path) -> tuple[dict[str, Any], list[str]]:
    fit_argv = cli_argv + ["fit", "--input", str(workspace_path), "--threads", "1"]
    fit_doc = _run_json(fit_argv)
    converged = bool(fit_doc.get("converged"))
    if not converged:
        raise RuntimeError(f"fit did not converge for {workspace_path}")
    return (
        {
            "status": "ok",
            "workspace_kind": "patched" if "patched__" in workspace_path.name else "bkgonly",
            "timing": None,  # filled by caller
            "command": fit_argv,
            "result": {
                "converged": converged,
                "bestfit_len": len(fit_doc.get("bestfit", [])),
            },
        },
        fit_argv,
    )


def _validate_catalog_case(
    summary: dict[str, Any],
    *,
    expected_source_mode: str,
    expected_dataset_id: str | None = None,
    expected_download_mode: str | None = None,
) -> dict[str, Any]:
    if summary.get("schema_version") != "nextstat.hepdata_import.v1":
        raise RuntimeError("unexpected import schema version")
    if summary.get("mode") != "catalog":
        raise RuntimeError("expected catalog mode")
    if summary.get("source_mode") != expected_source_mode:
        raise RuntimeError("unexpected source_mode")
    datasets = summary.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise RuntimeError("expected non-empty datasets")

    selected = None
    if expected_dataset_id is None:
        selected = datasets[0]
    else:
        for candidate in datasets:
            if candidate.get("id") == expected_dataset_id:
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(f"dataset {expected_dataset_id!r} not found in catalog output")

    download = selected.get("download")
    if expected_download_mode is not None:
        if not isinstance(download, dict) or download.get("mode") != expected_download_mode:
            raise RuntimeError(f"expected download mode {expected_download_mode!r}")

    inputs = selected.get("inputs") if isinstance(selected.get("inputs"), dict) else {}
    available_patch_names = inputs.get("available_patch_names", [])
    return {
        "dataset": {
            "id": str(selected.get("id", "")),
            "doi": _normalize_benchmark_dataset_doi(str(selected.get("doi", ""))),
            "source_mode": expected_source_mode,
        },
        "validation": {
            "schema_version": str(summary["schema_version"]),
            "dataset_count": len(datasets),
            "workspace_count": 0,
            "download_mode": None if download is None else str(download.get("mode")),
            "lockfile_written": False,
            "available_patch_names": [str(name) for name in available_patch_names],
        },
        "artifacts": {
            "lockfile": None,
            "workspaces": [],
        },
        "summary_timings": _summary_stage_timings(selected),
    }


def _validate_materialize_case(
    summary: dict[str, Any],
    *,
    lock_path: Path,
    out_dir: Path,
    expected_source_mode: str,
    expected_download_mode: str,
) -> tuple[dict[str, Any], Path | None]:
    if summary.get("schema_version") != "nextstat.hepdata_import.v1":
        raise RuntimeError("unexpected import schema version")
    if summary.get("mode") != "materialize":
        raise RuntimeError("expected materialize mode")
    if summary.get("source_mode") != expected_source_mode:
        raise RuntimeError("unexpected source_mode")
    datasets = summary.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise RuntimeError("expected exactly one materialized dataset")

    dataset = datasets[0]
    materialized = dataset.get("materialized")
    if not isinstance(materialized, list) or not materialized:
        raise RuntimeError("expected non-empty materialized list")

    download = dataset.get("download")
    if not isinstance(download, dict) or download.get("mode") != expected_download_mode:
        raise RuntimeError(f"expected download mode {expected_download_mode!r}")

    if not lock_path.exists():
        raise RuntimeError(f"expected lockfile at {lock_path}")
    lock_doc = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock_doc.get("schema_version") != "nextstat.hepdata_lock.v1":
        raise RuntimeError("unexpected lock schema version")

    workspace_paths: list[str] = []
    for item in materialized:
        path = Path(item["path"])
        if not path.exists():
            raise RuntimeError(f"expected materialized workspace at {path}")
        workspace_paths.append(_relative_to(out_dir, path))

    inputs = dataset.get("inputs") if isinstance(dataset.get("inputs"), dict) else {}
    available_patch_names = [str(name) for name in inputs.get("available_patch_names", [])]

    fit_workspace = _fit_workspace_path(materialized)
    return (
        {
            "dataset": {
                "id": str(dataset.get("id", "")),
                "doi": _normalize_benchmark_dataset_doi(str(dataset.get("doi", ""))),
                "source_mode": expected_source_mode,
            },
            "validation": {
                "schema_version": str(summary["schema_version"]),
                "dataset_count": len(datasets),
                "workspace_count": len(materialized),
                "download_mode": str(download.get("mode")),
                "lockfile_written": True,
                "available_patch_names": available_patch_names,
            },
            "artifacts": {
                "lockfile": _relative_to(lock_path.parent, lock_path),
                "workspaces": workspace_paths,
            },
            "summary_timings": _summary_stage_timings(dataset),
        },
        fit_workspace,
    )


def _benchmark_catalog_case(
    *,
    case_id: str,
    repeat: int,
    command_factory,
    validator,
) -> dict[str, Any]:
    per_run_s: list[float] = []
    best_s: float | None = None
    best_summary: dict[str, Any] | None = None
    best_command: list[str] | None = None
    best_validation: dict[str, Any] | None = None

    for _ in range(repeat):
        argv = command_factory()
        t0 = time.perf_counter()
        summary = _run_json(argv)
        elapsed = time.perf_counter() - t0
        per_run_s.append(elapsed)
        validation = validator(summary)
        if best_s is None or elapsed < best_s:
            best_s = elapsed
            best_summary = summary
            best_command = argv
            best_validation = validation

    if best_summary is None or best_command is None or best_validation is None or best_s is None:
        raise RuntimeError(f"benchmark case {case_id} produced no results")

    summary_timings = best_validation.pop("summary_timings", None)

    return {
        "id": case_id,
        "status": "ok",
        "command": best_command,
        "summary_mode": str(best_summary["mode"]),
        "source_mode": str(best_summary["source_mode"]),
        "timing": _timing_doc(per_run_s),
        "stages": _benchmark_stages(
            summary_mode=str(best_summary["mode"]),
            command_best_s=best_s,
            dataset_timings=summary_timings,
            fit_best_s=None,
        ),
        **best_validation,
    }


def _benchmark_materialize_case(
    *,
    case_id: str,
    repeat: int,
    fit_repeat: int,
    skip_fit: bool,
    work_root: Path,
    command_factory,
    validator,
) -> dict[str, Any]:
    import_times: list[float] = []
    best_case: dict[str, Any] | None = None
    best_import_s: float | None = None
    best_preflight_timings: dict[str, float] | None = None

    for run_index in range(1, repeat + 1):
        run_root = work_root / case_id / f"run_{run_index}"
        cache_dir = run_root / "cache"
        out_dir = run_root / "out"
        lock_path = run_root / "workspaces.lock.json"
        run_root.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        argv, summary, preflight_timings = command_factory(
            cache_dir=cache_dir,
            out_dir=out_dir,
            lock_path=lock_path,
        )
        elapsed = time.perf_counter() - t0
        import_times.append(elapsed)

        case_payload, fit_workspace = validator(summary, lock_path=lock_path, out_dir=out_dir)
        current_case = {
            "id": case_id,
            "status": "ok",
            "command": argv,
            "summary_mode": str(summary["mode"]),
            "source_mode": str(summary["source_mode"]),
            "timing": None,  # filled below
            **case_payload,
        }

        if not skip_fit and fit_workspace is not None:
            fit_samples: list[float] = []
            fit_payload: dict[str, Any] | None = None
            for _ in range(fit_repeat):
                t_fit = time.perf_counter()
                fit_payload, _ = _run_fit(_ensure_nextstat_cli_argv(), fit_workspace)
                fit_samples.append(time.perf_counter() - t_fit)
            if fit_payload is None:
                raise RuntimeError("expected fit payload")
            fit_payload["timing"] = _timing_doc(fit_samples)
            current_case["fit"] = fit_payload

        if best_import_s is None or elapsed < best_import_s:
            best_import_s = elapsed
            best_case = current_case
            best_preflight_timings = preflight_timings

    if best_case is None or best_import_s is None:
        raise RuntimeError(f"benchmark case {case_id} produced no results")

    fit_best_s = None
    if "fit" in best_case:
        best_case["fit"]["timing"]["repeat"] = len(best_case["fit"]["timing"]["per_run_s"])
        fit_best_s = float(best_case["fit"]["timing"]["best_s"])
    summary_timings = _merge_summary_timings(
        best_case.pop("summary_timings", None),
        best_preflight_timings,
    )
    best_case["timing"] = _timing_doc(import_times)
    best_case["stages"] = _benchmark_stages(
        summary_mode=str(best_case["summary_mode"]),
        command_best_s=best_import_s,
        dataset_timings=summary_timings,
        fit_best_s=fit_best_s,
    )
    return best_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--fit-repeat", type=int, default=1)
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="use minimal repeat counts for CI/local smoke")
    parser.add_argument("--deterministic", action="store_true", help="record deterministic-mode intent in the artifact")
    args = parser.parse_args(argv)

    repeat = 1 if args.smoke else max(1, args.repeat)
    fit_repeat = 1 if args.smoke else max(1, args.fit_repeat)

    cli_argv = _ensure_nextstat_cli_argv()
    import_prefix = cli_argv + ["import", "hepdata"]
    archive_bytes = fixture_archive().read_bytes()

    def curated_catalog_command() -> list[str]:
        return import_prefix + ["--list"]

    def direct_patch_catalog_command() -> list[str]:
        run_root = args.work_root / "direct_patch_catalog_cached" / "seed"
        cache_dir = run_root / "cache"
        seed_cached_download(cache_dir, CANONICAL_DATASET_ID)
        return import_prefix + [
            "--list-patches",
            "--doi",
            CANONICAL_DOI,
            "--dataset-id",
            CANONICAL_DATASET_ID,
            "--bkgonly-filename",
            CANONICAL_BKGONLY_FILENAME,
            "--patchset-filename",
            CANONICAL_PATCHSET_FILENAME,
            "--cache-dir",
            str(cache_dir),
            "--offline",
        ]

    def curated_materialize_command(
        *,
        cache_dir: Path,
        out_dir: Path,
        lock_path: Path,
    ) -> tuple[list[str], dict[str, Any], dict[str, float] | None]:
        seed_cached_download(cache_dir, CURATED_DATASET_ID)
        argv = import_prefix + [
            "--dataset",
            CURATED_DATASET_ID,
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--lock",
            str(lock_path),
            "--offline",
        ]
        return argv, _run_json(argv), None

    def direct_materialize_command(
        *,
        cache_dir: Path,
        out_dir: Path,
        lock_path: Path,
    ) -> tuple[list[str], dict[str, Any], dict[str, float] | None]:
        discovery_cache = cache_dir.parent / "discovery-cache"
        seed_cached_download(discovery_cache, CANONICAL_DATASET_ID)
        catalog = _run_json(
            import_prefix
            + [
                "--list-patches",
                "--doi",
                CANONICAL_DOI,
                "--dataset-id",
                CANONICAL_DATASET_ID,
                "--bkgonly-filename",
                CANONICAL_BKGONLY_FILENAME,
                "--patchset-filename",
                CANONICAL_PATCHSET_FILENAME,
                "--cache-dir",
                str(discovery_cache),
                "--offline",
            ]
        )
        preflight_timings = _summary_stage_timings(catalog["datasets"][0])
        available_patch_names = catalog["datasets"][0]["inputs"]["available_patch_names"]
        patch_name = preferred_benchmark_patch_name(available_patch_names)
        with static_archive_server(archive_bytes) as doi_url:
            argv = import_prefix + [
                "--doi",
                doi_url,
                "--dataset-id",
                CANONICAL_DATASET_ID,
                "--display-name",
                CANONICAL_DISPLAY_NAME,
                "--bkgonly-filename",
                CANONICAL_BKGONLY_FILENAME,
                "--patchset-filename",
                CANONICAL_PATCHSET_FILENAME,
                "--patch",
                f"benchmark={patch_name}",
                "--cache-dir",
                str(cache_dir),
                "--out-dir",
                str(out_dir),
                "--lock",
                str(lock_path),
            ]
            return argv, _run_json(argv), preflight_timings

    cases: list[dict[str, Any]] = []
    exit_code = 0

    case_builders = [
        lambda: _benchmark_catalog_case(
            case_id="curated_catalog",
            repeat=repeat,
            command_factory=curated_catalog_command,
            validator=lambda summary: _validate_catalog_case(
                summary,
                expected_source_mode="curated",
                expected_dataset_id=CURATED_DATASET_ID,
            ),
        ),
        lambda: _benchmark_catalog_case(
            case_id="direct_patch_catalog_cached",
            repeat=repeat,
            command_factory=direct_patch_catalog_command,
            validator=lambda summary: _validate_catalog_case(
                summary,
                expected_source_mode="direct_doi",
                expected_dataset_id=CANONICAL_DATASET_ID,
                expected_download_mode="cached",
            ),
        ),
        lambda: _benchmark_materialize_case(
            case_id="curated_materialize_offline",
            repeat=repeat,
            fit_repeat=fit_repeat,
            skip_fit=args.skip_fit,
            work_root=args.work_root,
            command_factory=curated_materialize_command,
            validator=lambda summary, lock_path, out_dir: _validate_materialize_case(
                summary,
                lock_path=lock_path,
                out_dir=out_dir,
                expected_source_mode="curated",
                expected_download_mode="cached",
            ),
        ),
        lambda: _benchmark_materialize_case(
            case_id="direct_materialize_network",
            repeat=repeat,
            fit_repeat=fit_repeat,
            skip_fit=args.skip_fit,
            work_root=args.work_root,
            command_factory=direct_materialize_command,
            validator=lambda summary, lock_path, out_dir: _validate_materialize_case(
                summary,
                lock_path=lock_path,
                out_dir=out_dir,
                expected_source_mode="direct_doi",
                expected_download_mode="network",
            ),
        ),
    ]

    for build_case in case_builders:
        try:
            case_doc = build_case()
        except Exception as exc:
            case_doc = {
                "id": f"failed_case_{len(cases) + 1}",
                "status": "failed",
                "command": [],
                "summary_mode": "catalog",
                "source_mode": "curated",
                "timing": {"repeat": 0, "policy": "min", "per_run_s": [], "best_s": 0.0},
                "stages": {
                    "discovery_s": 0.0,
                    "download_s": 0.0,
                    "extract_s": 0.0,
                    "materialize_s": 0.0,
                    "fit_s": None,
                    "import_total_s": 0.0,
                },
                "dataset": {"id": "", "doi": "", "source_mode": "curated"},
                "validation": {
                    "schema_version": "",
                    "dataset_count": 0,
                    "workspace_count": 0,
                    "download_mode": None,
                    "lockfile_written": False,
                    "available_patch_names": [],
                },
                "artifacts": {"lockfile": None, "workspaces": []},
                "error": f"{type(exc).__name__}: {exc}",
            }
            exit_code = 1
        cases.append(case_doc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "nextstat.hepdata_import_benchmark_result.v1",
        "suite": "hepdata_import",
        "deterministic": bool(args.deterministic or args.smoke),
        "environment": collect_environment(),
        "meta": {
            "host_policy": "nextstat-bench",
            "nextstat_command": cli_argv,
            "repeat": repeat,
            "fit_repeat": fit_repeat,
            "smoke": bool(args.smoke),
            "fit_enabled": not args.skip_fit,
            "out": str(args.out),
            "work_root": str(args.work_root),
        },
        "cases": cases,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
