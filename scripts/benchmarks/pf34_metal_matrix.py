#!/usr/bin/env python3
"""PF3.4 Metal benchmark matrix runner (local host).

Runs reproducible `nextstat` benchmark commands for Metal on Apple Silicon and
writes per-run artifacts (`.meta.json`, `.metrics.json`, `.out.json`, `.err`) plus
aggregate summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(msg)


def split_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def split_csv(value: str) -> list[str]:
    out: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


VALID_MODES = {
    "fit_host",
    "fit_device",
    "hypotest_host",
    "hypotest_device",
}


@dataclass
class RunSpec:
    case_id: str
    mode: str
    n_toys: int
    spec_path: Path
    mu_test: float
    seed: int
    expected_set: bool


@dataclass
class RunResult:
    name: str
    rc: int
    elapsed_s: float


def find_nextstat(bin_override: str | None) -> str:
    if bin_override:
        p = Path(bin_override)
        if not p.is_file():
            die(f"nextstat binary not found: {p}")
        return str(p)

    root = Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        p = root / "target" / profile / "nextstat"
        if p.is_file():
            return str(p)

    found = shutil_which("nextstat")
    if found:
        return found
    die("nextstat CLI not found. Build with: cargo build -p ns-cli --release --features metal")


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def build_command(
    bin_path: str,
    rs: RunSpec,
    metrics_path: Path,
    log_level: str,
    threads: int,
) -> list[str]:
    if rs.mode == "fit_host":
        return [
            bin_path,
            "unbinned-fit-toys",
            "--config",
            str(rs.spec_path),
            "--n-toys",
            str(rs.n_toys),
            "--seed",
            str(rs.seed),
            "--threads",
            str(threads),
            "--gpu",
            "metal",
            "--log-level",
            log_level,
            "--json-metrics",
            str(metrics_path),
        ]
    if rs.mode == "fit_device":
        return [
            bin_path,
            "unbinned-fit-toys",
            "--config",
            str(rs.spec_path),
            "--n-toys",
            str(rs.n_toys),
            "--seed",
            str(rs.seed),
            "--threads",
            str(threads),
            "--gpu",
            "metal",
            "--gpu-sample-toys",
            "--log-level",
            log_level,
            "--json-metrics",
            str(metrics_path),
        ]
    if rs.mode == "hypotest_host":
        cmd = [
            bin_path,
            "unbinned-hypotest-toys",
            "--config",
            str(rs.spec_path),
            "--mu",
            str(rs.mu_test),
            "--n-toys",
            str(rs.n_toys),
            "--seed",
            str(rs.seed),
            "--threads",
            str(threads),
            "--gpu",
            "metal",
            "--log-level",
            log_level,
            "--json-metrics",
            str(metrics_path),
        ]
        if rs.expected_set:
            cmd.append("--expected-set")
        return cmd
    if rs.mode == "hypotest_device":
        cmd = [
            bin_path,
            "unbinned-hypotest-toys",
            "--config",
            str(rs.spec_path),
            "--mu",
            str(rs.mu_test),
            "--n-toys",
            str(rs.n_toys),
            "--seed",
            str(rs.seed),
            "--threads",
            str(threads),
            "--gpu",
            "metal",
            "--gpu-sample-toys",
            "--log-level",
            log_level,
            "--json-metrics",
            str(metrics_path),
        ]
        if rs.expected_set:
            cmd.append("--expected-set")
        return cmd
    die(f"unsupported mode: {rs.mode}")


def preflight_metal_runtime(
    bin_path: str,
    probe_spec: Path,
    threads: int,
    log_level: str,
    timeout_s: int,
) -> tuple[bool, str]:
    """Run a tiny probe command to verify Metal runtime availability."""
    cmd = [
        bin_path,
        "unbinned-fit-toys",
        "--config",
        str(probe_spec),
        "--n-toys",
        "1",
        "--seed",
        "1",
        "--threads",
        str(threads),
        "--gpu",
        "metal",
        "--log-level",
        log_level,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return False, f"preflight timeout after {timeout_s}s"

    if proc.returncode == 0:
        return True, "ok"

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    msg = stderr if stderr else stdout
    if "Metal is not available at runtime" in msg or "no Metal device found" in msg:
        return False, msg
    return False, msg or f"probe failed with rc={proc.returncode}"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                return None
        return None


def run_one(
    rs: RunSpec,
    bin_path: str,
    out_dir: Path,
    timeout_s: int,
    log_level: str,
    threads: int,
    dry_run: bool,
) -> RunResult:
    name = f"{rs.case_id}_{rs.mode}_t{rs.n_toys}"
    meta_path = out_dir / f"{name}.meta.json"
    metrics_path = out_dir / f"{name}.metrics.json"
    out_path = out_dir / f"{name}.out.json"
    err_path = out_dir / f"{name}.err"

    cmd = build_command(bin_path, rs, metrics_path, log_level, threads)
    cmd_str = " ".join(shlex.quote(x) for x in cmd)

    started = time.time()
    rc = 0
    if dry_run:
        out_path.write_text(json.dumps({"dry_run": True, "command": cmd}, indent=2), encoding="utf-8")
    else:
        with out_path.open("w", encoding="utf-8") as f_out, err_path.open("w", encoding="utf-8") as f_err:
            try:
                proc = subprocess.run(cmd, stdout=f_out, stderr=f_err, timeout=timeout_s, check=False)
                rc = int(proc.returncode)
            except subprocess.TimeoutExpired:
                rc = 124
                f_err.write(f"timeout after {timeout_s}s\n")

    elapsed_s = time.time() - started

    meta = {
        "schema_version": "nextstat.pf34_metal_run_meta.v1",
        "name": name,
        "case_id": rs.case_id,
        "mode": rs.mode,
        "n_toys": rs.n_toys,
        "mu_test": rs.mu_test,
        "seed": rs.seed,
        "expected_set": rs.expected_set,
        "spec": str(rs.spec_path),
        "rc": rc,
        "elapsed_s": elapsed_s,
        "timeout_s": timeout_s,
        "threads": threads,
        "log_level": log_level,
        "ts_unix": int(started),
        "command": cmd,
        "command_str": cmd_str,
        "dry_run": dry_run,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return RunResult(name=name, rc=rc, elapsed_s=elapsed_s)


def summarize(out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(out_dir.glob("*.meta.json")):
        meta = load_json(meta_path) or {}
        stem = meta_path.name[: -len(".meta.json")]
        metrics = load_json(out_dir / f"{stem}.metrics.json")
        out = load_json(out_dir / f"{stem}.out.json")

        timing = (metrics or {}).get("timing") or {}
        toys_timing = ((timing.get("breakdown") or {}).get("toys") or {})

        row: dict[str, Any] = {
            "name": meta.get("name", stem),
            "case_id": meta.get("case_id"),
            "mode": meta.get("mode"),
            "n_toys": meta.get("n_toys"),
            "rc": meta.get("rc"),
            "elapsed_s": meta.get("elapsed_s"),
            "wall_time_s": timing.get("wall_time_s"),
            "pipeline": toys_timing.get("pipeline"),
            "sample_s": toys_timing.get("sample_s"),
            "batch_build_s": toys_timing.get("batch_build_s"),
            "batch_fit_s": toys_timing.get("batch_fit_s"),
            "poi_sigma_s": toys_timing.get("poi_sigma_s"),
            "sampler_init_s": toys_timing.get("sampler_init_s"),
            "sample_phase_detail": toys_timing.get("sample_phase_detail"),
        }

        if isinstance(out, dict):
            if "results" in out:
                res = out.get("results") or {}
                summ = out.get("summary") or {}
                n_toys = int(meta.get("n_toys") or 0)
                n_conv = summ.get("n_converged")
                if isinstance(n_conv, int) and n_toys > 0:
                    row["convergence_rate"] = n_conv / n_toys
                row["n_converged"] = n_conv
                row["n_error"] = res.get("n_error")
                row["n_validation_error"] = res.get("n_validation_error")
                row["n_computation_error"] = res.get("n_computation_error")
                row["n_nonconverged"] = res.get("n_nonconverged")
            else:
                row["cls"] = out.get("cls")
                row["clsb"] = out.get("clsb")
                row["clb"] = out.get("clb")
                nerr = out.get("n_error")
                if isinstance(nerr, dict):
                    row["n_error_b"] = nerr.get("b")
                    row["n_error_sb"] = nerr.get("sb")
                nnon = out.get("n_nonconverged")
                if isinstance(nnon, dict):
                    row["n_nonconverged_b"] = nnon.get("b")
                    row["n_nonconverged_sb"] = nnon.get("sb")

        rows.append(row)

    overall = {
        "n_runs": len(rows),
        "n_rc_fail": sum(1 for r in rows if int(r.get("rc") or 0) != 0),
    }
    return {
        "schema_version": "nextstat.pf34_metal_summary.v1",
        "generated_at": utc_now_iso(),
        "run_root": str(out_dir),
        "overall": overall,
        "rows": rows,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# PF3.4 Metal Benchmark Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Run root: `{summary['run_root']}`",
        "",
        f"- Runs: `{summary['overall']['n_runs']}`",
        f"- RC failures: `{summary['overall']['n_rc_fail']}`",
        "",
        "| run | mode | rc | pipeline | toys | wall_s | sample_s | build_s | fit_s | conv |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["rows"]:
        conv = "-"
        if isinstance(r.get("convergence_rate"), (int, float)):
            conv = f"{float(r['convergence_rate']):.4f}"
        lines.append(
            "| {name} | {mode} | {rc} | {pipeline} | {toys} | {wall} | {sample} | {build} | {fit} | {conv} |".format(
                name=r.get("name", "-"),
                mode=r.get("mode", "-"),
                rc=r.get("rc", "-"),
                pipeline=r.get("pipeline") or "-",
                toys=r.get("n_toys", "-"),
                wall=f"{float(r['wall_time_s']):.3f}" if isinstance(r.get("wall_time_s"), (int, float)) else "-",
                sample=f"{float(r['sample_s']):.3f}" if isinstance(r.get("sample_s"), (int, float)) else "-",
                build=f"{float(r['batch_build_s']):.3f}" if isinstance(r.get("batch_build_s"), (int, float)) else "-",
                fit=f"{float(r['batch_fit_s']):.3f}" if isinstance(r.get("batch_fit_s"), (int, float)) else "-",
                conv=conv,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plan_runs(matrix: dict[str, Any], root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for case in matrix.get("cases") or []:
        case_id = str(case["id"])
        spec_rel = str(case["spec_rel"])
        spec_path = (root / spec_rel).resolve()
        if not spec_path.exists():
            die(f"spec not found for case '{case_id}': {spec_path}")

        toys = split_csv_ints(str(case.get("toys", "")))
        if not toys:
            die(f"case '{case_id}' has empty toys")
        modes = split_csv(str(case.get("modes", "fit_host,fit_device")))
        for mode in modes:
            if mode not in VALID_MODES:
                die(f"case '{case_id}' has invalid mode '{mode}'")

        mu_test = float(case.get("mu", 1.0))
        seed = int(case.get("seed", 42))
        expected_set = bool(int(case.get("expected_set", 0)))

        for t in toys:
            for mode in modes:
                runs.append(
                    RunSpec(
                        case_id=case_id,
                        mode=mode,
                        n_toys=t,
                        spec_path=spec_path,
                        mu_test=mu_test,
                        seed=seed,
                        expected_set=expected_set,
                    )
                )
    return runs


def main() -> int:
    root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(description="Run PF3.4 Metal benchmark matrix locally")
    ap.add_argument(
        "--matrix",
        default=str(root / "benchmarks" / "unbinned" / "matrices" / "pf34_metal_v1.json"),
    )
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--bin", default=os.environ.get("PF34_BIN", ""))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("PF34_THREADS", "1")))
    ap.add_argument("--log-level", default=os.environ.get("PF34_LOG_LEVEL", "warn"))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("PF34_TIMEOUT", "21600")))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    matrix_path = Path(args.matrix).resolve()
    if not matrix_path.exists():
        die(f"matrix not found: {matrix_path}")

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != "nextstat.pf34_metal_matrix.v1":
        die("unexpected matrix schema_version (expected nextstat.pf34_metal_matrix.v1)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else root / "benchmarks" / "unbinned" / "artifacts" / date_tag / f"pf34_metal_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_path = find_nextstat(args.bin if args.bin else None)
    runs = plan_runs(matrix, root)

    run_meta = {
        "schema_version": "nextstat.pf34_metal_run_manifest.v1",
        "generated_at": utc_now_iso(),
        "matrix": str(matrix_path),
        "bin": bin_path,
        "threads": args.threads,
        "log_level": args.log_level,
        "timeout": args.timeout,
        "dry_run": bool(args.dry_run),
        "n_runs": len(runs),
        "runs": [
            {
                "name": f"{r.case_id}_{r.mode}_t{r.n_toys}",
                "case_id": r.case_id,
                "mode": r.mode,
                "n_toys": r.n_toys,
                "mu": r.mu_test,
                "seed": r.seed,
                "expected_set": r.expected_set,
                "spec": str(r.spec_path),
            }
            for r in runs
        ],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[pf34-metal] out_dir={out_dir}")
    print(f"[pf34-metal] bin={bin_path}")
    print(f"[pf34-metal] runs={len(runs)}")

    if not args.dry_run and not args.skip_preflight:
        probe_spec = runs[0].spec_path
        ok, reason = preflight_metal_runtime(
            bin_path=bin_path,
            probe_spec=probe_spec,
            threads=args.threads,
            log_level=args.log_level,
            timeout_s=min(args.timeout, 180),
        )
        preflight = {
            "schema_version": "nextstat.pf34_metal_preflight.v1",
            "generated_at": utc_now_iso(),
            "ok": ok,
            "reason": reason,
            "probe_spec": str(probe_spec),
            "bin": bin_path,
        }
        (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        print(f"[pf34-metal] preflight ok={ok} reason={reason}")
        if not ok:
            return 3

    rc = 0
    for rs in runs:
        rr = run_one(
            rs=rs,
            bin_path=bin_path,
            out_dir=out_dir,
            timeout_s=args.timeout,
            log_level=args.log_level,
            threads=args.threads,
            dry_run=bool(args.dry_run),
        )
        print(f"[pf34-metal] {rr.name}: rc={rr.rc} elapsed_s={rr.elapsed_s:.3f}")
        if rr.rc != 0:
            rc = 2

    summary = summarize(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "summary.md")
    print(f"[pf34-metal] summary={out_dir / 'summary.json'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
