#!/usr/bin/env python3
"""PF3.2 shard-sweep runner for CPU-farm scaling studies.

This tool is scheduler-oriented (SLURM/HTCondor) and builds on:
  - scripts/farm/render_unbinned_fit_toys_scheduler.py
  - scripts/farm/collect_unbinned_fit_toys_scheduler.py
  - scripts/farm/merge_unbinned_toys_results.py

Workflow:
  1) render: create one run directory per shard count + sweep.json
  2) submit: optionally submit and run HTCondor sweeps sequentially (avoid contention)
  3) collect: populate manifest results[] for each run directory
  4) merge: produce merged.out.json for each run directory
  5) summarize: write a markdown table (makespan + throughput toys/s)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_tool(filename: str) -> Path:
    """Resolve a helper script path.

    Primary mode: run from repo (ROOT_DIR/scripts/farm/...).
    Fallback mode: run from a standalone copied directory containing the tools
    (same directory as this script).
    """
    repo_path = ROOT_DIR / "scripts/farm" / filename
    if repo_path.exists():
        return repo_path

    local_path = Path(__file__).resolve().parent / filename
    if local_path.exists():
        return local_path

    raise FileNotFoundError(f"cannot resolve {filename} (tried {repo_path} and {local_path})")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _run(cmd: list[str]) -> None:
    cp = subprocess.run(cmd, text=True)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)


def _split_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty list")
    if any(x <= 0 for x in out):
        raise ValueError("all values must be > 0")
    return out


def _split_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("empty list")
    if any(x <= 0.0 for x in out):
        raise ValueError("all values must be > 0")
    return out


def _auto_shards_from_preflight(preflight_json: Path, base: str, factors: list[float]) -> tuple[list[int], int]:
    data = _load_json(preflight_json)
    stats = data.get("stats")
    if not isinstance(stats, list):
        raise ValueError(f"invalid preflight JSON (missing stats[]): {preflight_json}")
    total = 0
    key = "physical_cores" if base == "physical" else "logical_cores"
    for item in stats:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("ok")):
            continue
        val = item.get(key)
        if isinstance(val, int) and val > 0:
            total += val
    if total <= 0:
        raise ValueError(f"no reachable {key} in preflight JSON: {preflight_json}")

    shards = sorted(
        {
            max(1, int(math.ceil(total * f)))
            for f in factors
        }
    )
    return shards, total


@dataclass(frozen=True)
class RunRef:
    shards: int
    run_id: str
    run_dir: Path


def cmd_render(args: argparse.Namespace) -> int:
    stamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_id = args.sweep_id or f"pf32_shard_sweep_{stamp}"
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    auto_total_cores = None
    if args.shards_list:
        shards_list = _split_int_list(args.shards_list)
    else:
        if args.preflight_json is None:
            raise SystemExit("render requires either --shards-list or --preflight-json + --shard-factors")
        factors = _split_float_list(args.shard_factors)
        shards_list, auto_total_cores = _auto_shards_from_preflight(
            preflight_json=Path(args.preflight_json).expanduser().resolve(),
            base=args.auto_shards_base,
            factors=factors,
        )
        print(
            f"auto shards from preflight ({args.auto_shards_base} cores={auto_total_cores}, "
            f"factors={','.join(str(f) for f in factors)}): {','.join(str(s) for s in shards_list)}"
        )
    runs: list[RunRef] = []

    render_script = _resolve_tool("render_unbinned_fit_toys_scheduler.py")
    for shards in shards_list:
        run_id = f"{sweep_id}_sh{shards}"
        cmd = [
            sys.executable,
            str(render_script),
            "--scheduler",
            args.scheduler,
            "--config",
            str(Path(args.config).expanduser().resolve()),
            "--n-toys",
            str(args.n_toys),
            "--seed",
            str(args.seed),
            "--shards",
            str(shards),
            "--threads",
            str(args.threads),
            "--nextstat-bin",
            args.nextstat_bin,
            "--out-dir",
            str(out_root),
            "--run-id",
            run_id,
        ]
        if args.scheduler == "slurm":
            if args.slurm_cpus_per_task is not None:
                cmd += ["--slurm-cpus-per-task", str(args.slurm_cpus_per_task)]
            if args.slurm_time is not None:
                cmd += ["--slurm-time", str(args.slurm_time)]
        else:
            if args.condor_request_cpus is not None:
                cmd += ["--condor-request-cpus", str(args.condor_request_cpus)]
            if args.condor_request_memory is not None:
                cmd += ["--condor-request-memory", str(args.condor_request_memory)]
            if args.condor_requirements is not None:
                cmd += ["--condor-requirements", str(args.condor_requirements)]

        _run(cmd)
        run_dir = out_root / run_id
        runs.append(RunRef(shards=shards, run_id=run_id, run_dir=run_dir))

    sweep = {
        "schema_version": "nextstat.pf32_shard_sweep.v1",
        "created_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "sweep_id": sweep_id,
        "scheduler": args.scheduler,
        "config": str(Path(args.config).expanduser().resolve()),
        "n_toys": int(args.n_toys),
        "seed": int(args.seed),
        "threads": int(args.threads),
        "nextstat_bin": args.nextstat_bin,
        "out_root": str(out_root),
        "shards_mode": "manual" if args.shards_list else "auto",
        "auto_shards_base": args.auto_shards_base if not args.shards_list else None,
        "auto_shards_total_cores": auto_total_cores,
        "shard_factors": args.shard_factors if not args.shards_list else None,
        "runs": [
            {"shards": r.shards, "run_id": r.run_id, "run_dir": str(r.run_dir)}
            for r in runs
        ],
    }
    sweep_path = out_root / f"{sweep_id}.sweep.json"
    _write_json(sweep_path, sweep)
    print(f"wrote {sweep_path}")
    return 0


def _submit_one_condor(sub_path: Path) -> int:
    cp = subprocess.run(
        ["condor_submit", str(sub_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"condor_submit failed (rc={cp.returncode}) for {sub_path}\n"
            f"{cp.stdout}\n{cp.stderr}"
        )

    # Preserve submit output in stdout for run logs.
    sys.stdout.write(cp.stdout)
    sys.stdout.write(cp.stderr)

    m = re.search(r"submitted to cluster\\s+(\\d+)", cp.stdout)
    if not m:
        raise RuntimeError(f"failed to parse cluster id from condor_submit output for {sub_path}")
    return int(m.group(1))


def _condor_cluster_active(cluster_id: int) -> bool:
    # When a cluster is gone, condor_q exits non-zero. Treat that as "done".
    cp = subprocess.run(
        ["condor_q", str(cluster_id), "-autoformat", "ClusterId"],
        check=False,
        text=True,
        capture_output=True,
    )
    if cp.returncode != 0:
        return False
    return bool(cp.stdout.strip())


def cmd_submit(args: argparse.Namespace) -> int:
    """Sequential runner for HTCondor sweeps (one run dir at a time)."""
    if int(args.submit_retries) < 0:
        raise SystemExit("--submit-retries must be >= 0")
    if float(args.retry_backoff_s) < 0:
        raise SystemExit("--retry-backoff-s must be >= 0")

    sweep_path = Path(args.sweep_json).expanduser().resolve()
    sweep = _load_json(sweep_path)

    if sweep.get("scheduler") != "htcondor":
        raise SystemExit("submit is only supported for scheduler=htcondor")

    runs = sweep.get("runs") or []
    if not runs:
        raise SystemExit("no runs in sweep.json")

    submit_file = "condor_job_transfer.sub" if args.mode == "transfer" else "condor_job.sub"
    collect_script = _resolve_tool("collect_unbinned_fit_toys_scheduler.py")
    merge_script = _resolve_tool("merge_unbinned_toys_results.py")
    n_failed_runs = 0

    for r in runs:
        run_dir = Path(r["run_dir"]).expanduser().resolve()
        sub_path = run_dir / submit_file
        if not sub_path.exists():
            msg = f"missing submit file: {sub_path}"
            n_failed_runs += 1
            if not args.continue_on_error:
                raise SystemExit(msg)
            print(f"ERROR: {msg}")
            continue

        meta_path = run_dir / "submit_meta.json"
        if meta_path.exists() and not args.force:
            print(f"skip (already has submit_meta.json): {run_dir}")
            continue

        t0 = dt.datetime.now(tz=dt.timezone.utc).isoformat()
        print(f"=== submit: {run_dir} ({submit_file}) ===")
        cluster_id: int | None = None
        submit_error: str | None = None
        submit_attempts = 0
        for attempt in range(max(1, int(args.submit_retries) + 1)):
            submit_attempts = attempt + 1
            try:
                cluster_id = _submit_one_condor(sub_path)
                submit_error = None
                break
            except Exception as exc:
                submit_error = str(exc)
                if attempt + 1 < max(1, int(args.submit_retries) + 1):
                    time.sleep(max(0.0, float(args.retry_backoff_s)) * (2 ** attempt))

        if cluster_id is None:
            n_failed_runs += 1
            t1 = dt.datetime.now(tz=dt.timezone.utc).isoformat()
            _write_json(
                meta_path,
                {
                    "schema_version": "nextstat.pf32_htcondor_submit_meta.v1",
                    "created_at_utc": t0,
                    "finished_at_utc": t1,
                    "cluster_id": None,
                    "submit_file": submit_file,
                    "status": "failed_submit",
                    "submit_attempts": submit_attempts,
                    "error": submit_error,
                },
            )
            if not args.continue_on_error:
                raise SystemExit(submit_error or f"failed to submit {sub_path}")
            print(f"ERROR: failed to submit {sub_path}: {submit_error}")
            continue

        print(f"cluster_id={cluster_id}")

        while _condor_cluster_active(cluster_id):
            print(f"waiting: cluster_id={cluster_id}")
            time.sleep(float(args.poll_s))

        t1 = dt.datetime.now(tz=dt.timezone.utc).isoformat()
        post_error: str | None = None

        if not args.no_collect:
            try:
                _run(
                    [
                        sys.executable,
                        str(collect_script),
                        "--manifest",
                        str(run_dir / "manifest.json"),
                        "--in-place",
                    ]
                )
            except SystemExit as exc:
                post_error = f"collect failed (rc={exc.code})"
        if not args.no_merge and post_error is None:
            try:
                _run(
                    [
                        sys.executable,
                        str(merge_script),
                        "--manifest",
                        str(run_dir / "manifest.json"),
                        "--out",
                        str(run_dir / "merged.out.json"),
                    ]
                )
            except SystemExit as exc:
                post_error = f"merge failed (rc={exc.code})"

        status = "ok" if post_error is None else "failed_post"
        if post_error is not None:
            n_failed_runs += 1
            if not args.continue_on_error:
                raise SystemExit(post_error)
            print(f"ERROR: {post_error} for {run_dir}")

        _write_json(
            meta_path,
            {
                "schema_version": "nextstat.pf32_htcondor_submit_meta.v1",
                "created_at_utc": t0,
                "finished_at_utc": t1,
                "cluster_id": cluster_id,
                "submit_file": submit_file,
                "status": status,
                "submit_attempts": submit_attempts,
                "error": post_error,
            },
        )

    return 0 if n_failed_runs == 0 else 2


def cmd_collect(args: argparse.Namespace) -> int:
    sweep = _load_json(Path(args.sweep_json))
    runs = sweep.get("runs") or []
    collect_script = _resolve_tool("collect_unbinned_fit_toys_scheduler.py")

    for r in runs:
        run_dir = Path(r["run_dir"])
        manifest = run_dir / "manifest.json"
        cmd = [
            sys.executable,
            str(collect_script),
            "--manifest",
            str(manifest),
            "--in-place",
        ]
        _run(cmd)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    sweep = _load_json(Path(args.sweep_json))
    runs = sweep.get("runs") or []
    merge_script = _resolve_tool("merge_unbinned_toys_results.py")

    for r in runs:
        run_dir = Path(r["run_dir"])
        manifest = run_dir / "manifest.json"
        out = run_dir / "merged.out.json"
        try:
            cmd = [
                sys.executable,
                str(merge_script),
                "--manifest",
                str(manifest),
                "--out",
                str(out),
            ]
            _run(cmd)
        except SystemExit:
            # Allow partial sweeps (e.g., while jobs are still running).
            # The summarize step will reflect ok/failed shard counts.
            continue
    return 0


def _summarize_one(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)

    shards = int(manifest.get("shards") or 0)
    n_toys = int(manifest.get("n_toys") or 0)
    threads = manifest.get("threads")

    results = manifest.get("results") or []
    ok = [r for r in results if isinstance(r, dict) and r.get("status") == "ok"]
    failed = [r for r in results if isinstance(r, dict) and r.get("status") == "failed"]

    elapsed_ok: list[float] = []
    for r in ok:
        v = r.get("elapsed_s")
        if isinstance(v, (int, float)) and float(v) >= 0.0:
            elapsed_ok.append(float(v))
    makespan_s = max(elapsed_ok) if (elapsed_ok and max(elapsed_ok) > 0.0) else None

    merged_path = run_dir / "merged.out.json"
    n_error = None
    n_converged = None
    if merged_path.exists():
        merged = _load_json(merged_path)
        res = merged.get("results") or {}
        if isinstance(res, dict):
            if isinstance(res.get("n_error"), int):
                n_error = int(res.get("n_error"))
            if isinstance(res.get("n_converged"), int):
                n_converged = int(res.get("n_converged"))

    toys_per_s = None
    if makespan_s is not None and makespan_s > 0 and n_toys > 0:
        toys_per_s = n_toys / makespan_s

    return {
        "run_dir": str(run_dir),
        "shards": shards,
        "n_toys": n_toys,
        "threads": threads,
        "ok_shards": len(ok),
        "failed_shards": len(failed),
        "makespan_s": makespan_s,
        "toys_per_s": toys_per_s,
        "n_converged": n_converged,
        "n_error": n_error,
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    sweep = _load_json(Path(args.sweep_json))
    runs = sweep.get("runs") or []

    rows = [_summarize_one(Path(r["run_dir"])) for r in runs]
    rows.sort(key=lambda x: int(x.get("shards") or 0))
    baseline = next((r for r in rows if isinstance(r.get("makespan_s"), (int, float)) and float(r["makespan_s"]) > 0), None)
    baseline_shards = int(baseline["shards"]) if baseline else None
    baseline_makespan = float(baseline["makespan_s"]) if baseline else None

    out_md = Path(args.out_md).expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# PF3.2 Unbinned CPU Farm Shard Sweep")
    lines.append("")
    lines.append(f"- sweep_id: `{sweep.get('sweep_id')}`")
    lines.append(f"- scheduler: `{sweep.get('scheduler')}`")
    lines.append(f"- config: `{sweep.get('config')}`")
    lines.append(f"- n_toys: `{sweep.get('n_toys')}`")
    lines.append(f"- seed: `{sweep.get('seed')}`")
    lines.append(f"- threads: `{sweep.get('threads')}`")
    lines.append("")

    lines.append("| Shards | ok/total | Makespan | Throughput | Speedup | Efficiency | n_converged | n_error | Run dir |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        shards = r["shards"]
        ok_total = f'{r["ok_shards"]}/{r["ok_shards"] + r["failed_shards"]}'
        makespan = "-" if r["makespan_s"] is None else f'{r["makespan_s"]:.2f}s'
        tps = "-" if r["toys_per_s"] is None else f'{r["toys_per_s"]:.3f} toys/s'
        speedup = "-"
        eff = "-"
        if (
            baseline_shards is not None
            and baseline_makespan is not None
            and isinstance(r["makespan_s"], (int, float))
            and float(r["makespan_s"]) > 0
        ):
            sp = baseline_makespan / float(r["makespan_s"])
            speedup = f"{sp:.2f}x"
            if shards > 0 and baseline_shards > 0:
                eff_v = sp / (float(shards) / float(baseline_shards))
                eff = f"{eff_v:.2f}"
        n_conv = "-" if r["n_converged"] is None else str(r["n_converged"])
        n_err = "-" if r["n_error"] is None else str(r["n_error"])
        run_dir = r["run_dir"]
        lines.append(
            f"| {shards} | {ok_total} | {makespan} | {tps} | {speedup} | {eff} | {n_conv} | {n_err} | `{run_dir}` |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_r = sub.add_parser("render")
    ap_r.add_argument("--scheduler", choices=["slurm", "htcondor"], required=True)
    ap_r.add_argument("--config", required=True)
    ap_r.add_argument("--n-toys", type=int, required=True)
    ap_r.add_argument("--seed", type=int, default=42)
    ap_r.add_argument("--threads", type=int, default=0)
    ap_r.add_argument("--nextstat-bin", required=True)
    ap_r.add_argument("--out-root", type=Path, required=True)
    ap_r.add_argument("--sweep-id", default=None)
    ap_r.add_argument("--shards-list", default=None, help="comma list, e.g. 50,100,200,400")
    ap_r.add_argument("--preflight-json", type=Path, default=None, help="preflight JSON for auto shard matrix")
    ap_r.add_argument(
        "--auto-shards-base",
        choices=["physical", "logical"],
        default="physical",
        help="base core count for auto shard matrix (used when --shards-list is omitted)",
    )
    ap_r.add_argument(
        "--shard-factors",
        default="0.5,1.0,1.5,2.0,3.0",
        help="comma list of shard multipliers for auto matrix (used when --shards-list is omitted)",
    )
    ap_r.add_argument("--slurm-cpus-per-task", type=int, default=None)
    ap_r.add_argument("--slurm-time", default=None)
    ap_r.add_argument("--condor-request-cpus", type=int, default=None)
    ap_r.add_argument("--condor-request-memory", type=str, default=None)
    ap_r.add_argument("--condor-requirements", type=str, default=None)
    ap_r.set_defaults(func=cmd_render)

    ap_sub = sub.add_parser("submit")
    ap_sub.add_argument("--sweep-json", required=True)
    ap_sub.add_argument("--mode", choices=["transfer", "shared"], default="transfer")
    ap_sub.add_argument("--poll-s", type=float, default=10.0)
    ap_sub.add_argument("--no-collect", action="store_true", default=False)
    ap_sub.add_argument("--no-merge", action="store_true", default=False)
    ap_sub.add_argument("--force", action="store_true", default=False)
    ap_sub.add_argument("--submit-retries", type=int, default=0, help="retry condor_submit failures per run")
    ap_sub.add_argument("--retry-backoff-s", type=float, default=5.0, help="submit retry base backoff seconds")
    ap_sub.add_argument("--continue-on-error", action="store_true", default=False, help="continue sweep on failed run submit/collect/merge")
    ap_sub.set_defaults(func=cmd_submit)

    ap_c = sub.add_parser("collect")
    ap_c.add_argument("--sweep-json", required=True)
    ap_c.set_defaults(func=cmd_collect)

    ap_m = sub.add_parser("merge")
    ap_m.add_argument("--sweep-json", required=True)
    ap_m.set_defaults(func=cmd_merge)

    ap_s = sub.add_parser("summarize")
    ap_s.add_argument("--sweep-json", required=True)
    ap_s.add_argument("--out-md", required=True)
    ap_s.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
