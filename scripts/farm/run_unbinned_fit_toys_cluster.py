#!/usr/bin/env python3
"""Distributed CPU farm launcher for `nextstat unbinned-fit-toys`.

The script shards toys across SSH hosts, runs one shard per host, and saves a
manifest with all shard metadata to support deterministic merging.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class HostEntry:
    host: str
    user: str | None = None
    port: int | None = None
    weight: float | None = None
    threads: int | None = None


@dataclass
class PreflightStat:
    host: str
    ok: bool
    logical_cores: int | None
    physical_cores: int | None


@dataclass
class ShardPlan:
    shard_index: int
    host: str
    user: str | None
    port: int
    weight: float
    threads: int
    toy_start: int
    n_toys: int
    seed: int


@dataclass
class ShardResult:
    shard_index: int
    host: str
    target: str
    status: str
    n_toys: int
    toy_start: int
    seed: int
    threads: int
    weight: float
    rc: int | None
    elapsed_s: float
    local_dir: str
    local_output: str | None
    local_metrics: str | None
    local_stdout: str | None
    local_stderr: str | None
    remote_dir: str
    error: str | None = None


def read_hosts(path: Path) -> list[HostEntry]:
    hosts: list[HostEntry] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        entry = HostEntry(host=parts[0])
        for token in parts[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "weight":
                    entry.weight = float(value)
                elif key == "threads":
                    entry.threads = int(value)
                elif key == "user":
                    entry.user = value
                elif key == "port":
                    entry.port = int(value)
                else:
                    raise ValueError(f"{path}:{lineno}: unknown token {token!r}")
            else:
                if entry.weight is not None:
                    raise ValueError(
                        f"{path}:{lineno}: ambiguous token {token!r}; use key=value format"
                    )
                entry.weight = float(token)
        hosts.append(entry)
    if not hosts:
        raise ValueError(f"hosts file is empty: {path}")
    return hosts


def load_preflight(path: Path | None) -> dict[str, PreflightStat]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data.get("stats")
    if not isinstance(stats, list):
        raise ValueError(f"invalid preflight JSON (missing stats[]): {path}")
    out: dict[str, PreflightStat] = {}
    for item in stats:
        if not isinstance(item, dict):
            continue
        host = item.get("host")
        if not isinstance(host, str) or not host:
            continue
        out[host] = PreflightStat(
            host=host,
            ok=bool(item.get("ok", False)),
            logical_cores=_as_int_or_none(item.get("logical_cores")),
            physical_cores=_as_int_or_none(item.get("physical_cores")),
        )
    return out


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_positive_int(value: int, what: str) -> int:
    if value <= 0:
        raise ValueError(f"{what} must be > 0, got {value}")
    return value


def resolve_threads(entry: HostEntry, preflight: dict[str, PreflightStat], spec: str) -> int:
    if entry.threads is not None:
        return _safe_positive_int(entry.threads, f"threads for host {entry.host}")

    if spec == "nextstat-auto":
        return 0

    if spec.isdigit():
        return _safe_positive_int(int(spec), "--threads")

    st = preflight.get(entry.host)
    if st is None or not st.ok:
        if spec == "auto":
            return 1
        raise ValueError(
            f"host {entry.host}: --threads={spec} requires --preflight-json with reachable host stats"
        )

    logical = st.logical_cores or 1
    physical = st.physical_cores or logical
    if spec == "physical":
        return _safe_positive_int(physical, f"physical cores for host {entry.host}")
    if spec == "logical":
        return _safe_positive_int(logical, f"logical cores for host {entry.host}")
    if spec == "auto":
        return _safe_positive_int(physical if physical > 0 else logical, f"auto threads for host {entry.host}")

    raise ValueError(f"unknown --threads policy: {spec}")


def resolve_weight(entry: HostEntry, preflight: dict[str, PreflightStat], mode: str) -> float:
    if entry.weight is not None:
        if entry.weight <= 0:
            raise ValueError(f"host {entry.host}: weight must be > 0")
        return float(entry.weight)

    if mode == "uniform":
        return 1.0

    st = preflight.get(entry.host)
    if st is None or not st.ok:
        return 1.0

    if mode == "physical":
        val = st.physical_cores
    elif mode == "logical":
        val = st.logical_cores
    else:
        raise ValueError(f"unknown --weight-mode: {mode}")

    if val is None or val <= 0:
        return 1.0
    return float(val)


def split_toys(weights: list[float], n_toys: int) -> list[int]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("sum(weights) must be > 0")

    raw = [(w / total) * n_toys for w in weights]
    base = [int(math.floor(x)) for x in raw]
    remain = n_toys - sum(base)
    frac = [(raw[i] - base[i], i) for i in range(len(weights))]
    frac.sort(key=lambda x: (-x[0], x[1]))
    for k in range(remain):
        base[frac[k][1]] += 1
    return base


def safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def render_remote_workdir(path: str) -> str:
    if path in ("~", "$HOME", "${HOME}"):
        return "$HOME"
    return shlex.quote(path)


def ssh_cmd_base(port: int, ssh_key: str | None, connect_timeout_s: int) -> list[str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={connect_timeout_s}",
        "-p",
        str(port),
    ]
    if ssh_key:
        cmd += ["-i", ssh_key]
    return cmd


def scp_cmd_base(port: int, ssh_key: str | None, connect_timeout_s: int) -> list[str]:
    cmd = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={connect_timeout_s}",
        "-P",
        str(port),
    ]
    if ssh_key:
        cmd += ["-i", ssh_key]
    return cmd


def run_cmd(cmd: list[str], timeout_s: int | None = None) -> tuple[int, str, str, float, str | None]:
    t0 = time.time()
    try:
        cp = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        dt_s = time.time() - t0
        return 124, exc.stdout or "", exc.stderr or "", dt_s, f"timeout ({timeout_s}s)"
    dt_s = time.time() - t0
    return cp.returncode, cp.stdout, cp.stderr, dt_s, None


def run_ssh(
    target: str,
    port: int,
    ssh_key: str | None,
    connect_timeout_s: int,
    remote_cmd: str,
    timeout_s: int | None,
) -> tuple[int, str, str, float, str | None]:
    cmd = ssh_cmd_base(port, ssh_key, connect_timeout_s) + [target, remote_cmd]
    return run_cmd(cmd, timeout_s=timeout_s)


def scp_put(
    local_path: Path,
    target: str,
    remote_path: str,
    port: int,
    ssh_key: str | None,
    connect_timeout_s: int,
) -> tuple[int, str]:
    cmd = scp_cmd_base(port, ssh_key, connect_timeout_s) + [str(local_path), f"{target}:{remote_path}"]
    rc, _out, err, _dt, terr = run_cmd(cmd)
    if terr:
        return rc, terr
    return rc, err.strip()


def scp_get(
    target: str,
    remote_path: str,
    local_path: Path,
    port: int,
    ssh_key: str | None,
    connect_timeout_s: int,
) -> tuple[int, str]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = scp_cmd_base(port, ssh_key, connect_timeout_s) + [f"{target}:{remote_path}", str(local_path)]
    rc, _out, err, _dt, terr = run_cmd(cmd)
    if terr:
        return rc, terr
    return rc, err.strip()


def build_target(user: str | None, host: str) -> str:
    if user:
        return f"{user}@{host}"
    return host


def run_shard(plan: ShardPlan, args: argparse.Namespace, local_run_dir: Path, remote_run_root: str) -> ShardResult:
    user = plan.user if plan.user else args.ssh_user
    target = build_target(user, plan.host)

    shard_name = f"shard_{plan.shard_index:03d}_{safe_name(plan.host)}"
    local_dir = local_run_dir / "shards" / shard_name
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_dir = f"{remote_run_root.rstrip('/')}/{shard_name}"
    remote_spec = args.remote_config if args.remote_config else f"{remote_dir}/spec.json"
    remote_output = f"{remote_dir}/out.json"
    remote_metrics = f"{remote_dir}/metrics.json"
    remote_stdout = f"{remote_dir}/stdout.log"
    remote_stderr = f"{remote_dir}/stderr.log"

    if args.dry_run:
        return ShardResult(
            shard_index=plan.shard_index,
            host=plan.host,
            target=target,
            status="planned",
            n_toys=plan.n_toys,
            toy_start=plan.toy_start,
            seed=plan.seed,
            threads=plan.threads,
            weight=plan.weight,
            rc=None,
            elapsed_s=0.0,
            local_dir=str(local_dir),
            local_output=None,
            local_metrics=None,
            local_stdout=None,
            local_stderr=None,
            remote_dir=remote_dir,
        )

    mkdir_cmd = f"mkdir -p {shlex.quote(remote_dir)}"
    rc, _out, err, _dt, terr = run_ssh(
        target=target,
        port=plan.port,
        ssh_key=args.ssh_key,
        connect_timeout_s=args.connect_timeout_s,
        remote_cmd=mkdir_cmd,
        timeout_s=args.command_timeout_s,
    )
    if rc != 0:
        return ShardResult(
            shard_index=plan.shard_index,
            host=plan.host,
            target=target,
            status="failed",
            n_toys=plan.n_toys,
            toy_start=plan.toy_start,
            seed=plan.seed,
            threads=plan.threads,
            weight=plan.weight,
            rc=rc,
            elapsed_s=0.0,
            local_dir=str(local_dir),
            local_output=None,
            local_metrics=None,
            local_stdout=None,
            local_stderr=None,
            remote_dir=remote_dir,
            error=terr or err or "remote mkdir failed",
        )

    if args.upload_config and not args.remote_config:
        put_rc, put_err = scp_put(
            local_path=args.config,
            target=target,
            remote_path=remote_spec,
            port=plan.port,
            ssh_key=args.ssh_key,
            connect_timeout_s=args.connect_timeout_s,
        )
        if put_rc != 0:
            return ShardResult(
                shard_index=plan.shard_index,
                host=plan.host,
                target=target,
                status="failed",
                n_toys=plan.n_toys,
                toy_start=plan.toy_start,
                seed=plan.seed,
                threads=plan.threads,
                weight=plan.weight,
                rc=put_rc,
                elapsed_s=0.0,
                local_dir=str(local_dir),
                local_output=None,
                local_metrics=None,
                local_stdout=None,
                local_stderr=None,
                remote_dir=remote_dir,
                error=f"scp config failed: {put_err}",
            )

    tokens = [
        args.nextstat_bin,
        "unbinned-fit-toys",
        "--config",
        remote_spec,
        "--n-toys",
        str(plan.n_toys),
        "--seed",
        str(plan.seed),
        "--threads",
        str(plan.threads),
        "--output",
        remote_output,
        "--json-metrics",
        remote_metrics,
        "--log-level",
        args.log_level,
    ]
    tokens.extend(args.extra_arg)
    nextstat_cmd = " ".join(shlex.quote(t) for t in tokens)

    lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(remote_dir)}",
        f"cd {render_remote_workdir(args.remote_workdir)}",
    ]
    if args.pin_blas_threads:
        lines.extend(
            [
                "export OMP_NUM_THREADS=1",
                "export OPENBLAS_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "export VECLIB_MAXIMUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
            ]
        )
    lines.append(f"{nextstat_cmd} > {shlex.quote(remote_stdout)} 2> {shlex.quote(remote_stderr)}")
    remote_script = "\n".join(lines)
    remote_cmd = f"bash -lc {shlex.quote(remote_script)}"

    rc, _out, err, elapsed_s, terr = run_ssh(
        target=target,
        port=plan.port,
        ssh_key=args.ssh_key,
        connect_timeout_s=args.connect_timeout_s,
        remote_cmd=remote_cmd,
        timeout_s=args.command_timeout_s,
    )

    local_output = local_dir / "out.json"
    local_metrics = local_dir / "metrics.json"
    local_stdout = local_dir / "stdout.log"
    local_stderr = local_dir / "stderr.log"

    pull_errors: list[str] = []
    for remote_path, local_path in [
        (remote_output, local_output),
        (remote_metrics, local_metrics),
        (remote_stdout, local_stdout),
        (remote_stderr, local_stderr),
    ]:
        c_rc, c_err = scp_get(
            target=target,
            remote_path=remote_path,
            local_path=local_path,
            port=plan.port,
            ssh_key=args.ssh_key,
            connect_timeout_s=args.connect_timeout_s,
        )
        if c_rc != 0:
            pull_errors.append(f"{remote_path}: {c_err}")

    status = "ok" if rc == 0 else "failed"
    error_parts: list[str] = []
    if terr:
        error_parts.append(terr)
    if err.strip():
        error_parts.append(err.strip())
    if pull_errors:
        error_parts.append("pull: " + " | ".join(pull_errors))

    return ShardResult(
        shard_index=plan.shard_index,
        host=plan.host,
        target=target,
        status=status,
        n_toys=plan.n_toys,
        toy_start=plan.toy_start,
        seed=plan.seed,
        threads=plan.threads,
        weight=plan.weight,
        rc=rc,
        elapsed_s=elapsed_s,
        local_dir=str(local_dir),
        local_output=str(local_output) if local_output.exists() else None,
        local_metrics=str(local_metrics) if local_metrics.exists() else None,
        local_stdout=str(local_stdout) if local_stdout.exists() else None,
        local_stderr=str(local_stderr) if local_stderr.exists() else None,
        remote_dir=remote_dir,
        error="; ".join(error_parts) if error_parts else None,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run nextstat unbinned-fit-toys across an SSH CPU farm and emit shard manifest."
    )
    ap.add_argument("--hosts-file", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path, help="local spec path")
    ap.add_argument("--n-toys", required=True, type=int)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--preflight-json", type=Path, default=None)
    ap.add_argument(
        "--threads",
        default="physical",
        help="per-host threads: physical|logical|auto|nextstat-auto|<int> (host threads=... overrides)",
    )
    ap.add_argument(
        "--weight-mode",
        choices=["uniform", "physical", "logical"],
        default="physical",
        help="toy split weights when host weight=... is not set",
    )

    ap.add_argument("--nextstat-bin", default="nextstat", help="remote nextstat binary path")
    ap.add_argument("--remote-workdir", default="$HOME", help="remote cwd before running nextstat")
    ap.add_argument(
        "--remote-base-dir",
        default="/tmp/nextstat_cpu_farm",
        help="remote base directory for this run",
    )
    ap.add_argument(
        "--remote-config",
        default=None,
        help="remote config path (disables config upload if set)",
    )
    ap.add_argument(
        "--upload-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="upload local config to each host (default: true)",
    )

    ap.add_argument("--ssh-user", default=None)
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--ssh-port", type=int, default=22)
    ap.add_argument("--connect-timeout-s", type=int, default=15)
    ap.add_argument("--command-timeout-s", type=int, default=172800)

    ap.add_argument("--jobs", type=int, default=0, help="parallel hosts (0 = auto)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--local-out-dir", required=True, type=Path)
    ap.add_argument("--log-level", default="warn")
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="extra token forwarded to nextstat (repeat flag for multiple tokens)",
    )
    ap.add_argument(
        "--pin-blas-threads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="set OMP/BLAS env vars to 1 on remote (default: true)",
    )
    ap.add_argument("--allow-partial", action="store_true", help="exit 0 if at least one shard succeeded")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def print_plan(plans: list[ShardPlan]) -> None:
    print("shard  host                          toys      toy_start  seed      threads  weight")
    print("-----  ----------------------------  --------  ---------  --------  -------  ------")
    for p in plans:
        print(
            f"{p.shard_index:>5}  {p.host:<28}  {p.n_toys:>8}  {p.toy_start:>9}  {p.seed:>8}  {p.threads:>7}  {p.weight:>6.1f}"
        )


def print_results(results: list[ShardResult]) -> None:
    print()
    print("shard  host                          status   rc    elapsed_s  toys")
    print("-----  ----------------------------  -------  ----  ---------  --------")
    for r in sorted(results, key=lambda x: x.shard_index):
        rc = "-" if r.rc is None else str(r.rc)
        print(f"{r.shard_index:>5}  {r.host:<28}  {r.status:<7}  {rc:>4}  {r.elapsed_s:>9.2f}  {r.n_toys:>8}")


def main() -> int:
    args = parse_args()

    if args.n_toys <= 0:
        raise SystemExit("--n-toys must be > 0")
    if args.seed < 0:
        raise SystemExit("--seed must be >= 0")

    hosts = read_hosts(args.hosts_file)
    preflight = load_preflight(args.preflight_json)

    weights: list[float] = []
    threads: list[int] = []
    ports: list[int] = []
    for entry in hosts:
        weights.append(resolve_weight(entry, preflight, args.weight_mode))
        threads.append(resolve_threads(entry, preflight, args.threads))
        ports.append(entry.port if entry.port is not None else args.ssh_port)

    shard_sizes = split_toys(weights, args.n_toys)

    run_id = args.run_id or dt.datetime.now(tz=dt.timezone.utc).strftime("cpufarm_%Y%m%dT%H%M%SZ")
    local_run_dir = args.local_out_dir / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)
    remote_run_root = f"{args.remote_base_dir.rstrip('/')}/{run_id}"

    plans: list[ShardPlan] = []
    toy_start = 0
    for i, entry in enumerate(hosts):
        n_i = shard_sizes[i]
        if n_i <= 0:
            continue
        plans.append(
            ShardPlan(
                shard_index=len(plans),
                host=entry.host,
                user=entry.user,
                port=ports[i],
                weight=weights[i],
                threads=threads[i],
                toy_start=toy_start,
                n_toys=n_i,
                seed=args.seed + toy_start,
            )
        )
        toy_start += n_i

    if not plans:
        raise SystemExit("no shard plans generated (all hosts got 0 toys)")

    print(f"run_id={run_id}")
    print(f"local_run_dir={local_run_dir}")
    print(f"remote_run_root={remote_run_root}")
    print_plan(plans)

    results: list[ShardResult] = []
    if args.dry_run:
        results = [run_shard(plan, args, local_run_dir, remote_run_root) for plan in plans]
    else:
        max_jobs = args.jobs if args.jobs > 0 else min(len(plans), max(1, os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_jobs)) as ex:
            futs = [
                ex.submit(run_shard, plan, args, local_run_dir, remote_run_root)
                for plan in plans
            ]
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda x: x.shard_index)
    print_results(results)

    manifest = {
        "schema_version": "nextstat.cpu_farm_unbinned_fit_toys_run.v1",
        "created_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "hosts_file": str(args.hosts_file),
        "config": str(args.config),
        "n_toys": args.n_toys,
        "seed": args.seed,
        "threads": args.threads,
        "weight_mode": args.weight_mode,
        "nextstat_bin": args.nextstat_bin,
        "remote_workdir": args.remote_workdir,
        "remote_run_root": remote_run_root,
        "dry_run": args.dry_run,
        "plans": [asdict(p) for p in plans],
        "results": [asdict(r) for r in results],
    }
    manifest_path = local_run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}")

    n_ok = sum(1 for r in results if r.status == "ok")
    n_failed = sum(1 for r in results if r.status == "failed")
    print(f"ok_shards={n_ok}, failed_shards={n_failed}, total_shards={len(results)}")

    if args.dry_run:
        return 0
    if n_failed == 0:
        return 0
    if args.allow_partial and n_ok > 0:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
