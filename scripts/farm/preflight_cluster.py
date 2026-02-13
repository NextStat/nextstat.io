#!/usr/bin/env python3
"""CPU farm preflight for NextStat toy workloads.

Collects per-node CPU topology (logical/physical cores, HT ratio) over SSH
and prints a compact table + optional JSON artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class HostStat:
    host: str
    ok: bool
    ssh_target: str
    hostname: str | None = None
    logical_cores: int | None = None
    physical_cores: int | None = None
    threads_per_core: float | None = None
    cpu_model: str | None = None
    error: str | None = None


def read_hosts(path: Path) -> list[str]:
    hosts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        hosts.append(parts[0])
    if not hosts:
        raise ValueError(f"hosts file is empty: {path}")
    return hosts


def ssh_target(host: str, ssh_user: str | None) -> str:
    if ssh_user is None:
        return host
    return f"{ssh_user}@{host}"


def run_ssh(
    target: str,
    remote_cmd: str,
    ssh_key: str | None,
    ssh_port: int,
    timeout_s: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={timeout_s}",
        "-p",
        str(ssh_port),
    ]
    if ssh_key:
        cmd += ["-i", ssh_key]
    cmd += [target, remote_cmd]
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s + 5, check=False)


def collect_one(
    host: str,
    ssh_user: str | None,
    ssh_key: str | None,
    ssh_port: int,
    timeout_s: int,
) -> HostStat:
    target = ssh_target(host, ssh_user)
    remote_cmd = r"""bash -lc '
set -euo pipefail
host_name="$(hostname -s 2>/dev/null || hostname)"
logical="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
if command -v lscpu >/dev/null 2>&1; then
  physical="$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v "^#" | sort -u | wc -l | tr -d " ")"
  tpc="$(lscpu 2>/dev/null | awk -F: "/Thread\\(s\\) per core/ {gsub(/^[ \t]+/, \"\", $2); print $2; exit}")"
  model="$(lscpu 2>/dev/null | awk -F: "/Model name/ {sub(/^[ \t]+/, \"\", $2); print $2; exit}")"
else
  physical="$logical"
  tpc="1"
  model="unknown"
fi
printf "%s\t%s\t%s\t%s\t%s\n" "$host_name" "$logical" "$physical" "${tpc:-1}" "${model:-unknown}"
'"""
    try:
        cp = run_ssh(target, remote_cmd, ssh_key, ssh_port, timeout_s)
    except subprocess.TimeoutExpired:
        return HostStat(host=host, ok=False, ssh_target=target, error=f"timeout ({timeout_s}s)")

    if cp.returncode != 0:
        err = (cp.stderr or cp.stdout or "").strip()
        return HostStat(host=host, ok=False, ssh_target=target, error=err or f"ssh rc={cp.returncode}")

    line = cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else ""
    parts = line.split("\t", 4)
    if len(parts) != 5:
        return HostStat(
            host=host,
            ok=False,
            ssh_target=target,
            error=f"unexpected preflight output: {line!r}",
        )
    host_name, logical, physical, tpc, model = parts
    try:
        logical_i = int(logical)
        physical_i = int(physical)
        tpc_f = float(tpc) if tpc else (logical_i / max(physical_i, 1))
    except ValueError as e:
        return HostStat(host=host, ok=False, ssh_target=target, error=f"parse error: {e}")
    return HostStat(
        host=host,
        ok=True,
        ssh_target=target,
        hostname=host_name,
        logical_cores=logical_i,
        physical_cores=max(1, physical_i),
        threads_per_core=tpc_f,
        cpu_model=model,
    )


def collect_all(
    hosts: Iterable[str],
    ssh_user: str | None,
    ssh_key: str | None,
    ssh_port: int,
    timeout_s: int,
    jobs: int,
) -> list[HostStat]:
    out: list[HostStat] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = [
            ex.submit(collect_one, h, ssh_user, ssh_key, ssh_port, timeout_s)
            for h in hosts
        ]
        for fut in concurrent.futures.as_completed(futs):
            out.append(fut.result())
    out.sort(key=lambda x: x.host)
    return out


def print_table(stats: list[HostStat]) -> None:
    header = ("host", "ok", "logical", "physical", "ht", "hostname", "cpu_model/error")
    rows = []
    for s in stats:
        if s.ok:
            rows.append(
                (
                    s.host,
                    "yes",
                    str(s.logical_cores),
                    str(s.physical_cores),
                    f"{s.threads_per_core:.2f}" if s.threads_per_core else "-",
                    s.hostname or "-",
                    s.cpu_model or "-",
                )
            )
        else:
            rows.append((s.host, "no", "-", "-", "-", "-", s.error or "unknown error"))

    widths = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt(cols: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(fmt(header))
    print(fmt(tuple("-" * w for w in widths)))
    for r in rows:
        print(fmt(r))

    ok_stats = [s for s in stats if s.ok]
    logical_total = sum(s.logical_cores or 0 for s in ok_stats)
    physical_total = sum(s.physical_cores or 0 for s in ok_stats)
    print()
    print(f"reachable_hosts={len(ok_stats)}/{len(stats)}")
    print(f"total_logical_cores={logical_total}")
    print(f"total_physical_cores={physical_total}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight CPU farm nodes via SSH.")
    ap.add_argument("--hosts-file", required=True, type=Path)
    ap.add_argument("--ssh-user", default=None)
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--ssh-port", type=int, default=22)
    ap.add_argument("--timeout-s", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8, help="parallel SSH probes")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = ap.parse_args()

    hosts = read_hosts(args.hosts_file)
    stats = collect_all(
        hosts,
        ssh_user=args.ssh_user,
        ssh_key=args.ssh_key,
        ssh_port=args.ssh_port,
        timeout_s=args.timeout_s,
        jobs=args.jobs,
    )
    print_table(stats)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "nextstat.cpu_farm_preflight.v1",
            "created_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            "hosts_file": str(args.hosts_file),
            "stats": [asdict(s) for s in stats],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 0 if any(s.ok for s in stats) else 2


if __name__ == "__main__":
    sys.exit(main())
