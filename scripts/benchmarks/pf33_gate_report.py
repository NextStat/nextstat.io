#!/usr/bin/env python3
"""Write a strict PF3.3 benchmark gate report (numbers-only table).

Protocol: /.claude/benchmark-protocol.md

Inputs:
  - run_root produced by scripts/benchmarks/pf31_publication_matrix.sh
  - matrix json used for the run

Output:
  - markdown report that begins with environment snapshot and ends with the mandatory table:
    | Case | NS (median) | Competitor (median) | Speedup | Parity | Status |
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class Run:
    name: str
    rc: int
    n_toys: int | None
    n_converged: int | None
    n_error: int | None
    wall_time_s: float | None
    termination_kind: str | None
    termination_reason: str | None


def extract_counts(metrics: dict[str, Any] | None, out: dict[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    n_toys = None
    n_converged = None
    n_error = None

    if metrics:
        mm = metrics.get("metrics") or {}
        if isinstance(mm, dict):
            if isinstance(mm.get("n_toys"), int):
                n_toys = mm.get("n_toys")
            if isinstance(mm.get("n_converged"), int):
                n_converged = mm.get("n_converged")
            if isinstance(mm.get("n_error"), int):
                n_error = mm.get("n_error")

    if out:
        gen = out.get("gen") or {}
        res = out.get("results") or {}
        if n_toys is None:
            if isinstance(gen.get("n_toys"), int):
                n_toys = gen.get("n_toys")
            elif isinstance(res.get("n_toys"), int):
                n_toys = res.get("n_toys")
        if n_converged is None:
            conv = res.get("converged")
            if isinstance(conv, list):
                n_converged = sum(1 for x in conv if bool(x))
            elif isinstance(res.get("n_converged"), int):
                n_converged = res.get("n_converged")
        if n_error is None and isinstance(res.get("n_error"), int):
            n_error = res.get("n_error")

    return n_toys, n_converged, n_error


def extract_wall_time_s(meta: dict[str, Any], metrics: dict[str, Any] | None) -> float | None:
    if metrics:
        timing = metrics.get("timing") or {}
        if isinstance(timing.get("wall_time_s"), (int, float)):
            return float(timing.get("wall_time_s"))
    if isinstance(meta.get("elapsed_s"), (int, float)):
        return float(meta.get("elapsed_s"))
    return None


def load_run(case_dir: Path, stem: str) -> Run:
    meta = load_json(case_dir / f"{stem}.meta.json") or {}
    metrics = load_json(case_dir / f"{stem}.metrics.json")
    out = load_json(case_dir / f"{stem}.out.json")

    rc = int(meta.get("rc", 1))
    n_toys, n_converged, n_error = extract_counts(metrics, out)
    wall_time_s = extract_wall_time_s(meta, metrics)

    return Run(
        name=str(meta.get("name", stem)),
        rc=rc,
        n_toys=n_toys,
        n_converged=n_converged,
        n_error=n_error,
        wall_time_s=wall_time_s,
        termination_kind=meta.get("termination_kind"),
        termination_reason=meta.get("termination_reason"),
    )


def fmt_s(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.3f}s"


def fmt_speedup(ns_s: float | None, comp_s: float | None) -> str:
    if ns_s is None or comp_s is None or ns_s <= 0 or comp_s <= 0:
        return "-"
    return f"{(comp_s / ns_s):.1f}x"


def parity_str(r: Run) -> tuple[str, bool]:
    conv = None
    if isinstance(r.n_toys, int) and r.n_toys > 0 and isinstance(r.n_converged, int):
        conv = r.n_converged / r.n_toys
    nerr = r.n_error

    ok = (r.rc == 0) and (nerr == 0) and (conv == 1.0 if conv is not None else False)
    conv_s = "-" if conv is None else f"{conv:.3f}"
    nerr_s = "-" if nerr is None else str(nerr)

    if ok:
        return f"ok (conv={conv_s}, n_error={nerr_s})", True
    return f"fail (conv={conv_s}, n_error={nerr_s})", False


def status_str(r: Run, speedup: str, parity_ok: bool) -> str:
    if r.rc != 0:
        if r.termination_kind == "timeout":
            return "warn:timeout"
        return f"warn:rc_{r.rc}"
    if not parity_ok:
        return "fail:parity"
    # Speedup parsing: "-"/"0.7x"/"8.1x"
    if speedup.endswith("x"):
        try:
            val = float(speedup[:-1])
        except ValueError:
            val = 0.0
        if val < 1.0:
            return "fail:slower"
    return "pass"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    matrix_path = Path(args.matrix)
    out_md = Path(args.out_md)

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    cases = matrix.get("cases") or []

    remote_env = (run_root / "remote_env.txt").read_text(encoding="utf-8") if (run_root / "remote_env.txt").exists() else ""

    lines: list[str] = []
    lines.append("## Environment")
    lines.append("```text")
    lines.append(remote_env.rstrip("\n"))
    lines.append("```")
    lines.append("")

    lines.append("| Case | NS (median) | Competitor (median) | Speedup | Parity | Status |")
    lines.append("|------|-------------|---------------------|---------|--------|--------|")

    for c in cases:
        case_id = str(c.get("id"))
        case_dir = run_root / case_id
        if not case_dir.exists():
            # Protocol: no N/A without reason.
            lines.append(f"| {case_id} | - | - | - | fail (conv=-, n_error=-) | warn:missing_case_dir |")
            continue

        # Load all stems from meta files.
        stems = sorted(p.name[: -len(".meta.json")] for p in case_dir.glob("*.meta.json"))
        runs = {stem: load_run(case_dir, stem) for stem in stems}

        # Build cpu baselines by n_toys.
        cpu_by_toys: dict[int, Run] = {}
        for stem, r in runs.items():
            if stem.startswith("cpu_t") and isinstance(r.n_toys, int):
                cpu_by_toys[r.n_toys] = r

        for stem, r in runs.items():
            if not stem.startswith("cuda"):
                continue
            if not isinstance(r.n_toys, int):
                lines.append(f"| {case_id}/{stem} | {fmt_s(r.wall_time_s)} | - | - | fail (conv=-, n_error=-) | warn:missing_n_toys |")
                continue

            cpu = cpu_by_toys.get(r.n_toys)
            comp_s = cpu.wall_time_s if cpu else None
            ns_s = r.wall_time_s
            speedup = fmt_speedup(ns_s, comp_s)
            parity, parity_ok = parity_str(r)
            status = status_str(r, speedup, parity_ok)
            comp_str = fmt_s(comp_s)
            ns_str = fmt_s(ns_s)
            lines.append(f"| {case_id}/{stem} | {ns_str} | {comp_str} | {speedup} | {parity} | {status} |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
