#!/usr/bin/env python3
"""Aggregate PF3.1 publication benchmark artifacts into a compact summary.

Expected layout:
  <run-root>/<case-id>/*.meta.json
  <run-root>/<case-id>/*.metrics.json (optional)
  <run-root>/<case-id>/*.out.json (optional)
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_case(name: str) -> str:
    if "_device_sh" in name:
        return "device_sharded"
    if "_host_sh" in name:
        return "host_sharded"
    if "_native_t" in name:
        return "native"
    if "_host_t" in name:
        return "host"
    if name.startswith("cpu_"):
        return "cpu"
    return "unknown"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some CLI outputs prepend log lines before the JSON object.
        start = text.find("{")
        if start >= 0:
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                return None
        return None


@dataclass
class RunRow:
    case_id: str
    name: str
    fit_mode: str
    rc: int
    n_toys: int | None
    n_converged: int | None
    n_error: int | None
    convergence_rate: float | None
    wall_time_s: float | None
    toys_per_s: float | None
    pipeline: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "fit_mode": self.fit_mode,
            "rc": self.rc,
            "n_toys": self.n_toys,
            "n_converged": self.n_converged,
            "n_error": self.n_error,
            "convergence_rate": self.convergence_rate,
            "wall_time_s": self.wall_time_s,
            "toys_per_s": self.toys_per_s,
            "pipeline": self.pipeline,
        }


def extract_counts(metrics: dict[str, Any] | None, out: dict[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    n_toys = None
    n_converged = None
    n_error = None

    if metrics:
        mm = metrics.get("metrics") or {}
        if isinstance(mm, dict):
            n_toys = mm.get("n_toys")
            n_converged = mm.get("n_converged")
            n_error = mm.get("n_error")

    if out:
        gen = out.get("gen") or {}
        res = out.get("results") or {}
        if n_toys is None:
            n_toys = gen.get("n_toys") or res.get("n_toys")
        if n_converged is None:
            conv = res.get("converged")
            if isinstance(conv, list):
                n_converged = sum(1 for x in conv if x)
            elif isinstance(res.get("n_converged"), int):
                n_converged = res.get("n_converged")
        if n_error is None and isinstance(res.get("n_error"), int):
            n_error = res.get("n_error")

    return n_toys, n_converged, n_error


def collect_case_rows(case_id: str, case_dir: Path) -> list[RunRow]:
    rows: list[RunRow] = []
    for meta_path_str in sorted(glob.glob(str(case_dir / "*.meta.json"))):
        meta_path = Path(meta_path_str)
        stem = meta_path.name[: -len(".meta.json")]

        meta = load_json(meta_path) or {}
        metrics = load_json(case_dir / f"{stem}.metrics.json")
        out = load_json(case_dir / f"{stem}.out.json")

        rc = int(meta.get("rc", 1))
        name = str(meta.get("name", stem))
        fit_mode = classify_case(name)

        n_toys, n_converged, n_error = extract_counts(metrics, out)

        wall_time_s = None
        pipeline = None
        if metrics:
            timing = metrics.get("timing") or {}
            toys = ((timing.get("breakdown") or {}).get("toys") or {})
            if isinstance(timing.get("wall_time_s"), (int, float)):
                wall_time_s = float(timing.get("wall_time_s"))
            pipeline = toys.get("pipeline")

        if wall_time_s is None:
            elapsed_s = meta.get("elapsed_s")
            if isinstance(elapsed_s, (int, float)):
                wall_time_s = float(elapsed_s)

        convergence_rate = None
        if isinstance(n_toys, int) and n_toys > 0 and isinstance(n_converged, int):
            convergence_rate = float(n_converged) / float(n_toys)

        toys_per_s = None
        if isinstance(n_toys, int) and n_toys > 0 and isinstance(wall_time_s, float) and wall_time_s > 0:
            toys_per_s = float(n_toys) / wall_time_s

        rows.append(
            RunRow(
                case_id=case_id,
                name=name,
                fit_mode=fit_mode,
                rc=rc,
                n_toys=n_toys,
                n_converged=n_converged,
                n_error=n_error,
                convergence_rate=convergence_rate,
                wall_time_s=wall_time_s,
                toys_per_s=toys_per_s,
                pipeline=pipeline,
            )
        )
    return rows


def gate_failures(rows: list[RunRow], gates: dict[str, Any]) -> list[str]:
    out: list[str] = []
    require_rc_zero = bool(gates.get("require_all_rc_zero", True))
    min_conv = float(gates.get("min_convergence_rate", 1.0))
    max_n_error = int(gates.get("max_n_error", 0))

    for r in rows:
        if require_rc_zero and r.rc != 0:
            out.append(f"{r.name}: rc={r.rc}")
        if r.n_error is not None and r.n_error > max_n_error:
            out.append(f"{r.name}: n_error={r.n_error} > {max_n_error}")
        if r.convergence_rate is not None and r.convergence_rate < min_conv:
            out.append(
                f"{r.name}: convergence_rate={r.convergence_rate:.6f} < {min_conv:.6f}"
            )
    return out


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# PF3.1 Publication Benchmark Summary")
    lines.append("")
    lines.append(f"Generated at: `{report['generated_at']}`")
    lines.append(f"Run root: `{report['run_root']}`")
    lines.append(f"Overall pass: `{report['overall_pass']}`")
    lines.append("")

    overall = report["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Cases: `{overall['n_cases']}`")
    lines.append(f"- Runs: `{overall['n_runs']}`")
    lines.append(f"- Runs with `rc!=0`: `{overall['n_rc_fail']}`")
    lines.append(f"- Runs with gate failures: `{overall['n_gate_failures']}`")
    lines.append("")

    if report["gate_failures"]:
        lines.append("## Gate Failures")
        lines.append("")
        for g in report["gate_failures"]:
            lines.append(f"- {g}")
        lines.append("")

    lines.append("## Case Details")
    lines.append("")
    for case in report["cases"]:
        lines.append(f"### `{case['id']}`")
        lines.append("")
        if case["missing"]:
            lines.append("No artifacts found for this case.")
            lines.append("")
            continue

        lines.append("| run | mode | rc | pipeline | toys | conv | n_error | wall_s | toys/s |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
        for r in case["rows"]:
            conv = "-"
            if r["n_converged"] is not None and r["n_toys"]:
                conv = f"{r['n_converged']}/{r['n_toys']}"
            lines.append(
                "| {name} | {fit_mode} | {rc} | {pipeline} | {n_toys} | {conv} | {n_error} | {wall} | {tps} |".format(
                    name=r["name"],
                    fit_mode=r["fit_mode"],
                    rc=r["rc"],
                    pipeline=r["pipeline"] or "-",
                    n_toys=r["n_toys"] if r["n_toys"] is not None else "-",
                    conv=conv,
                    n_error=r["n_error"] if r["n_error"] is not None else "-",
                    wall=format_float(r["wall_time_s"], 3),
                    tps=format_float(r["toys_per_s"], 3),
                )
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    cases = matrix.get("cases") or []
    gates = matrix.get("publication_gates") or {}

    case_reports: list[dict[str, Any]] = []
    all_rows: list[RunRow] = []

    for c in cases:
        cid = c["id"]
        cdir = run_root / cid
        rows = collect_case_rows(cid, cdir) if cdir.exists() else []
        rows_dict = [r.as_dict() for r in rows]
        case_reports.append(
            {
                "id": cid,
                "spec_rel": c.get("spec_rel"),
                "toys": c.get("toys"),
                "shards": c.get("shards"),
                "host_fit_modes": c.get("host_fit_modes"),
                "include_host_sharded": c.get("include_host_sharded"),
                "missing": len(rows) == 0,
                "rows": rows_dict,
            }
        )
        all_rows.extend(rows)

    failures = gate_failures(all_rows, gates)
    n_rc_fail = sum(1 for r in all_rows if r.rc != 0)

    report = {
        "schema_version": "nextstat.pf31_publication_summary.v1",
        "generated_at": utc_now_iso(),
        "run_root": str(run_root),
        "matrix_path": str(Path(args.matrix).resolve()),
        "gates": gates,
        "overall_pass": len(failures) == 0,
        "overall": {
            "n_cases": len(case_reports),
            "n_runs": len(all_rows),
            "n_rc_fail": n_rc_fail,
            "n_gate_failures": len(failures),
        },
        "gate_failures": failures,
        "cases": case_reports,
    }

    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, report)


if __name__ == "__main__":
    main()
