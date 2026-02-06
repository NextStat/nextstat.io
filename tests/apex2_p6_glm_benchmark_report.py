#!/usr/bin/env python3
"""Apex2 runner: P6 GLM benchmarks (end-to-end fit/predict baselines).

This wraps `tests/benchmark_glm_fit_predict.py` and produces a machine-readable JSON report.
Optionally compares a run against an existing baseline JSON.

Run (generate report only):
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_p6_glm_benchmark_report.py \\
    --bench-out tmp/p6_glm_fit_predict.json \\
    --out tmp/apex2_p6_glm_bench_report.json

Run (compare to baseline):
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_p6_glm_benchmark_report.py \\
    --baseline tmp/p6_glm_fit_predict_baseline.json \\
    --bench-out tmp/p6_glm_fit_predict_current.json \\
    --out tmp/apex2_p6_glm_bench_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _with_py_path(env: Dict[str, str]) -> Dict[str, str]:
    repo = _repo_root()
    add = str(repo / "bindings" / "ns-py" / "python")
    cur = env.get("PYTHONPATH", "")
    if cur:
        if add in cur.split(os.pathsep):
            return env
        env["PYTHONPATH"] = cur + os.pathsep + add
    else:
        env["PYTHONPATH"] = add
    return env


def _run(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return p.returncode, p.stdout


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _case_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    return (str(row["model"]), int(row["n"]), int(row["p"]))


def _fmt_key(key: Tuple[str, int, int]) -> str:
    m, n, p = key
    return f"{m}:n={n}:p={p}"


def _iter_results(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    xs = report.get("results")
    if not isinstance(xs, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in xs:
        if isinstance(r, dict) and {"model", "n", "p", "fit_s", "predict_s"} <= set(r.keys()):
            out.append(r)
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not (v >= 0.0) or v != v:  # NaN check via v!=v
        return None
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_p6_glm_bench_report.json"))
    ap.add_argument(
        "--bench-out",
        type=Path,
        default=Path("tmp/p6_glm_fit_predict.json"),
        help="Where the underlying benchmark writes its JSON report.",
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON (same schema as --bench-out) to compare against.",
    )
    ap.add_argument(
        "--max-slowdown",
        type=float,
        default=1.30,
        help="Allowed max slowdown ratio (current/baseline) for both fit and predict.",
    )
    ap.add_argument(
        "--max-slowdown-fit",
        type=float,
        default=None,
        help="Allowed max slowdown ratio for fit only (overrides --max-slowdown).",
    )
    ap.add_argument(
        "--max-slowdown-predict",
        type=float,
        default=None,
        help="Allowed max slowdown ratio for predict only (overrides --max-slowdown).",
    )
    ap.add_argument(
        "--min-baseline-fit-s",
        type=float,
        default=1e-3,
        help="Skip fit-time comparisons when baseline fit_s is below this (too noisy at sub-ms scale).",
    )
    ap.add_argument("--sizes", default="200,2000,20000", help="Comma-separated n values.")
    ap.add_argument("--p", type=int, default=20, help="Feature count (without intercept).")
    ap.add_argument("--l2", type=float, default=0.0, help="Optional ridge (0 disables).")
    ap.add_argument("--nb-alpha", type=float, default=0.5, help="NegBin alpha for synthetic data.")
    args = ap.parse_args()

    repo = _repo_root()
    env = _with_py_path(os.environ.copy())

    t0 = time.time()
    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(t0),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "params": {
                "sizes": str(args.sizes),
                "p": int(args.p),
                "l2": float(args.l2),
                "nb_alpha": float(args.nb_alpha),
            },
            "thresholds": {
                "max_slowdown": float(args.max_slowdown),
                "max_slowdown_fit": float(args.max_slowdown_fit)
                if args.max_slowdown_fit is not None
                else None,
                "max_slowdown_predict": float(args.max_slowdown_predict)
                if args.max_slowdown_predict is not None
                else None,
                "min_baseline_fit_s": float(args.min_baseline_fit_s),
            },
        },
        "status": None,
        "bench": None,
        "compare": None,
        "summary": None,
    }

    bench = repo / "tests" / "benchmark_glm_fit_predict.py"
    bench_cmd = [
        sys.executable,
        str(bench),
        "--sizes",
        str(args.sizes),
        "--p",
        str(int(args.p)),
        "--l2",
        str(float(args.l2)),
        "--nb-alpha",
        str(float(args.nb_alpha)),
        "--out",
        str(args.bench_out),
    ]

    rc, out = _run(bench_cmd, cwd=repo, env=env)
    bench_report = _read_json(args.bench_out) if args.bench_out.exists() else None
    report["bench"] = {
        "returncode": int(rc),
        "stdout_tail": out[-4000:],
        "bench_report_path": str(args.bench_out),
        "bench_report": bench_report,
    }

    if rc != 0 or not isinstance(bench_report, dict):
        report["status"] = "error"
        report["summary"] = {"reason": "benchmark_failed"}
        report["meta"]["wall_s"] = float(time.time() - t0)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote: {args.out}")
        return 4

    # ------------------------------------------------------------------
    # Optional compare to baseline
    # ------------------------------------------------------------------
    if args.baseline is None:
        report["status"] = "ok"
        report["compare"] = None
        report["summary"] = {"n_cases": len(list(_iter_results(bench_report)))}
        report["meta"]["wall_s"] = float(time.time() - t0)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote: {args.out}")
        return 0

    if not args.baseline.exists():
        report["status"] = "error"
        report["compare"] = {"baseline_path": str(args.baseline), "reason": "baseline_missing"}
        report["summary"] = {"reason": "baseline_missing"}
        report["meta"]["wall_s"] = float(time.time() - t0)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote: {args.out}")
        return 4

    baseline_report = _read_json(args.baseline)
    baseline_rows = { _case_key(r): r for r in _iter_results(baseline_report) }
    current_rows = { _case_key(r): r for r in _iter_results(bench_report) }

    max_fit = float(args.max_slowdown_fit) if args.max_slowdown_fit is not None else float(args.max_slowdown)
    max_pred = float(args.max_slowdown_predict) if args.max_slowdown_predict is not None else float(args.max_slowdown)
    min_fit = float(args.min_baseline_fit_s)

    cases: List[Dict[str, Any]] = []
    any_failed = False
    max_slow_fit = 0.0
    max_slow_pred = 0.0

    for key, cur in sorted(current_rows.items(), key=lambda kv: _fmt_key(kv[0])):
        base = baseline_rows.get(key)
        row: Dict[str, Any] = {"key": _fmt_key(key)}
        if base is None:
            any_failed = True
            row.update({"ok": False, "reason": "missing_baseline_case"})
            cases.append(row)
            continue

        b_fit = _safe_float(base.get("fit_s"))
        b_pred = _safe_float(base.get("predict_s"))
        c_fit = _safe_float(cur.get("fit_s"))
        c_pred = _safe_float(cur.get("predict_s"))

        if b_fit is None or b_pred is None or c_fit is None or c_pred is None or b_fit <= 0.0 or b_pred <= 0.0:
            any_failed = True
            row.update({"ok": False, "reason": "invalid_timings"})
            row["baseline"] = {"fit_s": base.get("fit_s"), "predict_s": base.get("predict_s")}
            row["current"] = {"fit_s": cur.get("fit_s"), "predict_s": cur.get("predict_s")}
            cases.append(row)
            continue

        slow_fit = c_fit / b_fit
        slow_pred = c_pred / b_pred
        max_slow_fit = max(max_slow_fit, slow_fit)
        max_slow_pred = max(max_slow_pred, slow_pred)

        fit_checked = b_fit >= min_fit
        ok_fit = (slow_fit <= max_fit) if fit_checked else True
        ok_pred = (slow_pred <= max_pred)
        ok = ok_fit and ok_pred
        if not ok:
            any_failed = True
        row.update(
            {
                "ok": bool(ok),
                "baseline": {"fit_s": float(b_fit), "predict_s": float(b_pred)},
                "current": {"fit_s": float(c_fit), "predict_s": float(c_pred)},
                "slowdown": {"fit": float(slow_fit), "predict": float(slow_pred)},
                "thresholds": {"fit": float(max_fit), "predict": float(max_pred)},
                "checks": {"fit_checked": bool(fit_checked), "predict_checked": True},
            }
        )
        cases.append(row)

    # Surface cases present only in baseline as warnings (not failures).
    baseline_only = sorted(set(baseline_rows.keys()) - set(current_rows.keys()), key=_fmt_key)
    compare: Dict[str, Any] = {
        "baseline_path": str(args.baseline),
        "baseline_meta": (baseline_report.get("meta") if isinstance(baseline_report, dict) else None),
        "cases": cases,
        "baseline_only": [_fmt_key(k) for k in baseline_only],
    }
    report["compare"] = compare
    report["status"] = "ok" if not any_failed else "fail"
    report["summary"] = {
        "n_cases": int(len(cases)),
        "n_ok": int(sum(1 for c in cases if c.get("ok") is True)),
        "n_fail": int(sum(1 for c in cases if c.get("ok") is False)),
        "max_slowdown_fit": float(max_slow_fit),
        "max_slowdown_predict": float(max_slow_pred),
    }

    report["meta"]["wall_s"] = float(time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {args.out}")
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
