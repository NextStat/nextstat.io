#!/usr/bin/env python3
"""Apex2 master runner: one report for pyhf parity + regression golden + P6 benches + ROOT parity.

This script combines existing Apex2 runners into a single JSON artifact:
  - pyhf: `tests/apex2_pyhf_validation_report.py`
  - regression golden: `tests/fixtures/regression/*.json` via `nextstat.glm.*`
  - P6 benchmarks (GLM fit/predict): `tests/apex2_p6_glm_benchmark_report.py` (optional)
  - bias/pulls: `tests/apex2_bias_pulls_report.py` (optional; slow)
  - SBC (NUTS): `tests/apex2_sbc_report.py` (optional; slow)
  - ROOT: `tests/apex2_root_suite_report.py` (runs if prereqs exist, else records skipped)

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py
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
from typing import Any, Dict, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _with_py_path(env: Dict[str, str]) -> Dict[str, str]:
    # Ensure the editable python package is importable for subprocess calls.
    # Prefer to preserve existing PYTHONPATH order.
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


def _run_json(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return p.returncode, p.stdout


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _max_abs_vec_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def _vec_allclose(a: list[float], b: list[float], *, rtol: float, atol: float = 0.0) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        x = float(x)
        y = float(y)
        scale = max(abs(x), abs(y), 1.0)
        if abs(x - y) > (float(atol) + float(rtol) * scale):
            return False
    return True


def _run_regression_golden() -> Dict[str, Any]:
    repo = _repo_root()
    fx_dir = repo / "tests" / "fixtures" / "regression"
    cases: list[Dict[str, Any]] = []

    try:
        import nextstat  # type: ignore
    except ModuleNotFoundError as e:
        return {"status": "skipped", "reason": f"import_nextstat_failed:{e}"}

    paths = sorted(fx_dir.glob("*.json"))
    if not paths:
        return {"status": "error", "reason": f"no_fixtures_found:{fx_dir}"}

    any_failed = False
    for path in paths:
        data = json.loads(path.read_text())
        kind = str(data.get("kind"))
        name = str(data.get("name") or path.stem)
        include_intercept = bool(data.get("include_intercept", True))
        x = data["x"]
        y = data["y"]

        row: Dict[str, Any] = {"name": name, "kind": kind, "path": str(path)}

        try:
            if kind == "ols":
                r = nextstat.glm.linear.fit(x, y, include_intercept=include_intercept)
                ok_coef = _vec_allclose(list(r.coef), list(data["beta_hat"]), rtol=1e-10, atol=1e-12)
                ok_se = _vec_allclose(
                    list(r.standard_errors), list(data["se_hat"]), rtol=1e-10, atol=1e-12
                )
                # Recompute NLL at hat (SSE/2).
                yhat = r.predict(x)
                nll = 0.5 * sum((float(a) - float(b)) ** 2 for a, b in zip(yhat, y))
                ok_nll = abs(float(nll) - float(data["nll_at_hat"])) <= 1e-8
                row.update(
                    {
                        "ok": bool(ok_coef and ok_se and ok_nll),
                        "max_abs_coef_diff": _max_abs_vec_diff(list(r.coef), list(data["beta_hat"])),
                        "max_abs_se_diff": _max_abs_vec_diff(
                            list(r.standard_errors), list(data["se_hat"])
                        ),
                        "abs_nll_diff": abs(float(nll) - float(data["nll_at_hat"])),
                    }
                )
            elif kind == "logistic":
                y_int = [1 if float(v) >= 0.5 else 0 for v in y]
                r = nextstat.glm.logistic.fit(x, y_int, include_intercept=include_intercept)
                ok_coef = _vec_allclose(list(r.coef), list(data["beta_hat"]), rtol=2e-3, atol=1e-6)
                ok_se = _vec_allclose(
                    list(r.standard_errors), list(data["se_hat"]), rtol=2e-2, atol=1e-6
                )
                ok_nll = abs(float(r.nll) - float(data["nll_at_hat"])) <= 1e-6
                row.update(
                    {
                        "ok": bool(ok_coef and ok_se and ok_nll and bool(r.converged)),
                        "converged": bool(r.converged),
                        "max_abs_coef_diff": _max_abs_vec_diff(list(r.coef), list(data["beta_hat"])),
                        "max_abs_se_diff": _max_abs_vec_diff(
                            list(r.standard_errors), list(data["se_hat"])
                        ),
                        "abs_nll_diff": abs(float(r.nll) - float(data["nll_at_hat"])),
                    }
                )
            elif kind == "poisson":
                y_int = [int(round(float(v))) for v in y]
                r = nextstat.glm.poisson.fit(x, y_int, include_intercept=include_intercept)
                ok_coef = _vec_allclose(list(r.coef), list(data["beta_hat"]), rtol=2e-3, atol=1e-6)
                ok_se = _vec_allclose(
                    list(r.standard_errors), list(data["se_hat"]), rtol=2e-2, atol=1e-6
                )
                ok_nll = abs(float(r.nll) - float(data["nll_at_hat"])) <= 1e-6
                row.update(
                    {
                        "ok": bool(ok_coef and ok_se and ok_nll and bool(r.converged)),
                        "converged": bool(r.converged),
                        "max_abs_coef_diff": _max_abs_vec_diff(list(r.coef), list(data["beta_hat"])),
                        "max_abs_se_diff": _max_abs_vec_diff(
                            list(r.standard_errors), list(data["se_hat"])
                        ),
                        "abs_nll_diff": abs(float(r.nll) - float(data["nll_at_hat"])),
                    }
                )
            elif kind == "negbin":
                # Avoid hard-failing the master report when the runtime doesn't yet expose the model.
                # (pytest should still catch missing bindings when/if the model is expected.)
                if not hasattr(nextstat._core, "NegativeBinomialRegressionModel"):  # type: ignore[attr-defined]
                    row.update({"ok": False, "skipped": True, "reason": "missing_binding:NegativeBinomialRegressionModel"})
                    cases.append(row)
                    continue
                y_int = [int(round(float(v))) for v in y]
                r = nextstat.glm.negbin.fit(x, y_int, include_intercept=include_intercept)
                ok_coef = _vec_allclose(list(r.coef), list(data["beta_hat"]), rtol=2e-3, atol=1e-6)
                ok_se = _vec_allclose(
                    list(r.standard_errors), list(data["se_hat"]), rtol=2e-2, atol=1e-6
                )
                ok_log_alpha = abs(float(r.log_alpha) - float(data["log_alpha_hat"])) <= 2e-2
                ok_nll = abs(float(r.nll) - float(data["nll_at_hat"])) <= 1e-6
                row.update(
                    {
                        "ok": bool(
                            ok_coef and ok_se and ok_log_alpha and ok_nll and bool(r.converged)
                        ),
                        "converged": bool(r.converged),
                        "max_abs_coef_diff": _max_abs_vec_diff(list(r.coef), list(data["beta_hat"])),
                        "max_abs_se_diff": _max_abs_vec_diff(
                            list(r.standard_errors), list(data["se_hat"])
                        ),
                        "abs_log_alpha_diff": abs(float(r.log_alpha) - float(data["log_alpha_hat"])),
                        "abs_alpha_diff": abs(float(r.alpha) - float(data["alpha_hat"])),
                        "abs_nll_diff": abs(float(r.nll) - float(data["nll_at_hat"])),
                    }
                )
            else:
                row.update({"ok": False, "skipped": True, "reason": f"unknown_kind:{kind}"})
        except Exception as e:
            row.update({"ok": False, "reason": f"exception:{type(e).__name__}:{e}"})

        if row.get("skipped"):
            cases.append(row)
            continue
        if not row.get("ok"):
            any_failed = True
        cases.append(row)

    n_ok = sum(1 for c in cases if c.get("ok") is True)
    n_skipped = sum(1 for c in cases if c.get("skipped") is True)
    out: Dict[str, Any] = {
        "status": "ok" if not any_failed else "fail",
        "summary": {
            "n_cases": len(cases),
            "n_ok": int(n_ok),
            "n_failed": int(len(cases) - n_ok - n_skipped),
            "n_skipped": int(n_skipped),
        },
        "cases": cases,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_master_report.json"))
    ap.add_argument("--pyhf-out", type=Path, default=Path("tmp/apex2_pyhf_report.json"))
    ap.add_argument(
        "--p6-glm-bench-out",
        type=Path,
        default=Path("tmp/p6_glm_fit_predict.json"),
        help="Underlying P6 GLM benchmark JSON output (from tests/benchmark_glm_fit_predict.py).",
    )
    ap.add_argument(
        "--p6-glm-bench-report-out",
        type=Path,
        default=Path("tmp/apex2_p6_glm_bench_report.json"),
        help="Apex2 P6 GLM benchmark report JSON output.",
    )
    ap.add_argument("--bias-pulls-out", type=Path, default=Path("tmp/apex2_bias_pulls_report.json"))
    ap.add_argument("--sbc-out", type=Path, default=Path("tmp/apex2_sbc_report.json"))
    ap.add_argument("--root-out", type=Path, default=Path("tmp/apex2_root_suite_report.json"))
    ap.add_argument("--root-cases", type=Path, default=None, help="Cases JSON for ROOT suite.")
    ap.add_argument(
        "--root-search-dir",
        type=Path,
        default=None,
        help="Auto-discover TRExFitter/HistFactory exports by scanning for combination.xml under this directory.",
    )
    ap.add_argument(
        "--root-glob",
        type=str,
        default="**/combination.xml",
        help="Glob relative to --root-search-dir (used only when auto-generating cases).",
    )
    ap.add_argument(
        "--root-cases-out",
        type=Path,
        default=Path("tmp/apex2_root_cases.json"),
        help="Where to write auto-generated ROOT cases JSON (used only with --root-search-dir).",
    )
    ap.add_argument(
        "--root-cases-absolute-paths",
        action="store_true",
        help="Write absolute paths in auto-generated ROOT cases JSON.",
    )
    ap.add_argument(
        "--root-include-fixtures",
        action="store_true",
        help="Include built-in fixture case(s) in auto-generated ROOT cases JSON.",
    )
    ap.add_argument("--root-mu-start", type=float, default=0.0)
    ap.add_argument("--root-mu-stop", type=float, default=5.0)
    ap.add_argument("--root-mu-points", type=int, default=51)
    ap.add_argument("--pyhf-sizes", type=str, default="2,16,64,256")
    ap.add_argument("--pyhf-n-random", type=int, default=8)
    ap.add_argument("--pyhf-seed", type=int, default=0)
    ap.add_argument("--pyhf-fit", action="store_true")
    ap.add_argument(
        "--p6-glm-bench",
        action="store_true",
        help="Also run P6 GLM end-to-end fit/predict benchmarks and embed report.",
    )
    ap.add_argument(
        "--p6-glm-bench-baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON to compare against (same schema as --p6-glm-bench-out).",
    )
    ap.add_argument("--p6-glm-bench-max-slowdown", type=float, default=1.30)
    ap.add_argument("--p6-glm-bench-min-baseline-fit-s", type=float, default=1e-3)
    ap.add_argument("--p6-glm-bench-sizes", type=str, default="200,2000,20000")
    ap.add_argument("--p6-glm-bench-p", type=int, default=20)
    ap.add_argument("--p6-glm-bench-l2", type=float, default=0.0)
    ap.add_argument("--p6-glm-bench-nb-alpha", type=float, default=0.5)
    ap.add_argument(
        "--bias-pulls",
        action="store_true",
        help="Also run slow bias/pulls regression (NextStat vs pyhf) and embed report.",
    )
    ap.add_argument("--bias-pulls-n-toys", type=int, default=200)
    ap.add_argument("--bias-pulls-seed", type=int, default=0)
    ap.add_argument("--bias-pulls-fixtures", type=str, default="simple")
    ap.add_argument(
        "--sbc",
        action="store_true",
        help="Also run slow SBC (NUTS) report and embed report (requires NS_RUN_SLOW=1).",
    )
    ap.add_argument("--sbc-cases", type=str, default="lin1d,lin2d")
    ap.add_argument("--sbc-n-runs", type=int, default=None)
    ap.add_argument("--sbc-warmup", type=int, default=None)
    ap.add_argument("--sbc-samples", type=int, default=None)
    ap.add_argument("--sbc-seed", type=int, default=None)
    ap.add_argument("--sbc-rhat-max", type=float, default=1.40)
    ap.add_argument("--sbc-divergence-rate-max", type=float, default=0.05)
    ap.add_argument("--root-prereq-only", action="store_true", help="Only check ROOT prereqs.")
    args = ap.parse_args()

    repo = _repo_root()
    cwd = repo
    env = _with_py_path(os.environ.copy())

    t0 = time.time()
    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(t0),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "pyhf": None,
        "regression_golden": None,
        "p6_glm_bench": None,
        "bias_pulls": None,
        "sbc": None,
        "root": None,
    }

    # ------------------------------------------------------------------
    # pyhf runner (always runnable if pyhf installed)
    # ------------------------------------------------------------------
    pyhf_runner = repo / "tests" / "apex2_pyhf_validation_report.py"
    pyhf_cmd = [
        sys.executable,
        str(pyhf_runner),
        "--out",
        str(args.pyhf_out),
        "--sizes",
        args.pyhf_sizes,
        "--n-random",
        str(args.pyhf_n_random),
        "--seed",
        str(args.pyhf_seed),
    ]
    if args.pyhf_fit:
        pyhf_cmd.append("--fit")

    rc_pyhf, out_pyhf = _run_json(pyhf_cmd, cwd=cwd, env=env)
    report["pyhf"] = {
        "status": "ok" if rc_pyhf == 0 else "fail",
        "returncode": int(rc_pyhf),
        "stdout_tail": out_pyhf[-4000:],
        "report_path": str(args.pyhf_out),
        "report": _read_json(args.pyhf_out) if args.pyhf_out.exists() else None,
    }

    # ------------------------------------------------------------------
    # Regression golden fixtures (GLM surface)
    # ------------------------------------------------------------------
    report["regression_golden"] = _run_regression_golden()

    # ------------------------------------------------------------------
    # P6 benchmarks (optional)
    # ------------------------------------------------------------------
    if args.p6_glm_bench:
        p6_runner = repo / "tests" / "apex2_p6_glm_benchmark_report.py"
        p6_cmd = [
            sys.executable,
            str(p6_runner),
            "--out",
            str(args.p6_glm_bench_report_out),
            "--bench-out",
            str(args.p6_glm_bench_out),
            "--sizes",
            str(args.p6_glm_bench_sizes),
            "--p",
            str(int(args.p6_glm_bench_p)),
            "--l2",
            str(float(args.p6_glm_bench_l2)),
            "--nb-alpha",
            str(float(args.p6_glm_bench_nb_alpha)),
            "--max-slowdown",
            str(float(args.p6_glm_bench_max_slowdown)),
            "--min-baseline-fit-s",
            str(float(args.p6_glm_bench_min_baseline_fit_s)),
        ]
        if args.p6_glm_bench_baseline is not None:
            p6_cmd += ["--baseline", str(args.p6_glm_bench_baseline)]

        rc_p6, out_p6 = _run_json(p6_cmd, cwd=cwd, env=env)
        p6_report = _read_json(args.p6_glm_bench_report_out) if args.p6_glm_bench_report_out.exists() else None
        p6_declared = (p6_report or {}).get("status") if isinstance(p6_report, dict) else None
        if p6_declared in ("ok", "fail", "skipped", "error"):
            status = str(p6_declared)
        else:
            status = "ok" if rc_p6 == 0 else "fail"
        report["p6_glm_bench"] = {
            "status": status,
            "returncode": int(rc_p6),
            "stdout_tail": out_p6[-4000:],
            "report_path": str(args.p6_glm_bench_report_out),
            "bench_report_path": str(args.p6_glm_bench_out),
            "baseline_path": str(args.p6_glm_bench_baseline) if args.p6_glm_bench_baseline else None,
            "report": p6_report,
        }
    else:
        report["p6_glm_bench"] = {"status": "skipped", "reason": "not_requested"}

    # ------------------------------------------------------------------
    # Bias/pulls regression (optional; slow)
    # ------------------------------------------------------------------
    if args.bias_pulls:
        bias_runner = repo / "tests" / "apex2_bias_pulls_report.py"
        bias_cmd = [
            sys.executable,
            str(bias_runner),
            "--out",
            str(args.bias_pulls_out),
            "--n-toys",
            str(args.bias_pulls_n_toys),
            "--seed",
            str(args.bias_pulls_seed),
            "--fixtures",
            str(args.bias_pulls_fixtures),
        ]
        rc_bias, out_bias = _run_json(bias_cmd, cwd=cwd, env=env)
        report["bias_pulls"] = {
            "status": "ok" if rc_bias == 0 else "fail",
            "returncode": int(rc_bias),
            "stdout_tail": out_bias[-4000:],
            "report_path": str(args.bias_pulls_out),
            "report": _read_json(args.bias_pulls_out) if args.bias_pulls_out.exists() else None,
        }
    else:
        report["bias_pulls"] = {"status": "skipped", "reason": "not_requested"}

    # ------------------------------------------------------------------
    # SBC report (optional; slow)
    # ------------------------------------------------------------------
    if args.sbc:
        sbc_runner = repo / "tests" / "apex2_sbc_report.py"
        sbc_cmd = [
            sys.executable,
            str(sbc_runner),
            "--out",
            str(args.sbc_out),
            "--cases",
            str(args.sbc_cases),
        ]
        if args.sbc_n_runs is not None:
            sbc_cmd += ["--n-runs", str(args.sbc_n_runs)]
        if args.sbc_warmup is not None:
            sbc_cmd += ["--warmup", str(args.sbc_warmup)]
        if args.sbc_samples is not None:
            sbc_cmd += ["--samples", str(args.sbc_samples)]
        if args.sbc_seed is not None:
            sbc_cmd += ["--seed", str(args.sbc_seed)]
        sbc_cmd += ["--rhat-max", str(args.sbc_rhat_max)]
        sbc_cmd += ["--divergence-rate-max", str(args.sbc_divergence_rate_max)]
        rc_sbc, out_sbc = _run_json(sbc_cmd, cwd=cwd, env=env)
        sbc_report = _read_json(args.sbc_out) if args.sbc_out.exists() else None
        # Prefer the report's declared status to avoid "ok but skipped" confusion.
        sbc_declared = (sbc_report or {}).get("status")
        if sbc_declared in ("ok", "fail", "skipped", "error"):
            status = str(sbc_declared)
        else:
            status = "ok" if rc_sbc == 0 else "fail"
        report["sbc"] = {
            "status": status,
            "returncode": int(rc_sbc),
            "stdout_tail": out_sbc[-4000:],
            "report_path": str(args.sbc_out),
            "report": sbc_report,
        }
    else:
        report["sbc"] = {"status": "skipped", "reason": "not_requested"}

    # ------------------------------------------------------------------
    # ROOT suite runner (may be skipped)
    # ------------------------------------------------------------------
    root_cases_used = args.root_cases
    root_cases_generation: Optional[Dict[str, Any]] = None
    if root_cases_used is None and args.root_search_dir is not None:
        gen = repo / "tests" / "generate_apex2_root_cases.py"
        root_cases_used = args.root_cases_out
        gen_cmd = [
            sys.executable,
            str(gen),
            "--search-dir",
            str(args.root_search_dir),
            "--glob",
            args.root_glob,
            "--out",
            str(root_cases_used),
            "--start",
            str(args.root_mu_start),
            "--stop",
            str(args.root_mu_stop),
            "--points",
            str(args.root_mu_points),
        ]
        if args.root_include_fixtures:
            gen_cmd.append("--include-fixtures")
        if args.root_cases_absolute_paths:
            gen_cmd.append("--absolute-paths")

        rc_gen, out_gen = _run_json(gen_cmd, cwd=cwd, env=env)
        root_cases_generation = {
            "returncode": int(rc_gen),
            "stdout_tail": out_gen[-4000:],
            "cases_path": str(root_cases_used),
        }
        if rc_gen != 0:
            report["root"] = {
                "status": "error",
                "reason": "case_generation_failed",
                "cases_generation": root_cases_generation,
            }
            report["meta"]["wall_s"] = float(time.time() - t0)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2))
            print(f"Wrote: {args.out}")
            return 2

    root_runner = repo / "tests" / "apex2_root_suite_report.py"
    root_cmd = [
        sys.executable,
        str(root_runner),
        "--out",
        str(args.root_out),
    ]
    if root_cases_used is not None:
        root_cmd += ["--cases", str(root_cases_used)]
    if args.root_prereq_only:
        root_cmd.append("--prereq-only")

    rc_root, out_root = _run_json(root_cmd, cwd=cwd, env=env)
    root_report = _read_json(args.root_out) if args.root_out.exists() else None
    prereqs = (root_report or {}).get("meta", {}).get("prereqs") if root_report else None
    prereqs_ok = (
        isinstance(prereqs, dict)
        and bool(prereqs.get("hist2workspace"))
        and bool(prereqs.get("root"))
        and (prereqs.get("uproot") is not False)
    )
    if root_report and not prereqs_ok:
        root_status = "skipped"
    elif rc_root == 0:
        root_status = "ok"
    else:
        root_status = "fail" if root_report else "error"

    report["root"] = {
        "status": root_status,
        "returncode": int(rc_root),
        "stdout_tail": out_root[-4000:],
        "report_path": str(args.root_out),
        "cases_path_used": str(root_cases_used) if root_cases_used is not None else None,
        "cases_generation": root_cases_generation,
        "report": root_report,
        "prereqs": prereqs,
    }

    report["meta"]["wall_s"] = float(time.time() - t0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Exit code policy: fail on pyhf mismatch; fail on ROOT mismatch only if prereqs exist.
    pyhf_ok = (rc_pyhf == 0)
    reg_ok_or_skipped = report["regression_golden"]["status"] in ("ok", "skipped")
    p6_ok_or_skipped = report["p6_glm_bench"]["status"] in ("ok", "skipped")
    bias_ok_or_skipped = report["bias_pulls"]["status"] in ("ok", "skipped")
    sbc_ok_or_skipped = report["sbc"]["status"] in ("ok", "skipped")
    root_ok_or_skipped = root_status in ("ok", "skipped")

    print(f"Wrote: {args.out}")
    if not pyhf_ok:
        return 2
    if not reg_ok_or_skipped:
        return 2
    if not p6_ok_or_skipped:
        return 2
    if not bias_ok_or_skipped:
        return 2
    if not sbc_ok_or_skipped:
        return 2
    if not root_ok_or_skipped:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
