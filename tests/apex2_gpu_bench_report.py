#!/usr/bin/env python3
"""Apex2 GPU benchmark runner (CPU vs CUDA).

This is a *manual* perf harness intended to run on machines with CUDA/Metal.
It writes a machine-readable JSON report under `tmp/` and prints a concise
stdout summary.

Notes:
- CPU is always measured.
- CUDA is measured only when `nextstat._core.has_cuda()` is true.
- This script is not used by CI by default.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
import runpy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nextstat

from _apex2_json import write_report_json


def _load_gpu_tolerances() -> Tuple[float, float]:
    repo = Path(__file__).resolve().parents[1]
    ns = runpy.run_path(str(repo / "tests" / "python" / "_tolerances.py"))
    return float(ns["GPU_PARAM_ATOL"]), float(ns["GPU_FIT_NLL_ATOL"])


@dataclass(frozen=True)
class FitSummary:
    converged: bool
    nll: float
    parameters: List[float]
    n_evaluations: int


def _git_rev() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _read_workspace_json(path: Path) -> str:
    # Accept either a JSON string file or a Python dict dumped as JSON.
    raw = path.read_text(encoding="utf-8")
    # Validate JSON once so we fail early with a clear error message.
    json.loads(raw)
    return raw


def _build_model(workspace_json: str):
    return nextstat.HistFactoryModel.from_workspace(workspace_json)


def _fit(model, *, device: str) -> FitSummary:
    r = nextstat.fit(model, device=device)
    return FitSummary(
        converged=bool(r.converged),
        nll=float(r.nll),
        parameters=list(r.parameters),
        n_evaluations=int(getattr(r, "n_evaluations", getattr(r, "n_iter", 0))),
    )


def _max_abs_delta(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return max(abs(a[i] - b[i]) for i in range(n))


def _profile_scan(model, *, mu_values: List[float], device: str) -> Dict[str, Any]:
    # Returns dict with points list (mu, q_mu, nll_mu, ...).
    return nextstat.profile_scan(model, mu_values, device=device)


def _batch_toys(model, *, params: List[float], n_toys: int, seed: int, device: str) -> Any:
    core = nextstat._core
    return core.fit_toys(model, params, n_toys=n_toys, seed=seed, device=device)


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2 == 1:
        return float(ys[mid])
    return float(0.5 * (ys[mid - 1] + ys[mid]))


def _bench_repeated(label: str, repeats: int, fn) -> Tuple[Any, Dict[str, float]]:
    times: List[float] = []
    last = None
    for _ in range(max(1, repeats)):
        last, dt = _timed(fn)
        times.append(float(dt))
    return last, {"median_s": _median(times), "min_s": float(min(times)), "max_s": float(max(times))}

def _scan_parity(cpu_scan: Dict[str, Any], gpu_scan: Dict[str, Any], *, nll_atol: float) -> Dict[str, Any]:
    cpu_points = list(cpu_scan.get("points") or [])
    gpu_points = list(gpu_scan.get("points") or [])
    n = min(len(cpu_points), len(gpu_points))
    if n == 0:
        return {"ok": None, "reason": "no_points"}

    deltas_nll: List[float] = []
    deltas_qmu: List[float] = []
    for i in range(n):
        cp = cpu_points[i]
        gp = gpu_points[i]
        try:
            if float(cp.get("mu")) != float(gp.get("mu")):
                return {"ok": None, "reason": "mu_grid_mismatch"}
            if not (bool(cp.get("converged")) and bool(gp.get("converged"))):
                continue
            deltas_nll.append(abs(float(cp.get("nll_mu")) - float(gp.get("nll_mu"))))
            deltas_qmu.append(abs(float(cp.get("q_mu")) - float(gp.get("q_mu"))))
        except Exception:
            return {"ok": None, "reason": "bad_point_format"}

    if not deltas_nll:
        return {"ok": None, "reason": "no_converged_points"}

    max_abs_nll = max(deltas_nll)
    return {
        "ok": bool(max_abs_nll <= float(nll_atol)),
        "n_points_compared": int(len(deltas_nll)),
        "max_abs_delta_nll_mu": float(max_abs_nll),
        "nll_atol": float(nll_atol),
        "max_abs_delta_q_mu": float(max(deltas_qmu) if deltas_qmu else float("nan")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/apex2_gpu_bench_report.json"),
        help="JSON report output path",
    )
    ap.add_argument(
        "--cases",
        type=Path,
        nargs="*",
        default=[
            Path("tests/fixtures/simple_workspace.json"),
            Path("tests/fixtures/tchannel_workspace.json"),
        ],
        help="Workspace JSON files to benchmark",
    )
    ap.add_argument("--repeats", type=int, default=1, help="Repeats per operation (median reported)")
    ap.add_argument("--profile-points", type=int, default=21, help="Number of profile scan points")
    ap.add_argument(
        "--profile-mu-max",
        type=float,
        default=2.0,
        help="Max mu for profile scan grid (0..mu_max)",
    )
    ap.add_argument("--n-toys", type=int, default=256, help="Number of toys for batch toy fit bench")
    ap.add_argument("--skip-batch-toys", action="store_true", help="Skip batch toy fit benches")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed for toy generation")
    ap.add_argument("--no-cuda", action="store_true", help="Skip CUDA even if available")
    args = ap.parse_args()

    core = nextstat._core
    has_cuda = bool(getattr(core, "has_cuda", lambda: False)())
    if args.no_cuda:
        has_cuda = False

    gpu_param_atol, gpu_fit_nll_atol = _load_gpu_tolerances()

    report: Dict[str, Any] = {
        "meta": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "git_rev": _git_rev(),
            "has_cuda": bool(has_cuda),
        },
        "cases": [],
        "summary": {},
    }

    failures: List[Dict[str, Any]] = []

    for ws_path in args.cases:
        ws_path = Path(ws_path)
        workspace_json = _read_workspace_json(ws_path)

        model, t_build = _timed(lambda: _build_model(workspace_json))

        case_row: Dict[str, Any] = {
            "name": ws_path.name,
            "path": str(ws_path),
            "n_params": int(len(model.parameter_names())),
            "perf": {"timing_s": {"build_model": float(t_build)}},
            "cpu": {},
            "cuda": None,
            "parity": None,
        }

        # --- CPU ---
        cpu_fit, cpu_fit_t = _bench_repeated("cpu_fit", args.repeats, lambda: _fit(model, device="cpu"))
        case_row["cpu"]["fit"] = asdict(cpu_fit)
        case_row["perf"]["timing_s"]["fit_cpu_median"] = float(cpu_fit_t["median_s"])

        mu_max = float(args.profile_mu_max)
        n_pts = int(max(2, args.profile_points))
        mu_values = [mu_max * (i / float(n_pts - 1)) for i in range(n_pts)]

        cpu_scan, cpu_scan_t = _bench_repeated(
            "cpu_profile_scan",
            args.repeats,
            lambda: _profile_scan(model, mu_values=mu_values, device="cpu"),
        )
        case_row["cpu"]["profile_scan"] = {
            "poi_index": int(cpu_scan.get("poi_index", 0) or 0),
            "mu_hat": float(cpu_scan.get("mu_hat", float("nan"))),
            "nll_hat": float(cpu_scan.get("nll_hat", float("nan"))),
            "n_points": int(len(cpu_scan.get("points", []) or [])),
        }
        case_row["perf"]["timing_s"]["profile_scan_cpu_median"] = float(cpu_scan_t["median_s"])

        if not args.skip_batch_toys and int(args.n_toys) > 0:
            toys_cpu, toys_cpu_t = _bench_repeated(
                "cpu_fit_toys_batch",
                args.repeats,
                lambda: _batch_toys(
                    model,
                    params=cpu_fit.parameters,
                    n_toys=int(args.n_toys),
                    seed=int(args.seed),
                    device="cpu",
                ),
            )
            case_row["cpu"]["batch_toys"] = {
                "n_toys": int(len(toys_cpu)),
                "nll_mean": float(sum(r.nll for r in toys_cpu) / max(1, len(toys_cpu))),
            }
            case_row["perf"]["timing_s"]["batch_toys_cpu_median"] = float(toys_cpu_t["median_s"])

        # --- CUDA ---
        if has_cuda:
            cuda_errs: List[Dict[str, Any]] = []
            case_row["cuda"] = {}

            cuda_fit = None
            try:
                cuda_fit, cuda_fit_t = _bench_repeated(
                    "cuda_fit",
                    args.repeats,
                    lambda: _fit(model, device="cuda"),
                )
                case_row["cuda"]["fit"] = asdict(cuda_fit)
                case_row["perf"]["timing_s"]["fit_cuda_median"] = float(cuda_fit_t["median_s"])
            except BaseException as e:
                cuda_errs.append({"op": "fit", "error": f"{type(e).__name__}: {e}"})

            try:
                cuda_scan, cuda_scan_t = _bench_repeated(
                    "cuda_profile_scan",
                    args.repeats,
                    lambda: _profile_scan(model, mu_values=mu_values, device="cuda"),
                )
                case_row["cuda"]["profile_scan"] = {
                    "poi_index": int(cuda_scan.get("poi_index", 0) or 0),
                    "mu_hat": float(cuda_scan.get("mu_hat", float("nan"))),
                    "nll_hat": float(cuda_scan.get("nll_hat", float("nan"))),
                    "n_points": int(len(cuda_scan.get("points", []) or [])),
                }
                case_row["perf"]["timing_s"]["profile_scan_cuda_median"] = float(cuda_scan_t["median_s"])

                # Scan parity is useful even when the global fit does not converge.
                case_row["parity_scan"] = _scan_parity(cpu_scan, cuda_scan, nll_atol=gpu_fit_nll_atol)
                if case_row["parity_scan"].get("ok") is False:
                    failures.append(
                        {
                            "case": ws_path.name,
                            "reason": "gpu_parity_scan_out_of_tolerance",
                            "max_abs_delta_nll_mu": case_row["parity_scan"].get("max_abs_delta_nll_mu"),
                        }
                    )
            except BaseException as e:
                cuda_errs.append({"op": "profile_scan", "error": f"{type(e).__name__}: {e}"})

            if not args.skip_batch_toys and int(args.n_toys) > 0 and cuda_fit is not None:
                try:
                    toys_cuda, toys_cuda_t = _bench_repeated(
                        "cuda_fit_toys_batch",
                        args.repeats,
                        lambda: _batch_toys(
                            model,
                            params=cuda_fit.parameters,
                            n_toys=int(args.n_toys),
                            seed=int(args.seed),
                            device="cuda",
                        ),
                    )
                    case_row["cuda"]["batch_toys"] = {
                        "n_toys": int(len(toys_cuda)),
                        "nll_mean": float(sum(r.nll for r in toys_cuda) / max(1, len(toys_cuda))),
                    }
                    case_row["perf"]["timing_s"]["batch_toys_cuda_median"] = float(toys_cuda_t["median_s"])
                except BaseException as e:
                    cuda_errs.append({"op": "batch_toys", "error": f"{type(e).__name__}: {e}"})

            if cuda_fit is not None:
                # Basic parity checks (fit results only).
                max_abs_param = _max_abs_delta(cpu_fit.parameters, cuda_fit.parameters)
                abs_delta_nll = abs(cpu_fit.nll - cuda_fit.nll)
                parity_reason = None
                ok: Optional[bool] = None
                if not cpu_fit.converged:
                    parity_reason = "cpu_fit_not_converged"
                elif not cuda_fit.converged:
                    parity_reason = "cuda_fit_not_converged"
                else:
                    ok = (max_abs_param <= gpu_param_atol) and (abs_delta_nll <= gpu_fit_nll_atol)

                case_row["parity"] = {
                    "ok": ok,
                    "reason": parity_reason,
                    "max_abs_delta_param": float(max_abs_param),
                    "param_atol": float(gpu_param_atol),
                    "abs_delta_fit_nll": float(abs_delta_nll),
                    "fit_nll_atol": float(gpu_fit_nll_atol),
                    "cpu_converged": bool(cpu_fit.converged),
                    "cuda_converged": bool(cuda_fit.converged),
                }

                if "fit_cuda_median" in case_row["perf"]["timing_s"]:
                    case_row["perf"]["speedup"] = {"fit": float(case_row["perf"]["timing_s"]["fit_cpu_median"])
                        / max(float(case_row["perf"]["timing_s"]["fit_cuda_median"]), 1e-12)}
                    if "profile_scan_cuda_median" in case_row["perf"]["timing_s"]:
                        case_row["perf"]["speedup"]["profile_scan"] = float(case_row["perf"]["timing_s"]["profile_scan_cpu_median"]) / max(
                            float(case_row["perf"]["timing_s"]["profile_scan_cuda_median"]), 1e-12
                        )
                    if "batch_toys_cuda_median" in case_row["perf"]["timing_s"]:
                        case_row["perf"]["speedup"]["batch_toys"] = float(case_row["perf"]["timing_s"]["batch_toys_cpu_median"]) / max(
                            float(case_row["perf"]["timing_s"]["batch_toys_cuda_median"]), 1e-12
                        )

                if ok is False:
                    failures.append(
                        {
                            "case": ws_path.name,
                            "reason": "gpu_parity_fit_out_of_tolerance",
                            "max_abs_delta_param": float(max_abs_param),
                            "abs_delta_fit_nll": float(abs_delta_nll),
                        }
                    )

            if cuda_errs:
                case_row["cuda"]["errors"] = cuda_errs
                failures.append({"case": ws_path.name, "reason": "cuda_bench_error", "errors": cuda_errs})
        else:
            case_row["cuda"] = {"skipped": True, "reason": "cuda_not_available_or_disabled"}

        report["cases"].append(case_row)

    n_cases = len(report["cases"])
    n_cuda = sum(1 for c in report["cases"] if isinstance(c.get("cuda"), dict) and c["cuda"] and not c["cuda"].get("skipped"))
    n_parity_ok = sum(1 for c in report["cases"] if isinstance(c.get("parity"), dict) and c["parity"].get("ok"))
    report["summary"] = {
        "n_cases": int(n_cases),
        "n_cases_with_cuda": int(n_cuda),
        "n_parity_ok": int(n_parity_ok),
        "n_failed": int(len(failures)),
        "failures": failures,
    }

    write_report_json(args.out, report, deterministic=False)

    # stdout summary
    print("=" * 92)
    print("Apex2 report: CPU vs CUDA (NextStat)")
    print("=" * 92)
    print(f"cases: {n_cases} (cuda benches: {n_cuda})")
    if failures:
        print(f"parity failures: {len(failures)}")
    header = f"{'case':<28} {'params':>6} | {'fit speedup':>10} {'scan speedup':>11} {'toys speedup':>11} | {'parity':>6}"
    print(header)
    print("-" * len(header))
    for c in report["cases"]:
        speed = c.get("perf", {}).get("speedup", {}) if isinstance(c.get("perf"), dict) else {}
        parity = c.get("parity")
        ok = "n/a"
        if isinstance(parity, dict):
            if parity.get("ok") is True:
                ok = "OK"
            elif parity.get("ok") is False:
                ok = "FAIL"
            else:
                ok = "SKIP"
        print(
            f"{c['name']:<28} {c['n_params']:>6} | "
            f"{speed.get('fit', float('nan')):>10.2f}x {speed.get('profile_scan', float('nan')):>11.2f}x {speed.get('batch_toys', float('nan')):>11.2f}x | "
            f"{ok:>6}"
        )
    print(f"Wrote: {args.out}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
