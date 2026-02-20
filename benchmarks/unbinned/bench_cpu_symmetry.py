#!/usr/bin/env python3
"""
Symmetric CPU benchmark for NextStat vs MoreFit (gauss_exp only).

Why this script exists:
- separates wall-to-wall timing (external process wall clock)
- separates warm in-process fit timing (compute-focused)
- runs deterministic datasets (same seeds, same N) for both frameworks

Output:
- JSON artifact with schema `nextstat.unbinned_cpu_symmetry_bench.v1`
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/benchmarks"))
from run_suite import _case_gauss_exp, _write_parquet  # noqa: E402
from _parse_utils import parse_json_stdout  # noqa: E402


def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(msg)


def _parse_csv_ints(s: str) -> List[int]:
    vals: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        _die(f"invalid integer list: {s!r}")
    return vals


def _find_nextstat_cli() -> Path:
    env = os.environ.get("NS_CLI_BIN")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
        _die(f"NS_CLI_BIN points to missing file: {p}")
    candidate = REPO_ROOT / "target" / "release" / "nextstat"
    if candidate.is_file():
        return candidate.resolve()
    _die("nextstat CLI not found. Set NS_CLI_BIN or build target/release/nextstat")


def _find_morefit_variants(args: argparse.Namespace) -> Dict[str, Path]:
    candidates = {
        "mf_1t_num": args.morefit_1t_num,
        "mf_1t_agrad": args.morefit_1t_agrad,
        "mf_20t_num": args.morefit_20t_num,
        "mf_20t_agrad": args.morefit_20t_agrad,
    }
    resolved: Dict[str, Path] = {}
    for key, value in candidates.items():
        p = Path(value).expanduser().resolve()
        if not p.is_file():
            _die(f"missing MoreFit binary for {key}: {p}")
        resolved[key] = p
    return resolved


def _parse_json_tail(stdout: str) -> Dict[str, Any]:
    s = stdout.strip()
    idx = s.rfind("\n{")
    if idx >= 0:
        idx += 1
    else:
        idx = s.rfind("{")
    if idx < 0:
        raise RuntimeError("no JSON object found in stdout")
    return json.loads(s[idx:])


def _run_nextstat_cli(
    nextstat_bin: Path,
    spec_path: Path,
    threads: int,
    opt_max_iter: int | None,
    opt_tol: float | None,
    opt_m: int | None,
    opt_smooth_bounds: bool,
) -> Tuple[float, Dict[str, Any]]:
    cmd = [
        str(nextstat_bin),
        "unbinned-fit",
        "--threads",
        str(threads),
        "--config",
        str(spec_path),
    ]
    if opt_max_iter is not None:
        cmd.extend(["--opt-max-iter", str(opt_max_iter)])
    if opt_tol is not None:
        cmd.extend(["--opt-tol", str(opt_tol)])
    if opt_m is not None:
        cmd.extend(["--opt-m", str(opt_m)])
    if opt_smooth_bounds:
        cmd.append("--opt-smooth-bounds")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(f"nextstat failed (rc={proc.returncode}): {proc.stderr[-500:]}")
    return wall_ms, parse_json_stdout(proc.stdout)


def _run_nextstat_library(
    spec_path: Path,
    warmup: int,
    repeats: int,
    threads: int,
    opt_max_iter: int | None,
    opt_tol: float | None,
    opt_m: int | None,
    opt_smooth_bounds: bool,
) -> Dict[str, Any]:
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    import nextstat  # type: ignore

    model = nextstat.UnbinnedModel.from_config(str(spec_path))
    mle = nextstat.MaximumLikelihoodEstimator(
        max_iter=1000 if opt_max_iter is None else int(opt_max_iter),
        tol=1e-6 if opt_tol is None else float(opt_tol),
        m=10 if opt_m is None else int(opt_m),
        smooth_bounds=bool(opt_smooth_bounds),
    )
    for _ in range(warmup):
        mle.fit(model)

    times_ms: List[float] = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = mle.fit(model)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "times_ms": times_ms,
        "min_ms": float(np.min(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "mean_ms": float(np.mean(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "nll": float(last.nll),
        "n_iter": int(last.n_iter),
        "n_fev": int(getattr(last, "n_fev", 0)),
        "n_gev": int(getattr(last, "n_gev", 0)),
        "termination_reason": str(getattr(last, "termination_reason", "")),
        "final_grad_norm": float(getattr(last, "final_grad_norm", float("nan"))),
        "initial_nll": float(getattr(last, "initial_nll", float("nan"))),
        "n_active_bounds": int(getattr(last, "n_active_bounds", 0)),
        "converged": bool(last.converged),
    }


def _run_nextstat_library_minimum(
    spec_path: Path,
    warmup: int,
    repeats: int,
    threads: int,
    opt_max_iter: int | None,
    opt_tol: float | None,
    opt_m: int | None,
    opt_smooth_bounds: bool,
) -> Dict[str, Any]:
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    import nextstat  # type: ignore

    model = nextstat.UnbinnedModel.from_config(str(spec_path))
    mle = nextstat.MaximumLikelihoodEstimator(
        max_iter=1000 if opt_max_iter is None else int(opt_max_iter),
        tol=1e-6 if opt_tol is None else float(opt_tol),
        m=10 if opt_m is None else int(opt_m),
        smooth_bounds=bool(opt_smooth_bounds),
    )
    for _ in range(warmup):
        mle.fit_minimum(model)

    times_ms: List[float] = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = mle.fit_minimum(model)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "times_ms": times_ms,
        "min_ms": float(np.min(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "mean_ms": float(np.mean(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "nll": float(last.nll),
        "n_iter": int(last.n_iter),
        "n_fev": int(getattr(last, "n_fev", 0)),
        "n_gev": int(getattr(last, "n_gev", 0)),
        "termination_reason": str(getattr(last, "termination_reason", "")),
        "final_grad_norm": float(getattr(last, "final_grad_norm", float("nan"))),
        "initial_nll": float(getattr(last, "initial_nll", float("nan"))),
        "n_active_bounds": int(getattr(last, "n_active_bounds", 0)),
        "converged": bool(last.converged),
    }


def _run_morefit(binary: Path, data_txt: Path, lo: float, hi: float, repeats: int) -> Tuple[float, Dict[str, Any]]:
    cmd = [str(binary), str(data_txt), str(lo), str(hi), str(repeats)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} failed (rc={proc.returncode}): {proc.stderr[-500:]}")
    return wall_ms, _parse_json_tail(proc.stdout)


def _summarize(records: List[Dict[str, Any]], n_events: List[int], variant_keys: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for n in n_events:
        subset = [r for r in records if int(r["n_events"]) == int(n)]
        if not subset:
            continue
        summary[str(n)] = {
            "nextstat_cli_wall_ms_median": float(np.median([r["nextstat_cli"]["wall_ms"] for r in subset])),
            "nextstat_library_median_ms_median": float(np.median([r["nextstat_library"]["median_ms"] for r in subset])),
            "nextstat_library_minimum_median_ms_median": float(
                np.median([r["nextstat_library_minimum"]["median_ms"] for r in subset])
            ),
            "nextstat_covariance_overhead_ms_median": float(
                np.median(
                    [
                        r["nextstat_library"]["median_ms"] - r["nextstat_library_minimum"]["median_ms"]
                        for r in subset
                    ]
                )
            ),
            "nextstat_cli_optimizer": {
                "n_iter_median": int(np.median([r["nextstat_cli"]["n_iter"] for r in subset])),
                "n_fev_median": int(np.median([r["nextstat_cli"]["n_fev"] for r in subset])),
                "n_gev_median": int(np.median([r["nextstat_cli"]["n_gev"] for r in subset])),
                "final_grad_norm_median": float(
                    np.median([r["nextstat_cli"]["final_grad_norm"] for r in subset])
                ),
            },
            "nextstat_library_optimizer": {
                "n_iter_median": int(np.median([r["nextstat_library"]["n_iter"] for r in subset])),
                "n_fev_median": int(np.median([r["nextstat_library"]["n_fev"] for r in subset])),
                "n_gev_median": int(np.median([r["nextstat_library"]["n_gev"] for r in subset])),
                "final_grad_norm_median": float(
                    np.median([r["nextstat_library"]["final_grad_norm"] for r in subset])
                ),
            },
            "nextstat_library_minimum_optimizer": {
                "n_iter_median": int(np.median([r["nextstat_library_minimum"]["n_iter"] for r in subset])),
                "n_fev_median": int(np.median([r["nextstat_library_minimum"]["n_fev"] for r in subset])),
                "n_gev_median": int(np.median([r["nextstat_library_minimum"]["n_gev"] for r in subset])),
                "final_grad_norm_median": float(
                    np.median([r["nextstat_library_minimum"]["final_grad_norm"] for r in subset])
                ),
            },
            "variants": {},
        }
        ns_cli_ms = summary[str(n)]["nextstat_cli_wall_ms_median"]
        ns_lib_ms = summary[str(n)]["nextstat_library_median_ms_median"]
        for key in variant_keys:
            walls = [r["morefit"][key]["wall_repeat1_ms"] for r in subset]
            warms = [r["morefit"][key]["warm_median_ms"] for r in subset]
            wall_med = float(np.median(walls))
            warm_med = float(np.median(warms))
            summary[str(n)]["variants"][key] = {
                "wall_repeat1_ms_median": wall_med,
                "warm_median_ms_median": warm_med,
                "wall_vs_nextstat_cli_ratio": wall_med / ns_cli_ms if ns_cli_ms > 0 else float("nan"),
                "warm_vs_nextstat_library_ratio": warm_med / ns_lib_ms if ns_lib_ms > 0 else float("nan"),
            }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Symmetric CPU benchmark: NextStat vs MoreFit (gauss_exp)")
    ap.add_argument("--n-events", default="10000,100000,1000000", help="Comma-separated list of event counts")
    ap.add_argument("--seeds", default="42,43", help="Comma-separated RNG seeds")
    ap.add_argument("--threads", type=int, default=20, help="Threads for NextStat CLI and library path")
    ap.add_argument("--nextstat-opt-max-iter", type=int, default=None, help="Pass through to nextstat --opt-max-iter")
    ap.add_argument("--nextstat-opt-tol", type=float, default=None, help="Pass through to nextstat --opt-tol")
    ap.add_argument("--nextstat-opt-m", type=int, default=None, help="Pass through to nextstat --opt-m")
    ap.add_argument(
        "--nextstat-opt-smooth-bounds",
        action="store_true",
        help="Pass through to nextstat --opt-smooth-bounds",
    )
    ap.add_argument("--lib-warmup", type=int, default=2, help="Warmup fits for NextStat library path")
    ap.add_argument("--lib-repeats", type=int, default=9, help="Timed repeats for NextStat library path")
    ap.add_argument(
        "--morefit-warm-repeats",
        type=int,
        default=9,
        help="Timed repeats for MoreFit warm measurement (first entry treated as cold)",
    )
    ap.add_argument(
        "--morefit-1t-num",
        default="/root/morefit/morefit_gauss_exp",
        help="Path to MoreFit 1-thread numerical-gradient binary",
    )
    ap.add_argument(
        "--morefit-1t-agrad",
        default="/root/morefit/morefit_gauss_exp_agrad",
        help="Path to MoreFit 1-thread analytical-gradient binary",
    )
    ap.add_argument(
        "--morefit-20t-num",
        default="/root/morefit/morefit_gauss_exp_mt",
        help="Path to MoreFit 20-thread numerical-gradient binary",
    )
    ap.add_argument(
        "--morefit-20t-agrad",
        default="/root/morefit/morefit_gauss_exp_agrad_mt",
        help="Path to MoreFit 20-thread analytical-gradient binary",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: benchmarks/unbinned/artifacts/<date>/bench_symm_cpu_<timestamp>.json)",
    )
    args = ap.parse_args()

    n_events = _parse_csv_ints(args.n_events)
    seeds = _parse_csv_ints(args.seeds)
    nextstat_bin = _find_nextstat_cli()
    variants = _find_morefit_variants(args)
    variant_keys = list(variants.keys())

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    date_dir = time.strftime("%Y-%m-%d", time.gmtime())
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = (REPO_ROOT / "benchmarks" / "unbinned" / "artifacts" / date_dir / f"bench_symm_cpu_{stamp}.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_dir = (out_path.parent / out_path.stem).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for n in n_events:
        for seed in seeds:
            print(f"[run] N={n} seed={seed}", file=sys.stderr, flush=True)
            rng = np.random.default_rng(seed)
            built = _case_gauss_exp(rng, n)

            case_dir = (run_dir / f"N{n}_s{seed}").resolve()
            case_dir.mkdir(parents=True, exist_ok=True)

            bounds = built["observables"][0]["bounds"]
            lo, hi = float(bounds[0]), float(bounds[1])
            mass = built["columns"]["mass"]

            parquet_path = (case_dir / "observed.parquet").resolve()
            _write_parquet(columns=built["columns"], observables=built["observables"], path=parquet_path)

            txt_path = (case_dir / "observed.txt").resolve()
            with txt_path.open("w") as f:
                for val in mass:
                    f.write(f"{float(val):.17g}\n")

            spec = built["ns_spec"]
            spec["channels"][0]["data"]["file"] = str(parquet_path)
            spec_path = (case_dir / "spec.json").resolve()
            spec_path.write_text(json.dumps(spec, indent=2))

            ns_cli_wall_ms, ns_cli = _run_nextstat_cli(
                nextstat_bin,
                spec_path,
                args.threads,
                args.nextstat_opt_max_iter,
                args.nextstat_opt_tol,
                args.nextstat_opt_m,
                args.nextstat_opt_smooth_bounds,
            )
            ns_lib = _run_nextstat_library(
                spec_path,
                args.lib_warmup,
                args.lib_repeats,
                args.threads,
                args.nextstat_opt_max_iter,
                args.nextstat_opt_tol,
                args.nextstat_opt_m,
                args.nextstat_opt_smooth_bounds,
            )
            ns_lib_min = _run_nextstat_library_minimum(
                spec_path,
                args.lib_warmup,
                args.lib_repeats,
                args.threads,
                args.nextstat_opt_max_iter,
                args.nextstat_opt_tol,
                args.nextstat_opt_m,
                args.nextstat_opt_smooth_bounds,
            )
            if not ns_cli.get("converged", False):
                _die(f"nextstat CLI did not converge for N={n}, seed={seed}")
            if not ns_lib.get("converged", False):
                _die(f"nextstat library fit did not converge for N={n}, seed={seed}")
            if not ns_lib_min.get("converged", False):
                _die(f"nextstat library minimum fit did not converge for N={n}, seed={seed}")

            morefit_result: Dict[str, Any] = {}
            for key, bin_path in variants.items():
                wall_repeat1_ms, one_payload = _run_morefit(bin_path, txt_path, lo, hi, repeats=1)
                wall_repeats_ms, multi_payload = _run_morefit(
                    bin_path,
                    txt_path,
                    lo,
                    hi,
                    repeats=args.morefit_warm_repeats,
                )
                times_ms = [float(x) for x in multi_payload.get("times_ms", [])]
                warm = times_ms[1:] if len(times_ms) > 1 else times_ms
                morefit_result[key] = {
                    "binary": str(bin_path),
                    "wall_repeat1_ms": float(wall_repeat1_ms),
                    "fit_repeat1_ms": float(one_payload.get("mean_ms", one_payload.get("min_ms", float("nan")))),
                    "one_payload": one_payload,
                    "warm_repeats": int(args.morefit_warm_repeats),
                    "wall_repeats_ms": float(wall_repeats_ms),
                    "multi_payload": multi_payload,
                    "cold_ms": float(times_ms[0]) if times_ms else float("nan"),
                    "warm_min_ms": float(np.min(warm)) if warm else float("nan"),
                    "warm_median_ms": float(np.median(warm)) if warm else float("nan"),
                    "warm_mean_ms": float(np.mean(warm)) if warm else float("nan"),
                    "warm_max_ms": float(np.max(warm)) if warm else float("nan"),
                }

            record = {
                "n_events": int(n),
                "seed": int(seed),
                "nextstat_cli": {
                    "threads": int(args.threads),
                    "wall_ms": float(ns_cli_wall_ms),
                    "nll": float(ns_cli["nll"]),
                    "n_iter": int(ns_cli["n_iter"]),
                    "n_fev": int(ns_cli["n_fev"]),
                    "n_gev": int(ns_cli["n_gev"]),
                    "termination_reason": str(ns_cli.get("termination_reason", "")),
                    "final_grad_norm": float(ns_cli.get("final_grad_norm", float("nan"))),
                    "initial_nll": float(ns_cli.get("initial_nll", float("nan"))),
                    "n_active_bounds": int(ns_cli.get("n_active_bounds", 0)),
                    "bestfit": [float(x) for x in ns_cli["bestfit"]],
                    "converged": bool(ns_cli["converged"]),
                },
                "nextstat_library": ns_lib,
                "nextstat_library_minimum": ns_lib_min,
                "morefit": morefit_result,
            }
            records.append(record)
            (case_dir / "record.json").write_text(json.dumps(record, indent=2))

    summary = _summarize(records, n_events, variant_keys)
    payload = {
        "schema_version": "nextstat.unbinned_cpu_symmetry_bench.v1",
        "timestamp_utc": stamp,
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "cpu_lscpu": subprocess.check_output(["lscpu"], text=True),
        "threads_config": {"nextstat_threads": int(args.threads), "rayon_num_threads": int(args.threads)},
        "nextstat_optimizer_override": {
            "opt_max_iter": args.nextstat_opt_max_iter,
            "opt_tol": args.nextstat_opt_tol,
            "opt_m": args.nextstat_opt_m,
            "opt_smooth_bounds": bool(args.nextstat_opt_smooth_bounds),
        },
        "n_list": [int(n) for n in n_events],
        "seeds": [int(s) for s in seeds],
        "library": {"warmup": int(args.lib_warmup), "repeats": int(args.lib_repeats)},
        "morefit": {
            "warm_repeats": int(args.morefit_warm_repeats),
            "variants": {k: str(v) for k, v in variants.items()},
        },
        "records": records,
        "summary": summary,
        "artifact_dir": str(run_dir),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"WROTE {out_path}", file=sys.stderr)
    print(f"ARTDIR {run_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
