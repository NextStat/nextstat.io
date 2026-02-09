#!/usr/bin/env python3
"""The "God Run": toy-based CLs benchmark (pyhf vs NextStat).

Scenario (typical LHC-style HistFactory):
- S + B model with many nuisance parameters and channels
- Task: CLs via toy-based q~_mu (hypothesis test inversion primitive)
- Load: N toys (b-only) + N toys (s+b)
- Compare: pyhf (multiprocessing) vs nextstat (Rayon, SIMD, tape reuse)

Run (from repo root, after building the extension with maturin --release):
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python scripts/god_run_benchmark.py --n-toys 10000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cpu_brand() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
            )
            return out.decode().strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _git_info(repo_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
            .decode()
            .strip()
        )
        out["commit_short"] = out["commit"][:8]
        out["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root))
            .decode()
            .strip()
        )
        out["dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root))
            .decode()
            .strip()
        )
    except Exception:
        out["error"] = "git_unavailable"
    return out


def make_workspace_god_run(*, n_channels: int, n_bins: int) -> dict[str, Any]:
    """Synthetic multi-channel S+B workspace with ~n_channels*n_bins nuisance parameters."""
    if n_channels <= 0:
        raise ValueError("n_channels must be > 0")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")

    channels: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    for i in range(n_channels):
        ch = f"ch{i:02d}"

        # Deterministic shapes, moderate counts.
        signal = [2.0 + 0.05 * (j % 3) for j in range(n_bins)]
        background = [50.0 + 0.25 * j + 0.1 * (i % 5) for j in range(n_bins)]

        channels.append(
            {
                "name": ch,
                "samples": [
                    {
                        "name": "signal",
                        "data": signal,
                        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                    },
                    {
                        "name": "background",
                        "data": background,
                        "modifiers": [
                            # One gamma nuisance per bin -> n_channels * n_bins parameters.
                            {"name": f"shapesys_{ch}", "type": "shapesys", "data": [5.0] * n_bins}
                        ],
                    },
                ],
            }
        )

        # Observations: background-only Asimov (mu=0, nuisances at their centers).
        observations.append({"name": ch, "data": [float(x) for x in background]})

    return {
        "channels": channels,
        "observations": observations,
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }


# ── pyhf multiprocessing runner ───────────────────────────────────────────────


PyhfHypothesis = Literal["b", "sb"]


@dataclass(frozen=True)
class PyhfWorkerPayload:
    workspace: dict[str, Any]
    measurement_name: str
    mu_test: float
    q_obs: float
    expected_main_b: list[float]
    expected_main_sb: list[float]
    auxdata_fixed: list[float]
    warm_start: bool


_PYHF_G: dict[str, Any] = {}


def _pyhf_worker_init(payload: PyhfWorkerPayload) -> None:
    # Avoid oversubscription when running many processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import numpy as np
    import pyhf

    pyhf.set_backend("numpy")
    ws = pyhf.Workspace(payload.workspace)
    model = ws.model(measurement_name=payload.measurement_name)

    init_pars = list(model.config.suggested_init())
    par_bounds = list(model.config.suggested_bounds())
    fixed_params = list(model.config.suggested_fixed())

    _PYHF_G.clear()
    _PYHF_G.update(
        {
            "model": model,
            "mu_test": float(payload.mu_test),
            "q_obs": float(payload.q_obs),
            "expected_main_b": np.asarray(payload.expected_main_b, dtype=float),
            "expected_main_sb": np.asarray(payload.expected_main_sb, dtype=float),
            "aux_fixed": np.asarray(payload.auxdata_fixed, dtype=float),
            "init_pars": init_pars,
            "par_bounds": par_bounds,
            "fixed_params": fixed_params,
            "warm_start": bool(payload.warm_start),
        }
    )


def _pyhf_qmu_tilde_warm(
    *,
    mu_test: float,
    data,
    model,
    init_free: list[float],
    init_fixed: list[float],
    par_bounds,
    fixed_params,
) -> tuple[float, Any, Any]:
    """Compute q~_mu with explicit warm-start init vectors.

    Mirrors pyhf.infer.test_statistics.qmu_tilde but allows per-call init updates.
    Returns: (q, free_pars, fixed_pars)
    """
    from pyhf.infer.mle import fit, fixed_poi_fit

    fixed_pars, fixed_twice_nll = fixed_poi_fit(
        mu_test,
        data,
        model,
        init_pars=init_fixed,
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        return_fitted_val=True,
    )
    free_pars, free_twice_nll = fit(
        data,
        model,
        init_pars=init_free,
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        return_fitted_val=True,
    )
    q = max(0.0, float(fixed_twice_nll) - float(free_twice_nll))
    mu_hat = float(free_pars[int(model.config.poi_index)])
    if mu_hat > mu_test:
        q = 0.0
    return q, free_pars, fixed_pars


def _pyhf_worker_run_range(args: tuple[PyhfHypothesis, int, int, int, float]) -> tuple[int, int]:
    which, start_idx, n_toys, seed, q_threshold = args

    import numpy as np
    from pyhf.infer.test_statistics import qmu_tilde

    model = _PYHF_G["model"]
    mu_test = _PYHF_G["mu_test"]
    expected_main = _PYHF_G["expected_main_b"] if which == "b" else _PYHF_G["expected_main_sb"]
    aux_fixed = _PYHF_G["aux_fixed"]
    init_pars = _PYHF_G["init_pars"]
    par_bounds = _PYHF_G["par_bounds"]
    fixed_params = _PYHF_G["fixed_params"]
    warm_start = _PYHF_G["warm_start"]

    count = 0
    errors = 0

    init_free = list(init_pars)
    init_fixed = list(init_pars)

    for i in range(start_idx, start_idx + n_toys):
        try:
            rng = np.random.default_rng(int(seed) + int(i))
            toy_main = rng.poisson(expected_main)
            data = np.concatenate([toy_main, aux_fixed])

            if warm_start:
                q, free_pars, fixed_pars = _pyhf_qmu_tilde_warm(
                    mu_test=mu_test,
                    data=data,
                    model=model,
                    init_free=init_free,
                    init_fixed=init_fixed,
                    par_bounds=par_bounds,
                    fixed_params=fixed_params,
                )
                init_free = list(map(float, free_pars))
                init_fixed = list(map(float, fixed_pars))
            else:
                q = float(qmu_tilde(mu_test, data, model, init_pars, par_bounds, fixed_params))

            if q >= q_threshold:
                count += 1
        except Exception:
            errors += 1

    return count, errors


@dataclass(frozen=True)
class PyhfToyResult:
    cls: float
    clsb: float
    clb: float
    q_obs: float
    n_toys: int
    n_procs: int
    warm_start: bool
    errors_b: int
    errors_sb: int


def _toy_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    base = total // parts
    rem = total % parts
    out: list[tuple[int, int]] = []
    start = 0
    for p in range(parts):
        n = base + (1 if p < rem else 0)
        out.append((start, n))
        start += n
    return out


def run_pyhf_multiprocessing(
    *,
    workspace: dict[str, Any],
    measurement_name: str,
    mu_test: float,
    n_toys: int,
    seed: int,
    n_procs: int,
    warm_start: bool,
) -> PyhfToyResult:
    import numpy as np
    import pyhf

    pyhf.set_backend("numpy")

    ws = pyhf.Workspace(workspace)
    model = ws.model(measurement_name=measurement_name)
    data_obs = ws.data(model)

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    fixed_params = model.config.suggested_fixed()

    # Observed test statistic (q~_mu).
    from pyhf.infer.test_statistics import qmu_tilde

    q_obs = float(qmu_tilde(mu_test, data_obs, model, init_pars, par_bounds, fixed_params))

    # Plug-in generation points (conditional fits on observed data).
    from pyhf.infer.mle import fixed_poi_fit

    pars_sb = fixed_poi_fit(mu_test, data_obs, model, init_pars, par_bounds, fixed_params)
    pars_b = fixed_poi_fit(0.0, data_obs, model, init_pars, par_bounds, fixed_params)

    exp_sb = np.asarray(model.expected_data(pars_sb), dtype=float)
    exp_b = np.asarray(model.expected_data(pars_b), dtype=float)

    n_aux = len(model.config.auxdata)
    n_main = len(data_obs) - n_aux
    expected_main_sb = exp_sb[:n_main].tolist()
    expected_main_b = exp_b[:n_main].tolist()

    # Match NextStat Phase-3.2 toy policy: fluctuate main only, keep aux fixed.
    aux_fixed = list(map(float, model.config.auxdata))

    payload = PyhfWorkerPayload(
        workspace=workspace,
        measurement_name=measurement_name,
        mu_test=mu_test,
        q_obs=q_obs,
        expected_main_b=expected_main_b,
        expected_main_sb=expected_main_sb,
        auxdata_fixed=aux_fixed,
        warm_start=warm_start,
    )

    actual_procs = max(1, min(int(n_procs), int(n_toys)))
    ctx = __import__("multiprocessing").get_context("spawn")

    with ctx.Pool(
        processes=actual_procs, initializer=_pyhf_worker_init, initargs=(payload,)
    ) as pool:
        print(f"pyhf: running s+b toys (n={n_toys}, procs={actual_procs}) ...", flush=True)
        tasks_sb = [
            ("sb", start, n, seed + 1_000_000_000, q_obs)
            for start, n in _toy_ranges(n_toys, actual_procs)
            if n > 0
        ]
        out_sb = pool.map(_pyhf_worker_run_range, tasks_sb)

        print(f"pyhf: running b-only toys (n={n_toys}, procs={actual_procs}) ...", flush=True)
        tasks_b = [
            ("b", start, n, seed, q_obs)
            for start, n in _toy_ranges(n_toys, actual_procs)
            if n > 0
        ]
        out_b = pool.map(_pyhf_worker_run_range, tasks_b)

    count_sb = sum(c for (c, _e) in out_sb)
    count_b = sum(c for (c, _e) in out_b)
    errors_sb = sum(e for (_c, e) in out_sb)
    errors_b = sum(e for (_c, e) in out_b)

    # Add-one smoothing (matches NextStat toybased.rs).
    clsb = (count_sb + 1.0) / (n_toys + 1.0)
    clb = (count_b + 1.0) / (n_toys + 1.0)
    cls = float("inf") if clb == 0.0 else float(clsb / clb)

    return PyhfToyResult(
        cls=float(cls),
        clsb=float(clsb),
        clb=float(clb),
        q_obs=float(q_obs),
        n_toys=int(n_toys),
        n_procs=int(actual_procs),
        warm_start=bool(warm_start),
        errors_b=int(errors_b),
        errors_sb=int(errors_sb),
    )


# ── Output helpers ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunMeta:
    timestamp_unix: int
    hostname: str
    python: str
    platform: str
    machine: str
    cpu: str
    git: dict[str, Any]
    versions: dict[str, str]
    env: dict[str, str]


def collect_meta(repo_root: Path) -> RunMeta:
    versions: dict[str, str] = {}
    try:
        import nextstat

        versions["nextstat"] = str(nextstat.__version__)
    except Exception:
        versions["nextstat"] = "unavailable"
    try:
        import pyhf

        versions["pyhf"] = str(pyhf.__version__)
    except Exception:
        versions["pyhf"] = "unavailable"

    return RunMeta(
        timestamp_unix=int(time.time()),
        hostname=socket.gethostname(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=_cpu_brand(),
        git=_git_info(repo_root),
        versions=versions,
        env={
            k: os.environ.get(k, "")
            for k in [
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
    )


def format_seconds(s: float) -> str:
    if s < 1.0:
        return f"{s * 1e3:.1f} ms"
    if s < 60.0:
        return f"{s:.2f} s"
    m, rem = divmod(s, 60.0)
    if m < 60.0:
        return f"{int(m)}m {rem:04.1f}s"
    h, rem_m = divmod(m, 60.0)
    return f"{int(h)}h {int(rem_m)}m {rem:04.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-channels", type=int, default=50)
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--mu-test", type=float, default=1.0)
    ap.add_argument("--n-toys", type=int, default=10_000, help="Toys per hypothesis (b-only + s+b).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pyhf-procs", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument(
        "--pyhf-warm-start",
        action="store_true",
        help="Warm-start pyhf fits across toys within each worker (more favorable to pyhf).",
    )
    ap.add_argument("--out-json", type=Path, default=Path("tmp/god_run_report.json"))
    ap.add_argument("--out-md", type=Path, default=Path("tmp/god_run_snippet.md"))
    args = ap.parse_args()

    # Disable thread oversubscription for numpy/scipy in parent and workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    repo = _repo_root()
    workspace = make_workspace_god_run(n_channels=args.n_channels, n_bins=args.n_bins)
    ws_json = json.dumps(workspace)

    # Imports after env vars are set.
    import pyhf

    pyhf.set_backend("numpy")
    ws = pyhf.Workspace(workspace)
    py_model = ws.model(measurement_name="m")
    py_data = ws.data(py_model)

    import nextstat

    ns_model = nextstat.HistFactoryModel.from_workspace(ws_json)

    meta = collect_meta(repo)

    # Summary
    n_channels = len(workspace["channels"])
    n_bins = int(args.n_bins)
    n_main = n_channels * n_bins
    n_aux = len(py_model.config.auxdata)
    n_params = int(py_model.config.npars)

    print("God Run: toy-based CLs (q~_mu)")
    print(f"- Model: S+B, channels={n_channels}, bins/channel={n_bins}")
    print(f"- Params: {n_params} (mu + {n_params - 1} nuisances)")
    print(f"- Data: main={n_main}, aux={n_aux}, total={len(py_data)}")
    print(f"- Toys: {args.n_toys} (b-only) + {args.n_toys} (s+b)")
    print(f"- mu_test: {args.mu_test}")
    print(f"- pyhf procs: {args.pyhf_procs} (warm_start={bool(args.pyhf_warm_start)})")

    # Warm up both stacks (avoid one-time threadpool init skew).
    from pyhf.infer.test_statistics import qmu_tilde

    _ = qmu_tilde(
        args.mu_test,
        py_data,
        py_model,
        py_model.config.suggested_init(),
        py_model.config.suggested_bounds(),
        py_model.config.suggested_fixed(),
    )
    _ = nextstat.hypotest_toys(args.mu_test, ns_model, n_toys=5, seed=args.seed)

    # NextStat
    t0 = time.perf_counter()
    ns_meta = nextstat.hypotest_toys(
        args.mu_test,
        ns_model,
        n_toys=args.n_toys,
        seed=args.seed,
        return_meta=True,
    )
    ns_wall = time.perf_counter() - t0
    print(f"\nNextStat done in {format_seconds(ns_wall)}", flush=True)

    # pyhf (multiprocessing)
    print("Running pyhf multiprocessing (this can take a while) ...", flush=True)
    t0 = time.perf_counter()
    py_res = run_pyhf_multiprocessing(
        workspace=workspace,
        measurement_name="m",
        mu_test=args.mu_test,
        n_toys=args.n_toys,
        seed=args.seed,
        n_procs=args.pyhf_procs,
        warm_start=bool(args.pyhf_warm_start),
    )
    py_wall = time.perf_counter() - t0

    speedup = py_wall / ns_wall if ns_wall > 0 else float("inf")

    print("\nResults")
    print(
        f"- NextStat: {format_seconds(ns_wall)} "
        f"(CLs={ns_meta['cls']:.6g}, errors={ns_meta['n_error_b']}/{ns_meta['n_error_sb']})"
    )
    print(
        f"- pyhf:     {format_seconds(py_wall)} "
        f"(CLs={py_res.cls:.6g}, errors={py_res.errors_b}/{py_res.errors_sb})"
    )
    print(f"- Speedup:  {speedup:.1f}x")

    report: dict[str, Any] = {
        "meta": asdict(meta),
        "scenario": {
            "n_channels": n_channels,
            "n_bins": n_bins,
            "n_params": n_params,
            "n_main": n_main,
            "n_aux": int(n_aux),
            "mu_test": float(args.mu_test),
            "n_toys_per_hypothesis": int(args.n_toys),
            "seed": int(args.seed),
        },
        "nextstat": {
            "wall_s": float(ns_wall),
            "result": ns_meta,
        },
        "pyhf": {
            "wall_s": float(py_wall),
            "result": asdict(py_res),
        },
        "speedup": {
            "pyhf_over_nextstat": float(speedup),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    md_lines = [
        "## The God Run (toy-based CLs)",
        "",
        f"- **Model:** S+B HistFactory (synthetic), {n_channels} channels × {n_bins} bins, "
        f"{n_params} parameters (mu + {n_params - 1} nuisances)",
        "- **Task:** CLs via toy-based q~_mu",
        f"- **Load:** {args.n_toys:,} toys (b-only) + {args.n_toys:,} toys (s+b)",
        f"- **Machine:** {meta.cpu} ({meta.machine}), {meta.platform}",
        f"- **Versions:** nextstat {meta.versions.get('nextstat','?')}, "
        f"pyhf {meta.versions.get('pyhf','?')}, Python {meta.python}",
        f"- **Recorded:** {time.strftime('%Y-%m-%d', time.gmtime(meta.timestamp_unix))} (UTC)",
        f"- **Commit:** {meta.git.get('commit_short', '?')}{' (dirty)' if meta.git.get('dirty') else ''}",
        "",
        "| Tool | Wall time | Speedup |",
        "|---|---:|---:|",
        f"| NextStat (Rayon) | {format_seconds(ns_wall)} | 1.0× |",
        f"| pyhf (multiprocessing, {py_res.n_procs} procs) | {format_seconds(py_wall)} | {speedup:.1f}× |",
        "",
        "Reproduce:",
        "```bash",
        "PYTHONPATH=bindings/ns-py/python ./.venv/bin/python scripts/god_run_benchmark.py --n-toys 10000",
        "```",
    ]
    if args.pyhf_warm_start:
        md_lines.insert(7, "- **pyhf note:** warm-start enabled (`--pyhf-warm-start`)")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n")

    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
