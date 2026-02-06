#!/usr/bin/env python3
"""Record reference baseline JSONs for Apex2 regression testing.

This script captures a full environment fingerprint and runs the pyhf
validation + P6 GLM fit/predict benchmarks (and optionally the ROOT/HistFactory
parity suite), saving the results as
timestamped baseline files under ``tmp/baselines/``.

Usage (from repo root, after ``maturin develop --release``):

  # Record both baselines (default):
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py

  # Record only pyhf baseline:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py --only pyhf

  # Record only P6 GLM baseline:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py --only p6

  # Record ROOT/HistFactory parity baseline (cluster/root required):
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py \
    --only root --root-search-dir /abs/path/to/trex/output

  # Custom output directory:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py --out-dir tmp/baselines

The resulting files are named:
  <out-dir>/pyhf_baseline_<host>_<date>.json
  <out-dir>/p6_glm_baseline_<host>_<date>.json
  <out-dir>/baseline_manifest_<host>_<date>.json   (links both + env fingerprint)
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _ensure_nextstat_import_path(repo: Path) -> None:
    """Ensure `import nextstat` works in this process (not just subprocesses)."""
    add = repo / "bindings" / "ns-py" / "python"
    add_s = str(add)
    if add_s not in sys.path:
        sys.path.insert(0, add_s)


# ── Environment fingerprint ────────────────────────────────────────────


def _git_info(repo: Path) -> Dict[str, Any]:
    """Capture git commit, branch, dirty status."""
    info: Dict[str, Any] = {}
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["commit_short"] = info["commit"][:8]
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(repo),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=str(repo), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["error"] = "git_not_available"
    return info


def _cpu_brand() -> str:
    """Best-effort CPU brand string."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
            )
            return out.decode().strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def collect_environment(repo: Path) -> Dict[str, Any]:
    """Full environment fingerprint for baseline reproducibility."""
    _ensure_nextstat_import_path(repo)
    env: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "datetime_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": _cpu_brand(),
    }

    # Package versions
    try:
        import nextstat  # type: ignore

        env["nextstat_version"] = str(nextstat.__version__)
    except (ImportError, AttributeError):
        env["nextstat_version"] = "unavailable"

    try:
        import pyhf  # type: ignore

        env["pyhf_version"] = str(pyhf.__version__)
    except ImportError:
        env["pyhf_version"] = "unavailable"

    try:
        import numpy as np  # type: ignore

        env["numpy_version"] = str(np.__version__)
    except ImportError:
        env["numpy_version"] = "unavailable"

    env["git"] = _git_info(repo)
    return env


# ── Runner helpers ──────────────────────────────────────────────────────


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


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return p.returncode, p.stdout


def _stamp(hostname: str) -> str:
    return f"{hostname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _copy_json_with_baseline_env(
    *,
    src: Path,
    dst: Path,
    environment: Dict[str, Any],
    recorded_as_baseline: bool = True,
) -> None:
    """Copy a JSON file and inject `baseline_env` if it is a JSON object."""
    try:
        obj = json.loads(src.read_text())
    except Exception:
        dst.write_text(src.read_text())
        return
    if isinstance(obj, dict):
        obj["baseline_env"] = environment
        meta = obj.get("meta")
        if recorded_as_baseline and isinstance(meta, dict):
            meta.setdefault("recorded_as_baseline", True)
        dst.write_text(json.dumps(obj, indent=2))
    else:
        dst.write_text(src.read_text())


# ── Baseline recording ─────────────────────────────────────────────────


def record_pyhf_baseline(
    *,
    repo: Path,
    env_dict: Dict[str, str],
    environment: Dict[str, Any],
    out_dir: Path,
    stamp: str,
    pyhf_sizes: str,
    pyhf_n_random: int,
    pyhf_seed: int,
    pyhf_fit: bool,
) -> Optional[Path]:
    """Run pyhf validation and save as baseline."""
    out_path = out_dir / f"pyhf_baseline_{stamp}.json"
    runner = repo / "tests" / "apex2_pyhf_validation_report.py"

    cmd = [
        sys.executable,
        str(runner),
        "--out",
        str(out_path),
        "--sizes",
        pyhf_sizes,
        "--n-random",
        str(pyhf_n_random),
        "--seed",
        str(pyhf_seed),
    ]
    if pyhf_fit:
        cmd.append("--fit")

    print(f"[pyhf] Running validation...")
    t0 = time.time()
    rc, output = _run(cmd, cwd=repo, env=env_dict)
    wall = time.time() - t0

    if rc != 0:
        print(f"[pyhf] FAILED (exit {rc})")
        print(output[-2000:])
        return None

    # Inject environment fingerprint into the report
    if out_path.exists():
        report = json.loads(out_path.read_text())
        report["baseline_env"] = environment
        report["meta"]["recorded_as_baseline"] = True
        report["meta"]["wall_s"] = wall
        out_path.write_text(json.dumps(report, indent=2))

    print(f"[pyhf] OK  ({wall:.1f}s) -> {out_path}")
    return out_path


def record_p6_glm_baseline(
    *,
    repo: Path,
    env_dict: Dict[str, str],
    environment: Dict[str, Any],
    out_dir: Path,
    stamp: str,
    sizes: str,
    p: int,
    l2: float,
    nb_alpha: float,
) -> Optional[Path]:
    """Run P6 GLM fit/predict benchmark and save as baseline."""
    out_path = out_dir / f"p6_glm_baseline_{stamp}.json"
    runner = repo / "tests" / "benchmark_glm_fit_predict.py"

    cmd = [
        sys.executable,
        str(runner),
        "--sizes",
        sizes,
        "--p",
        str(p),
        "--l2",
        str(l2),
        "--nb-alpha",
        str(nb_alpha),
        "--out",
        str(out_path),
    ]

    print(f"[p6_glm] Running benchmark (sizes={sizes}, p={p})...")
    t0 = time.time()
    rc, output = _run(cmd, cwd=repo, env=env_dict)
    wall = time.time() - t0

    if rc != 0:
        print(f"[p6_glm] FAILED (exit {rc})")
        print(output[-2000:])
        return None

    # Inject environment fingerprint into the report
    if out_path.exists():
        report = json.loads(out_path.read_text())
        report["baseline_env"] = environment
        report["meta"]["recorded_as_baseline"] = True
        report["meta"]["wall_s"] = wall
        out_path.write_text(json.dumps(report, indent=2))

    print(f"[p6_glm] OK  ({wall:.1f}s) -> {out_path}")
    return out_path


def record_nuts_quality_baseline(
    *,
    repo: Path,
    env_dict: Dict[str, str],
    environment: Dict[str, Any],
    out_dir: Path,
    stamp: str,
    nuts_cases: str,
    nuts_warmup: int,
    nuts_samples: int,
    nuts_funnel_warmup: int,
    nuts_funnel_samples: int,
    nuts_seed: int,
) -> Optional[Path]:
    """Run NUTS quality gate runner and save as baseline."""
    out_path = out_dir / f"nuts_quality_baseline_{stamp}.json"
    runner = repo / "tests" / "apex2_nuts_quality_report.py"

    cmd = [
        sys.executable,
        str(runner),
        "--out",
        str(out_path),
        "--cases",
        str(nuts_cases),
        "--warmup",
        str(int(nuts_warmup)),
        "--samples",
        str(int(nuts_samples)),
        "--funnel-warmup",
        str(int(nuts_funnel_warmup)),
        "--funnel-samples",
        str(int(nuts_funnel_samples)),
        "--seed",
        str(int(nuts_seed)),
    ]

    print("[nuts_quality] Running quality runner...")
    t0 = time.time()
    rc, output = _run(cmd, cwd=repo, env=env_dict)
    wall = time.time() - t0

    if rc != 0:
        print(f"[nuts_quality] FAILED (exit {rc})")
        print(output[-2000:])
        return None

    if out_path.exists():
        report = json.loads(out_path.read_text())
        report["baseline_env"] = environment
        report.setdefault("meta", {})
        if isinstance(report["meta"], dict):
            report["meta"]["recorded_as_baseline"] = True
            report["meta"]["wall_s"] = float(wall)
        out_path.write_text(json.dumps(report, indent=2))

    print(f"[nuts_quality] OK  ({wall:.1f}s) -> {out_path}")
    return out_path


def record_bias_pulls_baseline(
    *,
    repo: Path,
    env_dict: Dict[str, str],
    environment: Dict[str, Any],
    out_dir: Path,
    stamp: str,
    n_toys: int,
    seed: int,
    mu_truth: float,
    fixtures: str,
    include_zoo: bool,
    zoo_sizes: str,
    zoo_n_toys: Optional[int],
    params: str,
) -> Optional[Path]:
    """Run bias/pulls regression suite and save as baseline (slow)."""
    out_path = out_dir / f"bias_pulls_baseline_{stamp}.json"
    runner = repo / "tests" / "apex2_bias_pulls_report.py"

    cmd = [
        sys.executable,
        str(runner),
        "--out",
        str(out_path),
        "--n-toys",
        str(int(n_toys)),
        "--seed",
        str(int(seed)),
        "--mu-truth",
        str(float(mu_truth)),
        "--fixtures",
        str(fixtures),
        "--params",
        str(params),
    ]
    if include_zoo:
        cmd.append("--include-zoo")
        if str(zoo_sizes).strip():
            cmd += ["--zoo-sizes", str(zoo_sizes)]
        if zoo_n_toys is not None:
            cmd += ["--zoo-n-toys", str(int(zoo_n_toys))]

    print("[bias_pulls] Running suite (slow)...")
    t0 = time.time()
    rc, output = _run(cmd, cwd=repo, env=env_dict)
    wall = time.time() - t0

    if rc != 0:
        print(f"[bias_pulls] FAILED (exit {rc})")
        print(output[-2000:])
        return None

    if out_path.exists():
        report = json.loads(out_path.read_text())
        # Treat "skipped" as failure for baselines: if dependencies were missing, this baseline is not useful.
        status = ((report.get("summary") or {}).get("status") if isinstance(report, dict) else None) or None
        if status != "ok":
            print(f"[bias_pulls] FAILED (status {status!r})")
            return None
        report["baseline_env"] = environment
        report.setdefault("meta", {})
        if isinstance(report["meta"], dict):
            report["meta"]["recorded_as_baseline"] = True
            report["meta"]["wall_s"] = float(wall)
        out_path.write_text(json.dumps(report, indent=2))

    print(f"[bias_pulls] OK  ({wall:.1f}s) -> {out_path}")
    return out_path


def record_root_suite_baseline(
    *,
    repo: Path,
    env_dict: Dict[str, str],
    environment: Dict[str, Any],
    out_dir: Path,
    stamp: str,
    root_search_dir: Path,
    root_glob: str,
    root_include_fixtures: bool,
    root_cases_absolute_paths: bool,
    root_mu_start: float,
    root_mu_stop: float,
    root_mu_points: int,
    root_keep_going: bool,
    root_dq_atol: float,
    root_mu_hat_atol: float,
) -> Dict[str, Optional[Path]]:
    """Run ROOT suite parity and save as baseline if prereqs are available.

    Always records a prereq report. Only runs the full suite if prereqs are satisfied.
    """
    prereq_out = out_dir / f"root_prereq_{stamp}.json"
    prereq_runner = repo / "tests" / "apex2_root_suite_report.py"
    cmd_prereq = [sys.executable, str(prereq_runner), "--prereq-only", "--out", str(prereq_out)]
    rc_prereq, out_prereq = _run(cmd_prereq, cwd=repo, env=env_dict)
    if rc_prereq != 0:
        print("[root] Prereqs not satisfied; recording prereq report and skipping suite.")
        if out_prereq:
            print(out_prereq[-2000:])
        return {"prereq": prereq_out if prereq_out.exists() else None, "cases": None, "suite": None}

    cases_out = out_dir / f"root_cases_{stamp}.json"
    gen = repo / "tests" / "generate_apex2_root_cases.py"
    cmd_cases = [
        sys.executable,
        str(gen),
        "--search-dir",
        str(root_search_dir),
        "--glob",
        str(root_glob),
        "--out",
        str(cases_out),
        "--start",
        str(float(root_mu_start)),
        "--stop",
        str(float(root_mu_stop)),
        "--points",
        str(int(root_mu_points)),
    ]
    if root_include_fixtures:
        cmd_cases.append("--include-fixtures")
    if root_cases_absolute_paths:
        cmd_cases.append("--absolute-paths")

    print("[root] Generating cases JSON...")
    rc_cases, out_cases = _run(cmd_cases, cwd=repo, env=env_dict)
    if rc_cases != 0 or not cases_out.exists():
        print(f"[root] FAILED to generate cases (exit {rc_cases})")
        print(out_cases[-2000:])
        return {"prereq": prereq_out if prereq_out.exists() else None, "cases": None, "suite": None}

    suite_out = out_dir / f"root_suite_baseline_{stamp}.json"
    workdir = out_dir / f"root_parity_suite_{stamp}"
    cmd_suite = [
        sys.executable,
        str(prereq_runner),
        "--cases",
        str(cases_out),
        "--workdir",
        str(workdir),
        "--dq-atol",
        str(float(root_dq_atol)),
        "--mu-hat-atol",
        str(float(root_mu_hat_atol)),
        "--out",
        str(suite_out),
    ]
    if root_keep_going:
        cmd_suite.append("--keep-going")

    print("[root] Running suite (this can be slow)...")
    t0 = time.time()
    rc_suite, out_suite = _run(cmd_suite, cwd=repo, env=env_dict)
    wall = time.time() - t0
    if rc_suite != 0:
        print(f"[root] FAILED (exit {rc_suite})")
        print(out_suite[-2000:])
        return {"prereq": prereq_out if prereq_out.exists() else None, "cases": cases_out, "suite": None}

    if suite_out.exists():
        report = json.loads(suite_out.read_text())
        report["baseline_env"] = environment
        meta = report.get("meta")
        if isinstance(meta, dict):
            meta["recorded_as_baseline"] = True
            meta["wall_s"] = float(wall)
        suite_out.write_text(json.dumps(report, indent=2))

    print(f"[root] OK  ({wall:.1f}s) -> {suite_out}")
    return {"prereq": prereq_out if prereq_out.exists() else None, "cases": cases_out, "suite": suite_out}


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Record reference baseline JSONs for Apex2 regression testing."
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/baselines"),
        help="Directory for baseline files (default: tmp/baselines).",
    )
    ap.add_argument(
        "--only",
        choices=["pyhf", "p6", "nuts_quality", "bias_pulls", "root"],
        default=None,
        help="Record only one baseline type (default: both).",
    )
    # pyhf options
    ap.add_argument("--pyhf-sizes", default="2,16,64,256", help="Synthetic bin counts for pyhf.")
    ap.add_argument("--pyhf-n-random", type=int, default=8, help="Random param points per model.")
    ap.add_argument("--pyhf-seed", type=int, default=0)
    ap.add_argument("--pyhf-fit", action="store_true", help="Also profile MLE fits in pyhf.")
    # P6 GLM options
    ap.add_argument("--sizes", default="200,2000,20000", help="Sample sizes for GLM benchmark.")
    ap.add_argument("--p", type=int, default=20, help="Feature count.")
    ap.add_argument("--l2", type=float, default=0.0, help="Ridge penalty (0=off).")
    ap.add_argument("--nb-alpha", type=float, default=0.5, help="NegBin dispersion.")
    # NUTS quality options (Phase 3 Bayesian contract)
    ap.add_argument("--nuts-cases", default="gaussian,posterior,funnel,linear,histfactory")
    ap.add_argument("--nuts-warmup", type=int, default=200)
    ap.add_argument("--nuts-samples", type=int, default=200)
    ap.add_argument("--nuts-funnel-warmup", type=int, default=300)
    ap.add_argument("--nuts-funnel-samples", type=int, default=300)
    ap.add_argument("--nuts-seed", type=int, default=0)
    # Bias/pulls regression (slow; frequentist parity regression suite)
    ap.add_argument("--bias-n-toys", type=int, default=200)
    ap.add_argument("--bias-seed", type=int, default=0)
    ap.add_argument("--bias-mu-truth", type=float, default=1.0)
    ap.add_argument("--bias-fixtures", type=str, default="simple")
    ap.add_argument("--bias-params", type=str, default="poi", help="poi or all")
    ap.add_argument("--bias-include-zoo", action="store_true")
    ap.add_argument("--bias-zoo-sizes", type=str, default="")
    ap.add_argument(
        "--bias-zoo-n-toys",
        type=int,
        default=None,
        help="Optional override for number of toys for model-zoo cases (requires --bias-include-zoo).",
    )
    # ROOT suite options (optional; requires ROOT + hist2workspace + uproot)
    ap.add_argument("--root-search-dir", type=Path, default=None, help="Directory to scan for TRExFitter/HistFactory exports (combination.xml).")
    ap.add_argument("--root-glob", type=str, default="**/combination.xml", help="Glob for HistFactory XML under --root-search-dir.")
    ap.add_argument("--root-include-fixtures", action="store_true", help="Include the built-in smoke fixture case in the ROOT suite cases list.")
    ap.add_argument("--root-cases-absolute-paths", action="store_true", help="Write absolute paths in generated ROOT cases JSON.")
    ap.add_argument("--root-mu-start", type=float, default=0.0)
    ap.add_argument("--root-mu-stop", type=float, default=5.0)
    ap.add_argument("--root-mu-points", type=int, default=51)
    ap.add_argument("--root-keep-going", action="store_true", help="Keep running the ROOT suite after failures/skips.")
    ap.add_argument("--root-dq-atol", type=float, default=1e-3)
    ap.add_argument("--root-mu-hat-atol", type=float, default=1e-3)
    ap.add_argument(
        "--root-suite-existing",
        type=Path,
        default=None,
        help="Path to an already-produced ROOT suite JSON report to register as a baseline (e.g. HTCondor array + aggregation output).",
    )
    ap.add_argument(
        "--root-cases-existing",
        type=Path,
        default=None,
        help="Optional existing cases JSON to register alongside --root-suite-existing.",
    )
    ap.add_argument(
        "--root-prereq-existing",
        type=Path,
        default=None,
        help="Optional existing prereq JSON to register alongside --root-suite-existing.",
    )
    args = ap.parse_args()

    repo = _repo_root()
    env_dict = _with_py_path(os.environ.copy())
    _ensure_nextstat_import_path(repo)

    # Collect environment fingerprint
    environment = collect_environment(repo)

    print("=" * 72)
    print("Apex2 Baseline Recorder")
    print("=" * 72)
    print(f"  hostname:  {environment['hostname']}")
    print(f"  python:    {environment['python']}")
    print(f"  platform:  {environment['platform']}")
    print(f"  machine:   {environment['machine']}")
    print(f"  cpu:       {environment['cpu']}")
    print(f"  nextstat:  {environment['nextstat_version']}")
    print(f"  pyhf:      {environment['pyhf_version']}")
    print(f"  numpy:     {environment['numpy_version']}")
    git = environment.get("git", {})
    if git.get("commit"):
        dirty = " (dirty)" if git.get("dirty") else ""
        print(f"  git:       {git['commit_short']} ({git['branch']}){dirty}")
    print(f"  time:      {environment['datetime_utc']}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(environment["hostname"])

    pyhf_path: Optional[Path] = None
    p6_path: Optional[Path] = None
    nuts_path: Optional[Path] = None
    bias_path: Optional[Path] = None
    root_prereq_path: Optional[Path] = None
    root_cases_path: Optional[Path] = None
    root_suite_path: Optional[Path] = None
    any_failed = False

    # Record pyhf baseline
    if args.only is None or args.only == "pyhf":
        pyhf_path = record_pyhf_baseline(
            repo=repo,
            env_dict=env_dict,
            environment=environment,
            out_dir=args.out_dir,
            stamp=stamp,
            pyhf_sizes=args.pyhf_sizes,
            pyhf_n_random=args.pyhf_n_random,
            pyhf_seed=args.pyhf_seed,
            pyhf_fit=args.pyhf_fit,
        )
        if pyhf_path is None:
            any_failed = True

    # Record P6 GLM baseline
    if args.only is None or args.only == "p6":
        p6_path = record_p6_glm_baseline(
            repo=repo,
            env_dict=env_dict,
            environment=environment,
            out_dir=args.out_dir,
            stamp=stamp,
            sizes=args.sizes,
            p=args.p,
            l2=args.l2,
            nb_alpha=args.nb_alpha,
        )
        if p6_path is None:
            any_failed = True

    # Record NUTS quality baseline
    if args.only is None or args.only == "nuts_quality":
        nuts_path = record_nuts_quality_baseline(
            repo=repo,
            env_dict=env_dict,
            environment=environment,
            out_dir=args.out_dir,
            stamp=stamp,
            nuts_cases=str(args.nuts_cases),
            nuts_warmup=int(args.nuts_warmup),
            nuts_samples=int(args.nuts_samples),
            nuts_funnel_warmup=int(args.nuts_funnel_warmup),
            nuts_funnel_samples=int(args.nuts_funnel_samples),
            nuts_seed=int(args.nuts_seed),
        )
        if nuts_path is None:
            any_failed = True

    # Record bias/pulls baseline (slow; optional)
    if args.only is None or args.only == "bias_pulls":
        bias_path = record_bias_pulls_baseline(
            repo=repo,
            env_dict=env_dict,
            environment=environment,
            out_dir=args.out_dir,
            stamp=stamp,
            n_toys=int(args.bias_n_toys),
            seed=int(args.bias_seed),
            mu_truth=float(args.bias_mu_truth),
            fixtures=str(args.bias_fixtures),
            include_zoo=bool(args.bias_include_zoo),
            zoo_sizes=str(args.bias_zoo_sizes),
            zoo_n_toys=(None if args.bias_zoo_n_toys is None else int(args.bias_zoo_n_toys)),
            params=str(args.bias_params),
        )
        if bias_path is None:
            any_failed = True

    # Record ROOT suite baseline (optional)
    if args.only is None or args.only == "root":
        if args.root_suite_existing is not None or args.root_cases_existing is not None or args.root_prereq_existing is not None:
            # Register already-produced artifacts as baseline (useful for HTCondor arrays).
            if args.root_suite_existing is None:
                print("ERROR: when using existing ROOT artifacts, provide --root-suite-existing", file=sys.stderr)
                return 2
            if not args.root_suite_existing.exists():
                print(f"ERROR: missing --root-suite-existing: {args.root_suite_existing}", file=sys.stderr)
                return 2
            if args.root_cases_existing is not None and not args.root_cases_existing.exists():
                print(f"ERROR: missing --root-cases-existing: {args.root_cases_existing}", file=sys.stderr)
                return 2
            if args.root_prereq_existing is not None and not args.root_prereq_existing.exists():
                print(f"ERROR: missing --root-prereq-existing: {args.root_prereq_existing}", file=sys.stderr)
                return 2

            print("[root] Registering existing suite/cases/prereq artifacts as baseline...")
            root_suite_path = args.out_dir / f"root_suite_baseline_{stamp}.json"
            _copy_json_with_baseline_env(
                src=Path(args.root_suite_existing),
                dst=root_suite_path,
                environment=environment,
            )
            if args.root_cases_existing is not None:
                root_cases_path = args.out_dir / f"root_cases_{stamp}.json"
                root_cases_path.write_text(Path(args.root_cases_existing).read_text())
            if args.root_prereq_existing is not None:
                root_prereq_path = args.out_dir / f"root_prereq_{stamp}.json"
                _copy_json_with_baseline_env(
                    src=Path(args.root_prereq_existing),
                    dst=root_prereq_path,
                    environment=environment,
                    recorded_as_baseline=False,
                )
        elif args.root_search_dir is None:
            if args.only == "root":
                print("ERROR: --only root requires --root-search-dir (or --root-suite-existing)", file=sys.stderr)
                return 2
            print("[root] Skipping (no --root-search-dir provided).")
        else:
            root_res = record_root_suite_baseline(
                repo=repo,
                env_dict=env_dict,
                environment=environment,
                out_dir=args.out_dir,
                stamp=stamp,
                root_search_dir=Path(args.root_search_dir),
                root_glob=str(args.root_glob),
                root_include_fixtures=bool(args.root_include_fixtures),
                root_cases_absolute_paths=bool(args.root_cases_absolute_paths),
                root_mu_start=float(args.root_mu_start),
                root_mu_stop=float(args.root_mu_stop),
                root_mu_points=int(args.root_mu_points),
                root_keep_going=bool(args.root_keep_going),
                root_dq_atol=float(args.root_dq_atol),
                root_mu_hat_atol=float(args.root_mu_hat_atol),
            )
            root_prereq_path = root_res.get("prereq")
            root_cases_path = root_res.get("cases")
            root_suite_path = root_res.get("suite")
            if args.only == "root" and root_suite_path is None:
                any_failed = True

    # Write manifest linking both baselines + environment
    manifest: Dict[str, Any] = {
        "baseline_env": environment,
        "baselines": {},
    }
    if pyhf_path is not None:
        manifest["baselines"]["pyhf"] = {
            "path": str(pyhf_path),
            "filename": pyhf_path.name,
        }
    if p6_path is not None:
        manifest["baselines"]["p6_glm"] = {
            "path": str(p6_path),
            "filename": p6_path.name,
        }
    if nuts_path is not None:
        manifest["baselines"]["nuts_quality"] = {
            "path": str(nuts_path),
            "filename": nuts_path.name,
        }
    if bias_path is not None:
        manifest["baselines"]["bias_pulls"] = {
            "path": str(bias_path),
            "filename": bias_path.name,
        }
    if root_prereq_path is not None:
        manifest["baselines"]["root_prereq"] = {
            "path": str(root_prereq_path),
            "filename": Path(root_prereq_path).name,
        }
    if root_cases_path is not None:
        manifest["baselines"]["root_cases"] = {
            "path": str(root_cases_path),
            "filename": Path(root_cases_path).name,
        }
    if root_suite_path is not None:
        manifest["baselines"]["root_suite"] = {
            "path": str(root_suite_path),
            "filename": Path(root_suite_path).name,
        }

    manifest_path = args.out_dir / f"baseline_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Also write a "latest" symlink-like manifest for easy reference.
    #
    # Important: when recording only a subset (e.g. `--only root` on a cluster),
    # do not clobber an existing "full" latest manifest that might link other
    # baseline types. We still create it if it doesn't exist yet.
    latest_path = args.out_dir / "latest_manifest.json"
    if args.only is None or not latest_path.exists():
        latest_path.write_text(json.dumps(manifest, indent=2))
    else:
        print(f"NOTE: Not overwriting existing latest manifest: {latest_path}")

    # Also write per-type "latest" manifests so recording only one baseline type
    # (e.g. ROOT on a cluster) doesn't lose a stable pointer for other types.
    def _write_latest(name: str, baselines_subset: Dict[str, Any]) -> None:
        p = args.out_dir / name
        obj = {"baseline_env": environment, "baselines": baselines_subset}
        p.write_text(json.dumps(obj, indent=2))

    if pyhf_path is not None:
        _write_latest(
            "latest_pyhf_manifest.json",
            {"pyhf": {"path": str(pyhf_path), "filename": pyhf_path.name}},
        )
    if p6_path is not None:
        _write_latest(
            "latest_p6_glm_manifest.json",
            {"p6_glm": {"path": str(p6_path), "filename": p6_path.name}},
        )
    if nuts_path is not None:
        _write_latest(
            "latest_nuts_quality_manifest.json",
            {"nuts_quality": {"path": str(nuts_path), "filename": nuts_path.name}},
        )
    if bias_path is not None:
        _write_latest(
            "latest_bias_pulls_manifest.json",
            {"bias_pulls": {"path": str(bias_path), "filename": bias_path.name}},
        )
    if root_prereq_path is not None or root_cases_path is not None or root_suite_path is not None:
        subset: Dict[str, Any] = {}
        if root_prereq_path is not None:
            subset["root_prereq"] = {"path": str(root_prereq_path), "filename": Path(root_prereq_path).name}
        if root_cases_path is not None:
            subset["root_cases"] = {"path": str(root_cases_path), "filename": Path(root_cases_path).name}
        if root_suite_path is not None:
            subset["root_suite"] = {"path": str(root_suite_path), "filename": Path(root_suite_path).name}
        _write_latest("latest_root_manifest.json", subset)

    print()
    print("-" * 72)
    print(f"Manifest:  {manifest_path}")
    print(f"Latest:    {latest_path}")
    if pyhf_path:
        print(f"pyhf:      {pyhf_path}")
    if p6_path:
        print(f"p6_glm:    {p6_path}")
    if nuts_path:
        print(f"nuts:      {nuts_path}")
    if bias_path:
        print(f"bias:      {bias_path}")
    if root_suite_path:
        print(f"root:      {root_suite_path}")
    print()

    if any_failed:
        print("WARNING: Some baselines failed to record.")
        return 1

    # Print usage hint for comparing against these baselines
    print("To compare P6 GLM against this baseline:")
    if p6_path:
        print(f"  PYTHONPATH=bindings/ns-py/python {sys.executable} tests/apex2_p6_glm_benchmark_report.py \\")
        print(f"    --baseline {p6_path} \\")
        print(f"    --out tmp/apex2_p6_glm_bench_report.json")
    print()
    print("To run full master report with P6 baseline:")
    if p6_path:
        print(f"  PYTHONPATH=bindings/ns-py/python {sys.executable} tests/apex2_master_report.py \\")
        print(f"    --p6-glm-bench --p6-glm-bench-baseline {p6_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
