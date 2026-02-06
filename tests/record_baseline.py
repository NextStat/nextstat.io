#!/usr/bin/env python3
"""Record reference baseline JSONs for Apex2 regression testing.

This script captures a full environment fingerprint and runs the pyhf
validation + P6 GLM fit/predict benchmarks, saving the results as
timestamped baseline files under ``tmp/baselines/``.

Usage (from repo root, after ``maturin develop --release``):

  # Record both baselines (default):
  PYTHONPATH=bindings/ns-py/python .venv/bin/python tests/record_baseline.py

  # Record only pyhf baseline:
  PYTHONPATH=bindings/ns-py/python .venv/bin/python tests/record_baseline.py --only pyhf

  # Record only P6 GLM baseline:
  PYTHONPATH=bindings/ns-py/python .venv/bin/python tests/record_baseline.py --only p6

  # Custom output directory:
  PYTHONPATH=bindings/ns-py/python .venv/bin/python tests/record_baseline.py --out-dir tmp/baselines

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
        choices=["pyhf", "p6"],
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
    args = ap.parse_args()

    repo = _repo_root()
    env_dict = _with_py_path(os.environ.copy())

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

    manifest_path = args.out_dir / f"baseline_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Also write a "latest" symlink-like manifest for easy reference
    latest_path = args.out_dir / "latest_manifest.json"
    latest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print("-" * 72)
    print(f"Manifest:  {manifest_path}")
    print(f"Latest:    {latest_path}")
    if pyhf_path:
        print(f"pyhf:      {pyhf_path}")
    if p6_path:
        print(f"p6_glm:    {p6_path}")
    print()

    if any_failed:
        print("WARNING: Some baselines failed to record.")
        return 1

    # Print usage hint for comparing against these baselines
    print("To compare P6 GLM against this baseline:")
    if p6_path:
        print(f"  python tests/apex2_p6_glm_benchmark_report.py \\")
        print(f"    --baseline {p6_path} \\")
        print(f"    --out tmp/apex2_p6_glm_bench_report.json")
    print()
    print("To run full master report with P6 baseline:")
    if p6_path:
        print(f"  python tests/apex2_master_report.py \\")
        print(f"    --p6-glm-bench --p6-glm-bench-baseline {p6_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
