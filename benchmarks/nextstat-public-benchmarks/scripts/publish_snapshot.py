#!/usr/bin/env python3
"""Publish a benchmark snapshot locally (seed script).

Creates an immutable snapshot directory containing:
- suite outputs (e.g. `hep/hep_suite.json` + per-case results)
- a baseline manifest (`baseline_manifest.json`)
- schema validation (fails fast on invalid artifacts)

This is meant to be runnable by outsiders in the standalone benchmarks repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path


def git_short_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_artifact(path: Path, repo_root: Path, env: dict[str, str]) -> None:
    subprocess.check_call(
        [sys.executable, "scripts/validate_artifacts.py", "--strict", str(path)],
        cwd=str(repo_root),
        env=env,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", default="", help="Snapshot id (default: auto-generated).")
    ap.add_argument("--out-root", default="manifests/snapshots", help="Root directory for snapshots.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Fast mode for expensive suites (where supported).")
    ap.add_argument(
        "--suite",
        default="public-benchmarks-seed",
        help="Suite label recorded in snapshot_index.json (for discovery).",
    )
    ap.add_argument(
        "--nextstat-wheel",
        default="",
        help="Optional: path to the measured NextStat wheel (hashed into baseline_manifest.json).",
    )
    ap.add_argument("--hep", action="store_true", help="Run HEP suite.")
    ap.add_argument("--pharma", action="store_true", help="Run pharma suite.")
    ap.add_argument("--econometrics", action="store_true", help="Run econometrics suite.")
    ap.add_argument("--bayesian", action="store_true", help="Run Bayesian suite.")
    ap.add_argument("--glm", action="store_true", help="Run GLM suite.")
    ap.add_argument("--survival", action="store_true", help="Run survival suite.")
    ap.add_argument("--timeseries", action="store_true", help="Run timeseries suite.")
    ap.add_argument("--evt", action="store_true", help="Run EVT suite.")
    ap.add_argument("--insurance", action="store_true", help="Run insurance suite.")
    ap.add_argument("--meta-analysis", action="store_true", help="Run meta-analysis suite.")
    ap.add_argument("--mams", action="store_true", help="Run MAMS suite.")
    ap.add_argument("--montecarlo-safety", action="store_true", help="Run Monte Carlo safety suite.")
    ap.add_argument(
        "--montecarlo-device",
        default="cpu",
        help="Monte Carlo safety device(s): cpu, cuda, or comma-separated (e.g. cpu,cuda).",
    )
    ap.add_argument(
        "--bayesian-backends",
        default="nextstat,nextstat_dense",
        help="Bayesian suite backends (comma-separated): nextstat,nextstat_dense,cmdstanpy,pymc (optional).",
    )
    ap.add_argument("--bayesian-n-chains", type=int, default=4)
    ap.add_argument("--bayesian-warmup", type=int, default=500)
    ap.add_argument("--bayesian-samples", type=int, default=1000)
    ap.add_argument("--bayesian-seed", type=int, default=0)
    ap.add_argument("--bayesian-max-treedepth", type=int, default=10)
    ap.add_argument("--bayesian-target-accept", type=float, default=0.8)
    ap.add_argument("--bayesian-init-jitter-rel", type=float, default=0.10)
    ap.add_argument("--ml", action="store_true", help="Run ML suite.")
    ap.add_argument("--fit", action="store_true", help="Also benchmark MLE fits where supported.")
    ap.add_argument("--fit-repeat", type=int, default=3)
    ap.add_argument(
        "--extra-pythonpath",
        action="append",
        default=[],
        help="Optional: prepend an extra path to PYTHONPATH for suite subprocesses (repeatable).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    snapshot_id = str(args.snapshot_id).strip()
    if not snapshot_id:
        ts = time.strftime("%Y%m%d-%H%M%S")
        snapshot_id = f"snapshot-{ts}-{git_short_sha(repo_root)}"

    snap_dir = out_root / snapshot_id
    if snap_dir.exists():
        raise SystemExit(f"snapshot dir already exists: {snap_dir}")
    snap_dir.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    extra_pp = [str(p).strip() for p in (args.extra_pythonpath or []) if str(p).strip()]
    if extra_pp:
        # Preserve order: user-provided extra paths first.
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = ":".join(extra_pp + ([existing] if existing else []))

    # Optional: copy the measured NextStat wheel into the snapshot directory so DOI/public
    # snapshots can be fully self-contained.
    nextstat_wheel_path = str(args.nextstat_wheel).strip()
    if nextstat_wheel_path:
        wheel_src = Path(nextstat_wheel_path).resolve()
        if not wheel_src.exists():
            raise SystemExit(f"nextstat wheel not found: {wheel_src}")
        wheel_dst = snap_dir / "nextstat_wheel.whl"
        if wheel_dst.exists():
            if wheel_dst.resolve() != wheel_src:
                raise SystemExit(f"snapshot already contains nextstat_wheel.whl: {wheel_dst}")
        else:
            shutil.copy2(wheel_src, wheel_dst)
        nextstat_wheel_path = str(wheel_dst)

    results: list[Path] = []

    # Defaults: keep the seed publisher lightweight (HEP + pharma) unless a suite is explicitly requested.
    run_any = bool(
        args.hep
        or args.pharma
        or args.econometrics
        or args.bayesian
        or args.ml
        or args.glm
        or args.survival
        or args.timeseries
        or args.evt
        or args.insurance
        or args.meta_analysis
        or args.mams
        or args.montecarlo_safety
    )
    run_hep = bool(args.hep or (not run_any))
    run_pharma = bool(args.pharma or (not run_any))
    run_econometrics = bool(args.econometrics)
    run_bayesian = bool(args.bayesian)
    run_ml = bool(args.ml)
    run_glm = bool(args.glm)
    run_survival = bool(args.survival)
    run_timeseries = bool(args.timeseries)
    run_evt = bool(args.evt)
    run_insurance = bool(args.insurance)
    run_meta = bool(args.meta_analysis)
    run_mams = bool(args.mams)
    run_mc = bool(args.montecarlo_safety)

    if run_hep:
        hep_out = snap_dir / "hep"
        cmd = [
            sys.executable,
            "suites/hep/suite.py",
            "--out-dir",
            str(hep_out),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.fit:
            cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(hep_out / "hep_suite.json")
        # Generate a small README snippet for humans (does not replace machine JSONs).
        subprocess.check_call(
            [
                sys.executable,
                "suites/hep/report.py",
                "--suite",
                str(hep_out / "hep_suite.json"),
                "--format",
                "markdown",
                "--out",
                str(snap_dir / "README_snippet.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    if run_pharma:
        pharma_out = snap_dir / "pharma"
        cmd = [
            sys.executable,
            "suites/pharma/suite.py",
            "--out-dir",
            str(pharma_out),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.fit:
            cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(pharma_out / "pharma_suite.json")
        subprocess.check_call(
            [
                sys.executable,
                "suites/pharma/report.py",
                "--suite",
                str(pharma_out / "pharma_suite.json"),
                "--out",
                str(snap_dir / "README_snippet_pharma.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    if run_econometrics:
        econ_out = snap_dir / "econometrics"
        cmd = [
            sys.executable,
            "suites/econometrics/suite.py",
            "--out-dir",
            str(econ_out),
            "--seed",
            "0",
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(econ_out / "econometrics_suite.json")
        subprocess.check_call(
            [
                sys.executable,
                "suites/econometrics/report.py",
                "--suite",
                str(econ_out / "econometrics_suite.json"),
                "--out",
                str(snap_dir / "README_snippet_econometrics.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    if run_bayesian:
        bayes_out = snap_dir / "bayesian"
        cmd = [
            sys.executable,
            "suites/bayesian/suite.py",
            "--out-dir",
            str(bayes_out),
            "--backends",
            str(args.bayesian_backends),
            "--n-chains",
            str(int(args.bayesian_n_chains)),
            "--warmup",
            str(int(args.bayesian_warmup)),
            "--samples",
            str(int(args.bayesian_samples)),
            "--seed",
            str(int(args.bayesian_seed)),
            "--max-treedepth",
            str(int(args.bayesian_max_treedepth)),
            "--target-accept",
            str(float(args.bayesian_target_accept)),
            "--init-jitter-rel",
            str(float(args.bayesian_init_jitter_rel)),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(bayes_out / "bayesian_suite.json")
        subprocess.check_call(
            [
                sys.executable,
                "suites/bayesian/report.py",
                "--suite",
                str(bayes_out / "bayesian_suite.json"),
                "--out",
                str(snap_dir / "README_snippet_bayesian.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    if run_ml:
        ml_out = snap_dir / "ml"
        cmd = [
            sys.executable,
            "suites/ml/suite.py",
            "--out-dir",
            str(ml_out),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(ml_out / "ml_suite.json")
        subprocess.check_call(
            [
                sys.executable,
                "suites/ml/report.py",
                "--suite",
                str(ml_out / "ml_suite.json"),
                "--out",
                str(snap_dir / "README_snippet_ml.md"),
            ],
            cwd=str(repo_root),
            env=env,
        )

    if run_glm:
        glm_out = snap_dir / "glm"
        cmd = [sys.executable, "suites/glm/suite.py", "--out-dir", str(glm_out)]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(glm_out / "glm_suite.json")

    if run_survival:
        surv_out = snap_dir / "survival"
        cmd = [sys.executable, "suites/survival/suite.py", "--out-dir", str(surv_out)]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.smoke:
            cmd.append("--smoke")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(surv_out / "survival_suite.json")

    if run_timeseries:
        ts_out = snap_dir / "timeseries"
        cmd = [sys.executable, "suites/timeseries/suite.py", "--out-dir", str(ts_out)]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.smoke:
            cmd.append("--smoke")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(ts_out / "timeseries_suite.json")

    if run_evt:
        evt_out = snap_dir / "evt"
        cmd = [sys.executable, "suites/evt/suite.py", "--out-dir", str(evt_out)]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(evt_out / "evt_suite.json")

    if run_insurance:
        ins_out = snap_dir / "insurance"
        cmd = [sys.executable, "suites/insurance/suite.py", "--out-dir", str(ins_out)]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(ins_out / "insurance_suite.json")

    if run_meta:
        meta_out = snap_dir / "meta_analysis"
        cmd = [sys.executable, "suites/meta_analysis/suite.py", "--out-dir", str(meta_out)]
        if args.deterministic:
            cmd.append("--deterministic")
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(meta_out / "meta_analysis_suite.json")

    if run_mams:
        mams_out = snap_dir / "mams"
        cmd = [sys.executable, "suites/mams/suite.py", "--out-dir", str(mams_out)]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.smoke:
            cmd.append("--smoke")
            # Smoke snapshots should be fast and baseline-optional. BlackJAX can be very slow
            # on CPU runners due to JIT + dependency weight, so keep smoke MAMS to NextStat-only.
            cmd.extend(["--backends", "nextstat_mams,nextstat_nuts"])
        subprocess.check_call(cmd, cwd=str(repo_root), env=env)
        results.append(mams_out / "mams_suite.json")

    if run_mc:
        mc_out = snap_dir / "montecarlo_safety"
        devices = [x.strip() for x in str(args.montecarlo_device).split(",") if x.strip()]
        for dev in devices:
            if dev not in {"cpu", "cuda"}:
                raise SystemExit(f"unsupported montecarlo device: {dev!r} (expected cpu|cuda)")
            cmd = [
                sys.executable,
                "suites/montecarlo_safety/suite.py",
                "--out-dir",
                str(mc_out),
                "--device",
                dev,
            ]
            if args.deterministic:
                cmd.append("--deterministic")
            if args.smoke:
                cmd.append("--smoke")
            subprocess.check_call(cmd, cwd=str(repo_root), env=env)
            if dev == "cpu":
                results.append(mc_out / "montecarlo_safety_suite.json")
            else:
                results.append(mc_out / f"montecarlo_safety_suite_{dev}.json")

        # Human-facing report (optional; does not participate in schema validation).
        subprocess.check_call(
            [
                sys.executable,
                "suites/montecarlo_safety/report.py",
                str(mc_out),
            ],
            cwd=str(repo_root),
            env=env,
        )

    # Validate suite + case schemas.
    for r in results:
        validate_artifact(r, repo_root, env)

    # Write baseline manifest referencing produced results; dataset list will be derived from results.
    baseline_path = snap_dir / "baseline_manifest.json"
    cmd = [
        sys.executable,
        "scripts/write_baseline_manifest.py",
        "--snapshot-id",
        snapshot_id,
        "--out",
        str(baseline_path),
    ]
    if args.deterministic:
        cmd.append("--deterministic")
    if nextstat_wheel_path:
        cmd.extend(["--nextstat-wheel", nextstat_wheel_path])
    for r in results:
        cmd.extend(["--result", str(r)])
    subprocess.check_call(cmd, cwd=str(repo_root), env=env)
    validate_artifact(baseline_path, repo_root, env)

    # Write an index for the full snapshot artifact set (hash inventory).
    index_path = snap_dir / "snapshot_index.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/write_snapshot_index.py",
            "--suite",
            str(args.suite),
            "--artifacts-dir",
            str(snap_dir),
            "--out",
            str(index_path),
            "--snapshot-id",
            snapshot_id,
        ],
        cwd=str(repo_root),
        env=env,
    )
    validate_artifact(index_path, repo_root, env)

    print(f"snapshot ready: {snap_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
