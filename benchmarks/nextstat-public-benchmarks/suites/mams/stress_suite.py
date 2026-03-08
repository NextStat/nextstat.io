#!/usr/bin/env python3
"""Expanded MAMS stress suite orchestrator.

This lane is distinct from the canonical MAMS promotion suite. It extends the
evidence envelope with additional pathological and realistic hierarchical cases
while keeping canonical promotion claims unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


CASE_CATALOG: list[dict[str, str]] = [
    {
        "case": "neal_funnel_10d_centered",
        "case_tier": "pathological_control",
        "parity_scope": "informational",
        "description": "Centered 10D funnel hard-geometry control.",
    },
    {
        "case": "neal_funnel_ncp_10d",
        "case_tier": "supported",
        "parity_scope": "required",
        "description": "Non-centered 10D funnel supported repeatability case.",
    },
    {
        "case": "hier_random_intercept_non_centered",
        "case_tier": "supported",
        "parity_scope": "required",
        "description": "Hierarchical logistic random intercept stress case.",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _case_meta(case_id: str) -> dict[str, str]:
    for row in CASE_CATALOG:
        if row["case"] == case_id:
            return row
    raise KeyError(case_id)


def _write_stub_failed(
    out_path: Path,
    *,
    case: str,
    backend: str,
    seed: int,
    reason: str,
    n_chains: int,
    warmup: int,
    samples: int,
    dataset_seed: int,
    target_accept: float,
    n_groups: int,
    n_per_group: int,
    status: str = "failed",
) -> None:
    stub = {
        "schema_version": "nextstat.mams_benchmark_result.v1",
        "suite": "mams",
        "case": case,
        "backend": backend,
        "deterministic": False,
        "status": status,
        "reason": reason,
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": "unknown",
        },
        "dataset": {"id": f"generated:mams_stress:{case}", "sha256": "0" * 64},
        "config": {
            "n_chains": int(n_chains),
            "n_warmup": int(warmup),
            "n_samples": int(samples),
            "seed": int(seed),
            "dataset_seed": int(dataset_seed),
            "target_accept": float(target_accept),
            "n_groups": int(n_groups) if case == "hier_random_intercept_non_centered" else None,
            "n_per_group": int(n_per_group) if case == "hier_random_intercept_non_centered" else None,
        },
        "timing": {"wall_time_s": 0.0},
        "metrics": {},
    }
    out_path.write_text(json.dumps(stub, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Expanded MAMS stress suite orchestrator")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--backends", default="nextstat_mams,nextstat_nuts", help="Comma-separated backends")
    ap.add_argument("--seeds", default="42", help="Comma-separated seeds")
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3500)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--dataset-seed", type=int, default=12345)
    ap.add_argument("--target-accept", type=float, default=0.985)
    ap.add_argument("--n-groups", type=int, default=20)
    ap.add_argument("--n-per-group", type=int, default=20)
    ap.add_argument("--run-timeout-s", type=float, default=300.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--parity-warn-z", type=float, default=8.0)
    ap.add_argument("--parity-fail-z", type=float, default=12.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    try:
        import nextstat  # type: ignore
        ns_version = str(getattr(nextstat, "__version__", "unknown"))
    except Exception:
        ns_version = "unknown"

    run_py = Path(__file__).resolve().parent / "run.py"
    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    index_cases: list[dict[str, Any]] = []
    n_ok = 0
    n_warn = 0
    n_failed = 0

    for case_meta in CASE_CATALOG:
        case = case_meta["case"]
        for backend in backends:
            for seed in seeds:
                tag = f"{case}__{backend}__s{seed}"
                out_path = cases_dir / f"{tag}.json"
                cmd = [
                    sys.executable,
                    str(run_py),
                    "--case", case,
                    "--backend", backend,
                    "--out", str(out_path),
                    "--n-chains", str(int(args.n_chains)),
                    "--warmup", str(int(args.warmup)),
                    "--samples", str(int(args.samples)),
                    "--seed", str(int(seed)),
                    "--dataset-seed", str(int(args.dataset_seed)),
                    "--target-accept", str(float(args.target_accept)),
                    "--n-groups", str(int(args.n_groups)),
                    "--n-per-group", str(int(args.n_per_group)),
                ]
                if args.deterministic:
                    cmd.append("--deterministic")

                print(f"  [{tag}] running...", end=" ", flush=True)
                timeout_s = float(args.run_timeout_s)
                timed_out = False
                try:
                    p = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout_s if timeout_s > 0 else None,
                    )
                except subprocess.TimeoutExpired as e:
                    timed_out = True
                    p = subprocess.CompletedProcess(cmd, 124, "", f"timeout after {timeout_s:.1f}s: {e}")

                if not out_path.exists():
                    _write_stub_failed(
                        out_path,
                        case=case,
                        backend=backend,
                        seed=seed,
                        reason=(
                            f"case runner timed out after {timeout_s:.1f}s"
                            if timed_out
                            else f"case runner failed (exit={p.returncode}): {p.stderr.strip()[:4000]}"
                        ),
                        n_chains=int(args.n_chains),
                        warmup=int(args.warmup),
                        samples=int(args.samples),
                        dataset_seed=int(args.dataset_seed),
                        target_accept=float(args.target_accept),
                        n_groups=int(args.n_groups),
                        n_per_group=int(args.n_per_group),
                        status="warn" if timed_out else "failed",
                    )

                try:
                    obj = json.loads(out_path.read_text())
                except Exception:
                    _write_stub_failed(
                        out_path,
                        case=case,
                        backend=backend,
                        seed=seed,
                        reason="invalid JSON output from case runner",
                        n_chains=int(args.n_chains),
                        warmup=int(args.warmup),
                        samples=int(args.samples),
                        dataset_seed=int(args.dataset_seed),
                        target_accept=float(args.target_accept),
                        n_groups=int(args.n_groups),
                        n_per_group=int(args.n_per_group),
                    )
                    obj = json.loads(out_path.read_text())

                status = str(obj.get("status", "failed"))
                if status == "ok":
                    n_ok += 1
                elif status == "warn":
                    n_warn += 1
                else:
                    n_failed += 1

                metrics = obj.get("metrics", {})
                sha = sha256_file(out_path) if out_path.exists() else "0" * 64
                index_cases.append(
                    {
                        "case": case,
                        "case_tier": case_meta["case_tier"],
                        "backend": backend,
                        "seed": seed,
                        "path": os.path.relpath(out_path, out_dir),
                        "sha256": sha,
                        "status": status,
                        "wall_time_s": _safe_float(metrics.get("wall_time_s")),
                        "n_grad_evals": metrics.get("n_grad_evals"),
                        "min_ess_bulk": _safe_float(metrics.get("min_ess_bulk")),
                        "ess_per_grad": _safe_float(metrics.get("ess_per_grad")),
                        "ess_per_sec": _safe_float(metrics.get("ess_per_sec")),
                        "max_r_hat": _safe_float(metrics.get("max_r_hat")),
                    }
                )
                print(f"{status} (wall={_safe_float(metrics.get('wall_time_s'))}s)")

    def _load_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def _params_map(run_obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
        ps = ((run_obj.get("metrics") or {}).get("posterior_summary") or {})
        if not isinstance(ps, dict) or ps.get("status") != "ok":
            return {}
        params = ps.get("params")
        if not isinstance(params, list):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for row in params:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                out[str(row["name"])] = row
        return out

    def _mc_se(row: dict[str, Any], fallback_n: int) -> float | None:
        mean = _safe_float(row.get("mean"))
        sd = _safe_float(row.get("sd"))
        if mean is None or sd is None:
            return None
        ess = _safe_float(row.get("ess_bulk"))
        n_eff = ess if ess and ess > 1.0 else float(fallback_n)
        if n_eff <= 1.0:
            return None
        return sd / math.sqrt(n_eff)

    lookup: dict[tuple[str, int, str], Path] = {}
    for entry in index_cases:
        lookup[(str(entry["case"]), int(entry["seed"]), str(entry["backend"]))] = out_dir / str(entry["path"])

    parity_rows: list[dict[str, Any]] = []
    n_parity_warn = 0
    n_parity_fail = 0
    for case_meta in CASE_CATALOG:
        case = case_meta["case"]
        for seed in seeds:
            p_mams = lookup.get((case, seed, "nextstat_mams"))
            p_nuts = lookup.get((case, seed, "nextstat_nuts"))
            if not p_mams or not p_nuts or not p_mams.exists() or not p_nuts.exists():
                continue

            jm = _load_json(p_mams)
            jn = _load_json(p_nuts)
            if jm.get("status") != "ok" or jn.get("status") != "ok":
                continue

            pm = _params_map(jm)
            pn = _params_map(jn)
            if not pm or not pn:
                parity_rows.append(
                    {
                        "case": case,
                        "case_tier": case_meta["case_tier"],
                        "parity_scope": case_meta["parity_scope"],
                        "seed": seed,
                        "status": "warn",
                        "reason": "missing_posterior_summary",
                        "max_z": None,
                        "worst": [],
                    }
                )
                n_parity_warn += 1
                continue

            fallback_n = int(args.n_chains) * int(args.samples)
            worst: list[tuple[float, str]] = []
            max_z: float | None = None
            for name in sorted(set(pm.keys()) & set(pn.keys())):
                mm = _safe_float(pm[name].get("mean"))
                mn = _safe_float(pn[name].get("mean"))
                if mm is None or mn is None:
                    continue
                se_m = _mc_se(pm[name], fallback_n)
                se_n = _mc_se(pn[name], fallback_n)
                if se_m is None or se_n is None:
                    continue
                denom = math.sqrt(se_m * se_m + se_n * se_n)
                if denom <= 0:
                    continue
                z = abs(mm - mn) / denom
                if max_z is None or z > max_z:
                    max_z = z
                worst.append((z, name))

            worst.sort(reverse=True)
            status = "ok"
            if max_z is None:
                status = "warn"
            elif max_z >= float(args.parity_fail_z):
                status = "failed"
            elif max_z >= float(args.parity_warn_z):
                status = "warn"
            if case_meta["parity_scope"] == "required":
                if status == "warn":
                    n_parity_warn += 1
                elif status == "failed":
                    n_parity_fail += 1

            parity_rows.append(
                {
                    "case": case,
                    "case_tier": case_meta["case_tier"],
                    "parity_scope": case_meta["parity_scope"],
                    "seed": seed,
                    "status": status,
                    "max_z": float(max_z) if max_z is not None else None,
                    "worst": [{"param": name, "z": float(z)} for z, name in worst[:3]],
                }
            )

    index = {
        "schema_version": "nextstat.mams_stress_benchmark_suite_result.v1",
        "suite": "mams_stress",
        "deterministic": bool(args.deterministic),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": ns_version,
        },
        "config": {
            "n_chains": int(args.n_chains),
            "n_warmup": int(args.warmup),
            "n_samples": int(args.samples),
            "dataset_seed": int(args.dataset_seed),
            "target_accept": float(args.target_accept),
            "n_groups": int(args.n_groups),
            "n_per_group": int(args.n_per_group),
            "seeds": seeds,
            "backends": backends,
            "parity_warn_z": float(args.parity_warn_z),
            "parity_fail_z": float(args.parity_fail_z),
        },
        "case_catalog": CASE_CATALOG,
        "cases": index_cases,
        "summary": {
            "n_total": len(index_cases),
            "n_ok": n_ok,
            "n_warn": n_warn,
            "n_failed": n_failed,
            "n_parity_warn": n_parity_warn,
            "n_parity_fail": n_parity_fail,
        },
        "parity": {
            "method": "mean_zscore",
            "note": "Compares NextStat MAMS vs NextStat NUTS posterior means per case/seed; supported cases require parity, pathological controls are informational only.",
            "warn_z": float(args.parity_warn_z),
            "fail_z": float(args.parity_fail_z),
            "rows": parity_rows,
        },
    }

    index_path = out_dir / "mams_stress_suite.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"\nStress suite index: {index_path}")
    print(
        f"Total: {len(index_cases)} runs — {n_ok} ok, {n_warn} warn, {n_failed} failed "
        f"(parity: {n_parity_warn} warn, {n_parity_fail} failed)"
    )
    return 0 if (n_failed == 0 and n_parity_fail == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
