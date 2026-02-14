#!/usr/bin/env python3
"""MAMS suite orchestrator.

Runs all cases × backends, collects JSON artifacts, writes a suite index.

Usage:
    python suite.py --out-dir results_v1
    python suite.py --out-dir results_v1 --seeds 0,42,123
    python suite.py --out-dir results_v1 --backends nextstat_mams,nextstat_nuts

Parity checks (NextStat MAMS vs NextStat NUTS):
    python suite.py --out-dir results_v1 --parity-warn-z 8 --parity-fail-z 12
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="MAMS benchmark suite orchestrator")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--backends", default="nextstat_mams,nextstat_nuts",
                    help="Comma-separated backends")
    ap.add_argument("--seeds", default="42", help="Comma-separated seeds")
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Fast mode: fewer chains/draws and reduced case set.",
    )
    ap.add_argument(
        "--parity-warn-z",
        type=float,
        default=8.0,
        help="Warn if MAMS vs NUTS max z-score exceeds this (posterior mean parity).",
    )
    ap.add_argument(
        "--parity-fail-z",
        type=float,
        default=12.0,
        help="Fail suite if MAMS vs NUTS max z-score exceeds this (posterior mean parity).",
    )
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
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Smoke mode exists so snapshot publishing can stay <10 minutes on a CPU runner.
    # We always keep at least one "easy" and one "pathological" geometry case.
    if args.smoke:
        args.n_chains = min(int(args.n_chains), 2)
        args.warmup = min(int(args.warmup), 200)
        args.samples = min(int(args.samples), 400)
        cases = ["std_normal_10d", "neal_funnel_2d", "glm_logistic"]
    else:
        cases = ["std_normal_10d", "neal_funnel_2d", "eight_schools", "glm_logistic"]

    index_cases: list[dict] = []
    n_ok = 0
    n_warn = 0
    n_failed = 0

    def _write_stub_failed(out_path: Path, *, case: str, backend: str, seed: int, reason: str) -> None:
        stub = {
            "schema_version": "nextstat.mams_benchmark_result.v1",
            "suite": "mams",
            "case": case,
            "backend": backend,
            "deterministic": bool(args.deterministic),
            "status": "failed",
            "reason": reason,
            "meta": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "nextstat_version": ns_version,
            },
            "dataset": {"id": f"generated:mams:{case}", "sha256": "0" * 64},
            "config": {
                "n_chains": int(args.n_chains),
                "n_warmup": int(args.warmup),
                "n_samples": int(args.samples),
                "seed": int(seed),
                "target_accept": float(args.target_accept),
            },
            "timing": {"wall_time_s": 0.0},
            "metrics": {},
        }
        out_path.write_text(json.dumps(stub, indent=2, sort_keys=True) + "\n")

    for case in cases:
        for backend in backends:
            for seed in seeds:
                tag = f"{case}__{backend}__s{seed}"
                out_path = cases_dir / f"{tag}.json"

                cmd = [
                    sys.executable, str(run_py),
                    "--case", case,
                    "--backend", backend,
                    "--out", str(out_path),
                    "--n-chains", str(args.n_chains),
                    "--warmup", str(args.warmup),
                    "--samples", str(args.samples),
                    "--seed", str(seed),
                    "--target-accept", str(args.target_accept),
                ]
                if args.deterministic:
                    cmd.append("--deterministic")

                print(f"  [{tag}] running...", end=" ", flush=True)
                p = subprocess.run(cmd, capture_output=True, text=True)

                if not out_path.exists():
                    _write_stub_failed(
                        out_path,
                        case=case,
                        backend=backend,
                        seed=seed,
                        reason=f"case runner failed (exit={p.returncode}): {p.stderr.strip()[:4000]}",
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

                entry = {
                    "case": case,
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
                index_cases.append(entry)
                print(f"{status} (wall={_safe_float(metrics.get('wall_time_s'))}s)")

    # ------------------------------------------------------------------
    # Cross-sampler parity: NextStat MAMS vs NextStat NUTS on same case/seed.
    # Uses posterior mean z-scores with MC SE estimated via ESS_bulk.
    # ------------------------------------------------------------------

    def _load_json(p: Path) -> dict:
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def _params_map(run_obj: dict) -> dict[str, dict]:
        ps = ((run_obj.get("metrics") or {}).get("posterior_summary") or {})
        if not isinstance(ps, dict):
            return {}
        if ps.get("status") != "ok":
            return {}
        params = ps.get("params")
        if not isinstance(params, list):
            return {}
        out: dict[str, dict] = {}
        for row in params:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                out[row["name"]] = row
        return out

    def _mc_se(row: dict, fallback_n: int) -> float | None:
        mean = _safe_float(row.get("mean"))
        sd = _safe_float(row.get("sd"))
        if mean is None or sd is None:
            return None
        ess = _safe_float(row.get("ess_bulk"))
        n_eff = ess if ess and ess > 1.0 else float(fallback_n)
        if n_eff <= 1.0:
            return None
        return sd / math.sqrt(n_eff)

    # Build quick lookup: (case, seed, backend) -> path
    lookup: dict[tuple[str, int, str], Path] = {}
    for e in index_cases:
        c = str(e.get("case", ""))
        b = str(e.get("backend", ""))
        s = int(e.get("seed", 0))
        p = out_dir / str(e.get("path", ""))
        lookup[(c, s, b)] = p

    parity_rows: list[dict] = []
    n_parity_warn = 0
    n_parity_fail = 0
    for case in sorted(set(str(e.get("case")) for e in index_cases)):
        for seed in sorted(set(int(e.get("seed", 0)) for e in index_cases if str(e.get("case")) == case)):
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
            worst_top = [{"param": n, "z": float(z)} for z, n in worst[:3]]

            status = "ok"
            if max_z is None:
                status = "warn"
                n_parity_warn += 1
            elif max_z >= float(args.parity_fail_z):
                status = "failed"
                n_parity_fail += 1
            elif max_z >= float(args.parity_warn_z):
                status = "warn"
                n_parity_warn += 1

            parity_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "status": status,
                    "max_z": float(max_z) if max_z is not None else None,
                    "worst": worst_top,
                }
            )

    index = {
        "schema_version": "nextstat.mams_benchmark_suite_result.v1",
        "suite": "mams",
        "deterministic": bool(args.deterministic),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": ns_version,
        },
        "config": {
            "n_chains": args.n_chains,
            "n_warmup": args.warmup,
            "n_samples": args.samples,
            "target_accept": args.target_accept,
            "seeds": seeds,
            "backends": backends,
            "smoke": bool(args.smoke),
            "parity_warn_z": float(args.parity_warn_z),
            "parity_fail_z": float(args.parity_fail_z),
        },
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
            "note": "Compares NextStat MAMS vs NextStat NUTS posterior means per case/seed; SE uses ESS_bulk when available.",
            "warn_z": float(args.parity_warn_z),
            "fail_z": float(args.parity_fail_z),
            "rows": parity_rows,
        },
    }

    index_path = out_dir / "mams_suite.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"\nSuite index: {index_path}")
    print(
        f"Total: {len(index_cases)} runs — {n_ok} ok, {n_warn} warn, {n_failed} failed "
        f"(parity: {n_parity_warn} warn, {n_parity_fail} failed)"
    )

    return 0 if (n_failed == 0 and n_parity_fail == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
