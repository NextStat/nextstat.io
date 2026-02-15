#!/usr/bin/env python3
"""Run the Bayesian suite across multiple seeds and aggregate results.

This is a convenience wrapper around `suites/bayesian/suite.py` that:
- creates one subdirectory per seed (schema-compatible artifacts per run)
- emits an aggregated Markdown summary for quick comparison/stability checks

It does not change the per-run JSON schemas; it runs the suite multiple times.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _fmt(x) -> str:
    v = _safe_float(x)
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Base output directory.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--backends",
        default="nextstat",
        help="Comma-separated list of backends: nextstat,cmdstanpy,pymc,numpyro (optional).",
    )
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seeds", default="42,0,123", help="Comma-separated seed list.")
    ap.add_argument(
        "--dataset-seed",
        type=int,
        default=12345,
        help="Seed used for generated datasets (kept fixed across chain seeds).",
    )
    ap.add_argument("--max-treedepth", type=int, default=10)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--init-jitter-rel", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_py = Path(__file__).resolve().parent / "suite.py"
    seeds: list[int] = []
    for tok in str(args.seeds).split(","):
        tok = tok.strip()
        if not tok:
            continue
        seeds.append(int(tok))
    if not seeds:
        raise SystemExit("no seeds provided")

    # Run suite once per seed (schema artifacts preserved per run).
    rc = 0
    per_seed_suite_paths: dict[int, Path] = {}
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        cmd = [
            sys.executable,
            str(suite_py),
            "--out-dir",
            str(seed_dir),
            "--backends",
            str(args.backends),
            "--n-chains",
            str(int(args.n_chains)),
            "--warmup",
            str(int(args.warmup)),
            "--samples",
            str(int(args.samples)),
            "--seed",
            str(int(seed)),
            "--dataset-seed",
            str(int(args.dataset_seed)),
            "--max-treedepth",
            str(int(args.max_treedepth)),
            "--target-accept",
            str(float(args.target_accept)),
            "--init-jitter-rel",
            str(float(args.init_jitter_rel)),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        p = subprocess.run(cmd)
        if p.returncode != 0:
            rc = 2

        suite_path = seed_dir / "bayesian_suite.json"
        if suite_path.exists():
            per_seed_suite_paths[seed] = suite_path

    # Aggregate across seeds: key = (case, backend).
    by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    status_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    config_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for seed, suite_path in per_seed_suite_paths.items():
        obj = json.loads(suite_path.read_text())
        for c in (obj.get("cases") if isinstance(obj.get("cases"), list) else []):
            case_id = str(c.get("case") or "unknown")
            backend = str(c.get("backend") or "unknown")
            key = (case_id, backend)
            status_by_key[key].append(str(c.get("status") or "unknown"))

            # Pull the referenced per-case JSON to capture the config (warmup/samples/seed).
            rel = c.get("path")
            if isinstance(rel, str) and rel:
                case_path = (suite_path.parent / rel).resolve()
                if case_path.exists():
                    try:
                        case_obj = json.loads(case_path.read_text())
                        if isinstance(case_obj.get("config"), dict):
                            config_by_key[key].append(case_obj["config"])
                    except Exception:
                        pass

            for metric_key, out_key in [
                ("wall_time_s", "wall_time_s"),
                ("min_ess_bulk", "min_ess_bulk"),
                ("min_ess_tail", "min_ess_tail"),
                ("max_r_hat", "max_r_hat"),
                ("min_ess_bulk_per_sec", "min_ess_bulk_per_sec"),
            ]:
                v = _safe_float(c.get(metric_key))
                if v is not None:
                    by_key[key][out_key].append(v)

    # Render Markdown summary.
    md_lines: list[str] = []
    md_lines.append("# Bayesian suite (multi-seed summary)")
    md_lines.append("")
    md_lines.append(f"- Seeds: `{', '.join(str(s) for s in seeds)}`")
    md_lines.append(f"- Backends: `{args.backends}`")
    md_lines.append(
        f"- Config: `chains={args.n_chains}`, `warmup={args.warmup}`, `samples={args.samples}`, "
        f"`max_treedepth={args.max_treedepth}`, `target_accept={args.target_accept}`, `init_jitter_rel={args.init_jitter_rel}`"
    )
    md_lines.append("")
    md_lines.append("Metrics are aggregated across seeds as mean ± std (where available).")
    md_lines.append("")

    md_lines.append("## Aggregate table")
    md_lines.append("")
    md_lines.append("| Case | Backend | Statuses | min ESS_bulk/s | Wall (s) | min ESS_bulk | max R-hat |")
    md_lines.append("|---|---|---|---:|---:|---:|---:|")

    # Stable sort for readability: case, backend.
    for (case_id, backend) in sorted(by_key.keys()):
        statuses = status_by_key.get((case_id, backend), [])
        status_str = ",".join(statuses) if statuses else "—"

        m_essps, s_essps = _mean_std(by_key[(case_id, backend)].get("min_ess_bulk_per_sec", []))
        m_wall, s_wall = _mean_std(by_key[(case_id, backend)].get("wall_time_s", []))
        m_ess, s_ess = _mean_std(by_key[(case_id, backend)].get("min_ess_bulk", []))
        m_rhat, s_rhat = _mean_std(by_key[(case_id, backend)].get("max_r_hat", []))

        essps_cell = (
            f"{_fmt(m_essps)} ± {_fmt(s_essps)}" if m_essps is not None else "—"
        )
        wall_cell = f"{_fmt(m_wall)} ± {_fmt(s_wall)}" if m_wall is not None else "—"
        ess_cell = f"{_fmt(m_ess)} ± {_fmt(s_ess)}" if m_ess is not None else "—"
        rhat_cell = f"{_fmt(m_rhat)} ± {_fmt(s_rhat)}" if m_rhat is not None else "—"

        md_lines.append(
            f"| {case_id} | {backend} | `{status_str}` | {essps_cell} | {wall_cell} | {ess_cell} | {rhat_cell} |"
        )

    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- If some seeds produced `warn`/`failed`, inspect the per-seed `bayesian_suite.json` under each `seed_*` directory.")
    md_lines.append("- Publishable snapshots should pin toolchains and report exact versions; this summary is meant for quick stability checks.")
    md_lines.append("")

    out_md = out_dir / "bayesian_multiseed_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n")

    # Also emit a small machine-readable summary (non-schema).
    out_json = out_dir / "bayesian_multiseed_summary.json"
    out_json.write_text(
        json.dumps(
            {
                "schema_version": "nextstat.bayesian_multiseed_summary.v1",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "suite": "bayesian",
                "seeds": seeds,
                "backends": args.backends,
                "config": {
                    "n_chains": int(args.n_chains),
                    "n_warmup": int(args.warmup),
                    "n_samples": int(args.samples),
                    "max_treedepth": int(args.max_treedepth),
                    "target_accept": float(args.target_accept),
                    "init_jitter_rel": float(args.init_jitter_rel),
                    "deterministic": bool(args.deterministic),
                },
                "cases": [
                    {
                        "case": case_id,
                        "backend": backend,
                        "statuses": status_by_key.get((case_id, backend), []),
                        "min_ess_bulk_per_sec": by_key[(case_id, backend)].get("min_ess_bulk_per_sec", []),
                        "wall_time_s": by_key[(case_id, backend)].get("wall_time_s", []),
                        "min_ess_bulk": by_key[(case_id, backend)].get("min_ess_bulk", []),
                        "max_r_hat": by_key[(case_id, backend)].get("max_r_hat", []),
                        "configs": config_by_key.get((case_id, backend), []),
                    }
                    for (case_id, backend) in sorted(by_key.keys())
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
