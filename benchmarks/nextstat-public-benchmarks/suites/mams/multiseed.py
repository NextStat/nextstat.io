#!/usr/bin/env python3
"""Run the MAMS suite across multiple seeds and aggregate repeatability evidence.

This is a convenience wrapper around `suites/mams/suite.py` that:
- creates one subdirectory per seed (schema-compatible artifacts per run)
- emits an aggregated Markdown/JSON summary for stability and parity review

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
from typing import Any


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _fmt(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if v == 0:
        return "0"
    return f"{v:.3g}"


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _summary_contract(
    *,
    config_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, object]:
    configs = [cfg for cfgs in config_by_key.values() for cfg in cfgs if isinstance(cfg, dict)]
    if not configs:
        return {
            "n_chains": int(args.n_chains),
            "n_warmup": int(args.warmup),
            "n_samples": int(args.samples),
            "dataset_seed": int(args.dataset_seed),
            "target_accept": float(args.target_accept),
            "run_timeout_s": float(args.run_timeout_s),
            "parity_warn_z": float(args.parity_warn_z),
            "parity_fail_z": float(args.parity_fail_z),
            "deterministic": bool(args.deterministic),
        }

    first = configs[0]
    target_accept_vals = [_safe_float(cfg.get("target_accept")) for cfg in configs]
    target_accept_vals = [v for v in target_accept_vals if v is not None]
    dataset_seed_vals = [_safe_float(cfg.get("dataset_seed")) for cfg in configs]
    dataset_seed_vals = [int(v) for v in dataset_seed_vals if v is not None]

    return {
        "n_chains": int(_safe_float(first.get("n_chains")) or int(args.n_chains)),
        "n_warmup": int(_safe_float(first.get("n_warmup")) or int(args.warmup)),
        "n_samples": int(_safe_float(first.get("n_samples")) or int(args.samples)),
        "dataset_seed": dataset_seed_vals[0] if dataset_seed_vals else int(args.dataset_seed),
        "target_accept": min(target_accept_vals) if target_accept_vals else float(args.target_accept),
        "run_timeout_s": float(args.run_timeout_s),
        "parity_warn_z": float(args.parity_warn_z),
        "parity_fail_z": float(args.parity_fail_z),
        "deterministic": bool(args.deterministic),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Base output directory.")
    ap.add_argument("--backends", default="nextstat_mams,nextstat_nuts", help="Comma-separated backend list.")
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3500)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seeds", default="42,0,123", help="Comma-separated seed list.")
    ap.add_argument(
        "--dataset-seed",
        type=int,
        default=12345,
        help="Fixed dataset seed used for generated fixtures such as glm_logistic.",
    )
    ap.add_argument("--target-accept", type=float, default=0.985)
    ap.add_argument("--run-timeout-s", type=float, default=300.0)
    ap.add_argument("--parity-warn-z", type=float, default=8.0)
    ap.add_argument("--parity-fail-z", type=float, default=12.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Do not rerun the suite; regenerate the multiseed summary from existing seed_* artifacts.",
    )
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

    rc = 0
    per_seed_suite_paths: dict[int, Path] = {}
    if args.reuse_existing:
        for seed in seeds:
            suite_path = out_dir / f"seed_{seed}" / "mams_suite.json"
            if suite_path.exists():
                per_seed_suite_paths[seed] = suite_path
    else:
        for seed in seeds:
            seed_dir = out_dir / f"seed_{seed}"
            cmd = [
                sys.executable,
                str(suite_py),
                "--out-dir",
                str(seed_dir),
                "--backends",
                str(args.backends),
                "--seeds",
                str(int(seed)),
                "--n-chains",
                str(int(args.n_chains)),
                "--warmup",
                str(int(args.warmup)),
                "--samples",
                str(int(args.samples)),
                "--dataset-seed",
                str(int(args.dataset_seed)),
                "--target-accept",
                str(float(args.target_accept)),
                "--run-timeout-s",
                str(float(args.run_timeout_s)),
                "--parity-warn-z",
                str(float(args.parity_warn_z)),
                "--parity-fail-z",
                str(float(args.parity_fail_z)),
            ]
            if args.deterministic:
                cmd.append("--deterministic")
            p = subprocess.run(cmd)
            if p.returncode != 0:
                rc = 2

            suite_path = seed_dir / "mams_suite.json"
            if suite_path.exists():
                per_seed_suite_paths[seed] = suite_path

    missing_seeds = [seed for seed in seeds if seed not in per_seed_suite_paths]
    if missing_seeds:
        raise SystemExit(f"missing mams_suite.json for seeds: {', '.join(str(seed) for seed in missing_seeds)}")

    by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    status_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    config_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    parity_by_case: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"statuses": [], "max_z": [], "seed_rows": []}
    )
    parity_method = "mean_zscore"
    parity_note = ""
    parity_warn_z = float(args.parity_warn_z)
    parity_fail_z = float(args.parity_fail_z)

    metric_keys = [
        "wall_time_s",
        "n_grad_evals",
        "n_integration_steps",
        "ess_per_grad",
        "ess_per_sec",
        "ess_per_sec_warm",
        "min_ess_bulk",
        "min_ess_tail",
        "max_r_hat",
        "accept_rate",
    ]

    for seed, suite_path in per_seed_suite_paths.items():
        obj = _load_json(suite_path)
        parity = obj.get("parity") if isinstance(obj.get("parity"), dict) else {}
        parity_method = str(parity.get("method") or parity_method)
        parity_note = str(parity.get("note") or parity_note)
        parity_warn_z = _safe_float(parity.get("warn_z")) or parity_warn_z
        parity_fail_z = _safe_float(parity.get("fail_z")) or parity_fail_z

        for row in parity.get("rows") if isinstance(parity.get("rows"), list) else []:
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("case") or "unknown")
            parity_by_case[case_id]["statuses"].append(str(row.get("status") or "unknown"))
            max_z = _safe_float(row.get("max_z"))
            if max_z is not None:
                parity_by_case[case_id]["max_z"].append(max_z)
            parity_by_case[case_id]["seed_rows"].append(
                {
                    "seed": int(row.get("seed", seed)),
                    "status": str(row.get("status") or "unknown"),
                    "max_z": max_z,
                    "worst": row.get("worst") if isinstance(row.get("worst"), list) else [],
                }
            )

        for c in obj.get("cases") if isinstance(obj.get("cases"), list) else []:
            if not isinstance(c, dict):
                continue
            case_id = str(c.get("case") or "unknown")
            backend = str(c.get("backend") or "unknown")
            key = (case_id, backend)
            status_by_key[key].append(str(c.get("status") or "unknown"))

            rel = c.get("path")
            case_obj: dict[str, Any] | None = None
            if isinstance(rel, str) and rel:
                case_path = (suite_path.parent / rel).resolve()
                if case_path.exists():
                    case_obj = _load_json(case_path)
                    if isinstance(case_obj.get("config"), dict):
                        config_by_key[key].append(case_obj["config"])

            metrics = case_obj.get("metrics") if isinstance(case_obj, dict) and isinstance(case_obj.get("metrics"), dict) else {}
            for metric_key in metric_keys:
                value = _safe_float(metrics.get(metric_key))
                if value is None:
                    value = _safe_float(c.get(metric_key))
                if value is not None:
                    by_key[key][metric_key].append(value)

    actual_backends = sorted({backend for (_, backend) in by_key.keys()})
    backends_str = ",".join(actual_backends) if actual_backends else str(args.backends)
    summary_config = _summary_contract(config_by_key=config_by_key, args=args)

    md_lines: list[str] = []
    md_lines.append("# MAMS suite (multi-seed summary)")
    md_lines.append("")
    md_lines.append(f"- Seeds: `{', '.join(str(s) for s in seeds)}`")
    md_lines.append(f"- Backends: `{backends_str}`")
    md_lines.append(
        f"- Config: `chains={summary_config['n_chains']}`, `warmup={summary_config['n_warmup']}`, `samples={summary_config['n_samples']}`, "
        f"`dataset_seed={summary_config['dataset_seed']}`, `target_accept={summary_config['target_accept']}`, "
        f"`parity_warn_z={summary_config['parity_warn_z']}`, `parity_fail_z={summary_config['parity_fail_z']}`"
    )
    md_lines.append("")
    md_lines.append("Metrics are aggregated across sampler seeds as mean ± std (where available).")
    md_lines.append("`dataset_seed` stays fixed so repeatability reflects sampler variation, not regenerated data variation.")
    md_lines.append("")
    md_lines.append("## Aggregate table")
    md_lines.append("")
    md_lines.append("| Case | Backend | Statuses | ESS/grad | ESS/s | Warm ESS/s | Wall (s) | min ESS_bulk | max R-hat |")
    md_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")

    for (case_id, backend) in sorted(by_key.keys()):
        statuses = status_by_key.get((case_id, backend), [])
        status_str = ",".join(statuses) if statuses else "—"
        row_metrics = by_key[(case_id, backend)]

        m_ess_grad, s_ess_grad = _mean_std(row_metrics.get("ess_per_grad", []))
        m_ess_sec, s_ess_sec = _mean_std(row_metrics.get("ess_per_sec", []))
        m_ess_sec_warm, s_ess_sec_warm = _mean_std(row_metrics.get("ess_per_sec_warm", []))
        m_wall, s_wall = _mean_std(row_metrics.get("wall_time_s", []))
        m_ess_bulk, s_ess_bulk = _mean_std(row_metrics.get("min_ess_bulk", []))
        m_rhat, s_rhat = _mean_std(row_metrics.get("max_r_hat", []))

        md_lines.append(
            f"| {case_id} | {backend} | `{status_str}` | "
            f"{_fmt(m_ess_grad)} ± {_fmt(s_ess_grad) if m_ess_grad is not None else '—'} | "
            f"{_fmt(m_ess_sec)} ± {_fmt(s_ess_sec) if m_ess_sec is not None else '—'} | "
            f"{_fmt(m_ess_sec_warm)} ± {_fmt(s_ess_sec_warm) if m_ess_sec_warm is not None else '—'} | "
            f"{_fmt(m_wall)} ± {_fmt(s_wall) if m_wall is not None else '—'} | "
            f"{_fmt(m_ess_bulk)} ± {_fmt(s_ess_bulk) if m_ess_bulk is not None else '—'} | "
            f"{_fmt(m_rhat)} ± {_fmt(s_rhat) if m_rhat is not None else '—'} |"
        )

    md_lines.append("")
    md_lines.append("## Health Summary")
    md_lines.append("")
    md_lines.append("| Case | Backend | Worst ESS/s | Worst min ESS_bulk | Worst min ESS_tail | Worst R-hat | Worst accept rate |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (case_id, backend) in sorted(by_key.keys()):
        row_metrics = by_key[(case_id, backend)]
        ess_sec_vals = row_metrics.get("ess_per_sec", [])
        ess_bulk_vals = row_metrics.get("min_ess_bulk", [])
        ess_tail_vals = row_metrics.get("min_ess_tail", [])
        rhat_vals = row_metrics.get("max_r_hat", [])
        accept_vals = row_metrics.get("accept_rate", [])
        md_lines.append(
            f"| {case_id} | {backend} | {_fmt(min(ess_sec_vals) if ess_sec_vals else None)} | "
            f"{_fmt(min(ess_bulk_vals) if ess_bulk_vals else None)} | "
            f"{_fmt(min(ess_tail_vals) if ess_tail_vals else None)} | "
            f"{_fmt(max(rhat_vals) if rhat_vals else None)} | "
            f"{_fmt(min(accept_vals) if accept_vals else None)} |"
        )

    if parity_by_case:
        md_lines.append("")
        md_lines.append("## Parity Summary")
        md_lines.append("")
        md_lines.append("| Case | Statuses | max z | Worst max z |")
        md_lines.append("|---|---|---:|---:|")
        for case_id in sorted(parity_by_case.keys()):
            statuses = ",".join(parity_by_case[case_id]["statuses"]) or "—"
            max_z_vals = parity_by_case[case_id]["max_z"]
            m_z, s_z = _mean_std(max_z_vals)
            md_lines.append(
                f"| {case_id} | `{statuses}` | "
                f"{_fmt(m_z)} ± {_fmt(s_z) if m_z is not None else '—'} | "
                f"{_fmt(max(max_z_vals) if max_z_vals else None)} |"
            )

    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- If some seeds produced `warn`/`failed`, inspect the per-seed `mams_suite.json` under each `seed_*` directory.")
    md_lines.append("- `--reuse-existing` regenerates the summary from existing `seed_*` artifacts without rerunning the suite.")
    md_lines.append("- Parity rows aggregate the tracked NextStat MAMS vs NextStat NUTS posterior mean z-score comparison from each per-seed suite run.")
    md_lines.append("")

    (out_dir / "mams_multiseed_summary.md").write_text("\n".join(md_lines) + "\n")

    cases_json = []
    for (case_id, backend) in sorted(by_key.keys()):
        row_metrics = by_key[(case_id, backend)]
        cases_json.append(
            {
                "case": case_id,
                "backend": backend,
                "statuses": status_by_key.get((case_id, backend), []),
                "wall_time_s": row_metrics.get("wall_time_s", []),
                "n_grad_evals": row_metrics.get("n_grad_evals", []),
                "n_integration_steps": row_metrics.get("n_integration_steps", []),
                "ess_per_grad": row_metrics.get("ess_per_grad", []),
                "ess_per_sec": row_metrics.get("ess_per_sec", []),
                "ess_per_sec_warm": row_metrics.get("ess_per_sec_warm", []),
                "min_ess_bulk": row_metrics.get("min_ess_bulk", []),
                "min_ess_tail": row_metrics.get("min_ess_tail", []),
                "max_r_hat": row_metrics.get("max_r_hat", []),
                "accept_rate": row_metrics.get("accept_rate", []),
                "configs": config_by_key.get((case_id, backend), []),
            }
        )

    parity_rows = []
    for case_id in sorted(parity_by_case.keys()):
        parity_rows.append(
            {
                "case": case_id,
                "statuses": parity_by_case[case_id]["statuses"],
                "max_z": parity_by_case[case_id]["max_z"],
                "seed_rows": parity_by_case[case_id]["seed_rows"],
            }
        )

    (out_dir / "mams_multiseed_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "nextstat.mams_multiseed_summary.v1",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "suite": "mams",
                "seeds": seeds,
                "backends": backends_str,
                "config": summary_config,
                "cases": cases_json,
                "parity": {
                    "method": parity_method,
                    "note": parity_note,
                    "warn_z": parity_warn_z,
                    "fail_z": parity_fail_z,
                    "rows": parity_rows,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
