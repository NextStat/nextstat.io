#!/usr/bin/env python3
"""Run the MAMS stress suite across multiple seeds and aggregate evidence."""

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


def _seed_semantics_contract() -> dict[str, object]:
    return {
        "benchmark_seed_field": "config.seed",
        "benchmark_seed_alias_field": "config.benchmark_seed",
        "cold_start_seed_field": "config.cold_start_seed",
        "warm_start_seed_field": "config.warm_start_seed",
        "reported_draws_seed_field": "config.reported_draws_seed",
        "reported_draws_source_field": "config.reported_draws_source",
        "warm_start_seed_offset": 1,
        "reported_draws_source": "warm_start",
    }


def _summary_config(*, config_by_key: dict[tuple[str, str], list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, object]:
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
            "n_groups": int(args.n_groups),
            "n_per_group": int(args.n_per_group),
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
        "n_groups": int(_safe_float(first.get("n_groups")) or int(args.n_groups)),
        "n_per_group": int(_safe_float(first.get("n_per_group")) or int(args.n_per_group)),
        "deterministic": bool(args.deterministic),
    }


def _stable_config_overrides(configs: list[dict[str, Any]], summary_config: dict[str, object]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not configs:
        return overrides

    keys: set[str] = set()
    for cfg in configs:
        if isinstance(cfg, dict):
            keys.update(str(key) for key in cfg.keys())

    for key in sorted(keys):
        if key in {
            "seed",
            "benchmark_seed",
            "cold_start_seed",
            "warm_start_seed",
            "reported_draws_seed",
            "reported_draws_source",
        }:
            continue
        values = [cfg.get(key) for cfg in configs if isinstance(cfg, dict) and key in cfg]
        if not values or any(value != values[0] for value in values[1:]):
            continue
        value = values[0]
        if value is None:
            continue
        if key in summary_config and summary_config[key] == value:
            continue
        overrides[key] = value
    return overrides


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Base output directory.")
    ap.add_argument("--backends", default="nextstat_mams,nextstat_nuts", help="Comma-separated backend list.")
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3500)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seeds", default="42,0,123", help="Comma-separated seed list.")
    ap.add_argument("--dataset-seed", type=int, default=12345)
    ap.add_argument("--target-accept", type=float, default=0.985)
    ap.add_argument("--run-timeout-s", type=float, default=300.0)
    ap.add_argument("--parity-warn-z", type=float, default=8.0)
    ap.add_argument("--parity-fail-z", type=float, default=12.0)
    ap.add_argument("--n-groups", type=int, default=20)
    ap.add_argument("--n-per-group", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Do not rerun the suite; regenerate the multiseed summary from existing seed_* artifacts.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_py = Path(__file__).resolve().parent / "stress_suite.py"
    seeds = [int(tok.strip()) for tok in str(args.seeds).split(",") if tok.strip()]
    if not seeds:
        raise SystemExit("no seeds provided")

    rc = 0
    per_seed_suite_paths: dict[int, Path] = {}
    if args.reuse_existing:
        for seed in seeds:
            suite_path = out_dir / f"seed_{seed}" / "mams_stress_suite.json"
            if suite_path.exists():
                per_seed_suite_paths[seed] = suite_path
    else:
        for seed in seeds:
            seed_dir = out_dir / f"seed_{seed}"
            cmd = [
                sys.executable,
                str(suite_py),
                "--out-dir", str(seed_dir),
                "--backends", str(args.backends),
                "--seeds", str(int(seed)),
                "--n-chains", str(int(args.n_chains)),
                "--warmup", str(int(args.warmup)),
                "--samples", str(int(args.samples)),
                "--dataset-seed", str(int(args.dataset_seed)),
                "--target-accept", str(float(args.target_accept)),
                "--run-timeout-s", str(float(args.run_timeout_s)),
                "--parity-warn-z", str(float(args.parity_warn_z)),
                "--parity-fail-z", str(float(args.parity_fail_z)),
                "--n-groups", str(int(args.n_groups)),
                "--n-per-group", str(int(args.n_per_group)),
            ]
            if args.deterministic:
                cmd.append("--deterministic")
            p = subprocess.run(cmd)
            if p.returncode != 0:
                rc = 2

            suite_path = seed_dir / "mams_stress_suite.json"
            if suite_path.exists():
                per_seed_suite_paths[seed] = suite_path

    missing_seeds = [seed for seed in seeds if seed not in per_seed_suite_paths]
    if missing_seeds:
        raise SystemExit(f"missing mams_stress_suite.json for seeds: {', '.join(str(seed) for seed in missing_seeds)}")

    by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    status_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    config_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    case_tier_by_case: dict[str, str] = {}
    parity_scope_by_case: dict[str, str] = {}
    case_catalog_rows: list[dict[str, str]] = []
    parity_by_case: dict[str, dict[str, Any]] = defaultdict(lambda: {"statuses": [], "max_z": [], "seed_rows": []})
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
        if not case_catalog_rows and isinstance(obj.get("case_catalog"), list):
            case_catalog_rows = [row for row in obj["case_catalog"] if isinstance(row, dict)]
            for row in case_catalog_rows:
                case_id = str(row.get("case") or "unknown")
                case_tier_by_case[case_id] = str(row.get("case_tier") or "unknown")
                parity_scope_by_case[case_id] = str(row.get("parity_scope") or "informational")

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

    summary_cfg = _summary_config(config_by_key=config_by_key, args=args)
    seed_semantics = _seed_semantics_contract()
    summary_cases: list[dict[str, Any]] = []
    for (case_id, backend), metric_map in sorted(by_key.items()):
        configs = config_by_key.get((case_id, backend), [])
        summary_cases.append(
            {
                "case": case_id,
                "case_tier": case_tier_by_case.get(case_id, "unknown"),
                "backend": backend,
                "statuses": status_by_key.get((case_id, backend), []),
                "wall_time_s": metric_map.get("wall_time_s", []),
                "n_grad_evals": metric_map.get("n_grad_evals", []),
                "n_integration_steps": metric_map.get("n_integration_steps", []),
                "ess_per_grad": metric_map.get("ess_per_grad", []),
                "ess_per_sec": metric_map.get("ess_per_sec", []),
                "ess_per_sec_warm": metric_map.get("ess_per_sec_warm", []),
                "min_ess_bulk": metric_map.get("min_ess_bulk", []),
                "min_ess_tail": metric_map.get("min_ess_tail", []),
                "max_r_hat": metric_map.get("max_r_hat", []),
                "accept_rate": metric_map.get("accept_rate", []),
                "configs": configs,
                "config_overrides": _stable_config_overrides(configs, summary_cfg),
            }
        )

    parity_rows: list[dict[str, Any]] = []
    for case_id, parity_row in sorted(parity_by_case.items()):
        parity_rows.append(
            {
                "case": case_id,
                "case_tier": case_tier_by_case.get(case_id, "unknown"),
                "parity_scope": parity_scope_by_case.get(case_id, "informational"),
                "statuses": parity_row["statuses"],
                "max_z": parity_row["max_z"],
                "seed_rows": parity_row["seed_rows"],
            }
        )

    summary = {
        "schema_version": "nextstat.mams_stress_multiseed_summary.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite": "mams_stress",
        "seeds": seeds,
        "backends": str(args.backends),
        "config": summary_cfg,
        "seed_semantics": seed_semantics,
        "case_catalog": case_catalog_rows,
        "cases": summary_cases,
        "parity": {
            "method": parity_method,
            "note": parity_note,
            "warn_z": parity_warn_z,
            "fail_z": parity_fail_z,
            "rows": parity_rows,
        },
    }
    summary_path = out_dir / "mams_stress_multiseed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append("# MAMS Stress Multi-Seed Summary")
    lines.append("")
    cfg = summary["config"]
    lines.append(
        f"Config: seeds=`{','.join(str(seed) for seed in seeds)}`, backends=`{args.backends}`, "
        f"n_chains=`{cfg['n_chains']}`, warmup=`{cfg['n_warmup']}`, samples=`{cfg['n_samples']}`, "
        f"dataset_seed=`{cfg['dataset_seed']}`, target_accept=`{cfg['target_accept']}`"
    )
    lines.append("")
    lines.append(
        "Seed semantics: `config.seed` / `config.benchmark_seed` is the requested benchmark seed; "
        "cold start uses that seed, warm start uses `seed+1`, and reported posterior/diagnostic metrics come "
        "from `config.reported_draws_seed`."
    )
    lines.append("")
    lines.append("## Case Catalog")
    lines.append("")
    lines.append("| Case | Tier | Parity scope | Description |")
    lines.append("|---|---|---|---|")
    for row in case_catalog_rows:
        lines.append(
            f"| {row.get('case', '—')} | {row.get('case_tier', '—')} | {row.get('parity_scope', '—')} | {row.get('description', '—')} |"
        )
    lines.append("")
    lines.append("## Aggregate Cases")
    lines.append("")
    lines.append("| Case | Tier | Backend | Statuses | Config overrides | ESS/s mean ± sd | min ESS_bulk worst | max R-hat worst |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for row in summary_cases:
        ess_mean, ess_sd = _mean_std([float(x) for x in row["ess_per_sec"]])
        ess_text = "—" if ess_mean is None else f"{_fmt(ess_mean)} ± {_fmt(ess_sd)}"
        worst_ess_bulk = min((_safe_float(x) for x in row["min_ess_bulk"]), default=None)
        worst_rhat = max((_safe_float(x) for x in row["max_r_hat"]), default=None)
        overrides = row.get("config_overrides") if isinstance(row.get("config_overrides"), dict) else {}
        override_text = "—"
        if overrides:
            override_text = ", ".join(f"{key}={overrides[key]}" for key in sorted(overrides))
        lines.append(
            f"| {row['case']} | {row['case_tier']} | {row['backend']} | `{','.join(row['statuses']) or '—'}` | {override_text} | {ess_text} | {_fmt(worst_ess_bulk)} | {_fmt(worst_rhat)} |"
        )
    if parity_rows:
        lines.append("")
        lines.append("## Parity Summary")
        lines.append("")
        lines.append("| Case | Tier | Scope | Parity statuses | max z mean ± sd | worst max z |")
        lines.append("|---|---|---|---|---:|---:|")
        for row in parity_rows:
            z_mean, z_sd = _mean_std([float(x) for x in row["max_z"]])
            z_text = "—" if z_mean is None else f"{_fmt(z_mean)} ± {_fmt(z_sd)}"
            worst_z = max((_safe_float(x) for x in row["max_z"]), default=None)
            lines.append(
                f"| {row['case']} | {row['case_tier']} | {row['parity_scope']} | `{','.join(row['statuses']) or '—'}` | {z_text} | {_fmt(worst_z)} |"
            )

    md_path = out_dir / "mams_stress_multiseed_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
