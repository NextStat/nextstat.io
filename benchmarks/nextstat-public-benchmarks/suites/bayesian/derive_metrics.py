#!/usr/bin/env python3
"""Derive supplementary Bayesian metrics from a multi-seed suite directory.

This keeps supplementary repeatability diagnostics such as ESS/leapfrog
artifact-driven and reproducible from the checked-in per-seed case JSON files.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iter_seed_dirs(results_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for d in sorted(results_dir.glob("seed_*")):
        if not d.is_dir():
            continue
        tok = d.name.split("_", 1)[-1]
        try:
            out.append((int(tok), d))
        except Exception:
            continue
    return out


def _metric_backend_name(raw_backend: str) -> str | None:
    if raw_backend == "nextstat":
        return "nextstat"
    if raw_backend == "cmdstanpy":
        return "cmdstan"
    if raw_backend == "pymc":
        return "pymc"
    return None


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def build_derived_metrics(results_dir: Path) -> dict[str, Any]:
    seed_dirs = _iter_seed_dirs(results_dir)
    by_case_backend_seed: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    base_config: dict[str, Any] | None = None

    for seed, seed_dir in seed_dirs:
        suite_path = seed_dir / "bayesian_suite.json"
        if not suite_path.exists():
            continue
        suite_obj = _load_json(suite_path)
        cases = suite_obj.get("cases")
        if not isinstance(cases, list):
            continue

        for case_row in cases:
            if not isinstance(case_row, dict):
                continue
            backend = _metric_backend_name(str(case_row.get("backend") or ""))
            if backend is None:
                continue
            if str(case_row.get("status") or "") != "ok":
                continue

            rel_path = case_row.get("path")
            if not isinstance(rel_path, str) or not rel_path:
                continue
            case_path = (suite_path.parent / rel_path).resolve()
            if not case_path.exists():
                continue

            case_obj = _load_json(case_path)
            timing = case_obj.get("timing") if isinstance(case_obj.get("timing"), dict) else {}
            diag_summary = (
                case_obj.get("diagnostics_summary")
                if isinstance(case_obj.get("diagnostics_summary"), dict)
                else {}
            )
            cfg = case_obj.get("config") if isinstance(case_obj.get("config"), dict) else {}
            if base_config is None and cfg:
                base_config = {
                    "dataset_seed": _safe_int(cfg.get("dataset_seed")),
                    "init_jitter_rel": _safe_float(cfg.get("init_jitter_rel")),
                    "max_treedepth": _safe_int(cfg.get("max_treedepth")),
                    "metric": cfg.get("metric"),
                    "n_chains": _safe_int(cfg.get("n_chains")),
                    "n_samples": _safe_int(cfg.get("n_samples")),
                    "n_warmup": _safe_int(cfg.get("n_warmup")),
                    "target_accept": _safe_float(cfg.get("target_accept")),
                }

            min_ess_bulk = _safe_float(diag_summary.get("min_ess_bulk"))
            total_leapfrog = _safe_int(timing.get("n_grad_evals"))
            if min_ess_bulk is None or total_leapfrog is None or total_leapfrog <= 0:
                continue

            case_id = str(case_obj.get("case") or case_row.get("case") or "unknown")
            by_case_backend_seed[case_id][backend][seed] = {
                "min_ess_bulk": float(min_ess_bulk),
                "total_leapfrog": int(total_leapfrog),
                "ess_per_leapfrog": float(min_ess_bulk) / float(total_leapfrog),
            }

    cases_out: dict[str, Any] = {}
    for case_id in sorted(by_case_backend_seed.keys()):
        case_payload: dict[str, Any] = {"by_seed": {}}

        per_backend_seed = by_case_backend_seed[case_id]
        seed_union = sorted({seed for backend_rows in per_backend_seed.values() for seed in backend_rows.keys()})
        for seed in seed_union:
            seed_payload: dict[str, Any] = {}
            for backend in ("nextstat", "cmdstan", "pymc"):
                row = per_backend_seed.get(backend, {}).get(seed)
                if row is not None:
                    seed_payload[backend] = row
            if seed_payload:
                case_payload["by_seed"][str(seed)] = seed_payload

        backend_means: dict[str, float] = {}
        for backend in ("nextstat", "cmdstan", "pymc"):
            vals = [
                float(entry["ess_per_leapfrog"])
                for entry in per_backend_seed.get(backend, {}).values()
            ]
            mean, std = _mean_std(vals)
            if mean is not None:
                case_payload[backend] = {"mean": mean, "std": std}
                backend_means[backend] = mean

        if "nextstat" in backend_means and "cmdstan" in backend_means and backend_means["cmdstan"] > 0.0:
            case_payload["ratio"] = backend_means["nextstat"] / backend_means["cmdstan"]

        cases_out[case_id] = case_payload

    config = base_config or {
        "dataset_seed": None,
        "init_jitter_rel": None,
        "max_treedepth": None,
        "metric": None,
        "n_chains": None,
        "n_samples": None,
        "n_warmup": None,
        "target_accept": None,
    }

    return {
        "schema_version": "nextstat.bayesian_derived_metrics.v2",
        "created_at": time.strftime("%Y-%m-%d", time.gmtime()),
        "source": "artifact_recompute_min_ess_bulk_over_n_grad_evals",
        "ess_per_leapfrog": {
            "config": config,
            "ess_method": "artifact_min_ess_bulk",
            "seeds": [seed for seed, _ in seed_dirs],
            "cases": cases_out,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="Path to a Bayesian multiseed directory.")
    ap.add_argument("--out", default=None, help="Output JSON path (defaults to <results_dir>/derived_metrics.json).")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (results_dir / "derived_metrics.json")
    out_path.write_text(json.dumps(build_derived_metrics(results_dir), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
