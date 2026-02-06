"""Helpers for deterministic Apex2 report JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DROP_KEYS_DETERMINISTIC = {
    # Wall clock / timestamps.
    "timestamp",
    "wall_s",
    "elapsed_s",
    "created_unix_ms",
    # Performance/timing breakdowns (vary by machine/load).
    "timing_s",
    "perf",
    "speedup",
    # Benchmark-style reports.
    "bench",
    "compare",
    "pyhf_wall_s",
    "nextstat_wall_s",
    "pyhf_nll_wall_s",
    "nextstat_nll_wall_s",
    "fit_s",
    "predict_s",
    # Subprocess output (can vary by environment).
    "stdout_tail",
}


def _maybe_sort_named_dict_list(items: list[Any]) -> list[Any]:
    if not items:
        return items
    if not all(isinstance(x, dict) and ("name" in x) for x in items):
        return items
    try:
        names = [str(x.get("name")) for x in items]
    except Exception:
        return items
    if len(set(names)) != len(names):
        return items
    return sorted(items, key=lambda x: str(x.get("name")))


def _canonicalize(obj: Any, *, deterministic: bool) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            k_str = str(k)
            if deterministic and k_str in _DROP_KEYS_DETERMINISTIC:
                continue
            out[k_str] = _canonicalize(v, deterministic=deterministic)
        return out
    if isinstance(obj, list):
        canon = [_canonicalize(x, deterministic=deterministic) for x in obj]
        return _maybe_sort_named_dict_list(canon)
    return obj


def dumps_report_json(report: Any, *, deterministic: bool) -> str:
    canon = _canonicalize(report, deterministic=deterministic)
    return json.dumps(canon, indent=2, sort_keys=True) + "\n"


def write_report_json(path: Path, report: Any, *, deterministic: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_report_json(report, deterministic=deterministic), encoding="utf-8")
