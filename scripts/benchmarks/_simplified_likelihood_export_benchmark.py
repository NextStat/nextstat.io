from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

REPORT_SCHEMA_VERSION = "nextstat_simplified_likelihood_export_benchmark_snapshot_report_v0"
APEX2_REPORT_SCHEMA_VERSION = "nextstat_apex2_simplified_likelihood_report_v0"
DEFAULT_GENERATED_AT = "1970-01-01T00:00:00Z"

REQUIRED_BENCHMARK_HOST = "nextstat-bench"
SNAPSHOT_ARTIFACT_SUITE = "simplified_likelihood_export_benchmark_snapshot"
DEFAULT_CURRENT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "simplified_likelihood_export_benchmarks"
    / REQUIRED_BENCHMARK_HOST
    / "current"
)
DEFAULT_HISTORY_DIR = DEFAULT_CURRENT_DIR.parent / "history"


def now_utc(deterministic: bool) -> str:
    if deterministic:
        return DEFAULT_GENERATED_AT
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def derive_stamp_from_path(path: Path) -> str | None:
    for candidate in [path.name, path.parent.name, *path.parts]:
        match = re.search(r"(\d{8}T\d{6}Z)", candidate)
        if match:
            return match.group(1)
    return None

