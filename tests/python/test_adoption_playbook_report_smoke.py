from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _report_path() -> Path:
    return Path(
        os.environ.get(
            "ADOPTION_PLAYBOOK_REPORT",
            "tmp/reports/adoption_playbook_smoke_report.json",
        )
    )


def _require_report() -> bool:
    return os.environ.get("NEXTSTAT_REQUIRE_ADOPTION_REPORT", "0") == "1"


def _assert_route(route_payload: dict, *, route_name: str) -> None:
    assert isinstance(route_payload.get("ok"), bool), f"{route_name}.ok must be bool"
    comparisons = route_payload.get("comparisons")
    assert isinstance(comparisons, list), f"{route_name}.comparisons must be list"
    assert comparisons, f"{route_name}.comparisons must be non-empty"
    for item in comparisons:
        assert isinstance(item, dict), f"{route_name} comparison item must be object"
        assert isinstance(item.get("generated"), str) and item["generated"], "missing generated"
        assert isinstance(item.get("expected"), str) and item["expected"], "missing expected"
        assert isinstance(item.get("ok"), bool), "comparison.ok must be bool"
        if item.get("reason") is not None:
            assert isinstance(item["reason"], str), "comparison.reason must be string/null"


def test_adoption_playbook_smoke_report_shape() -> None:
    report_path = _report_path()
    if not report_path.exists():
        if _require_report():
            pytest.fail(f"required adoption report is missing: {report_path}")
        pytest.skip(f"adoption report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(report, dict)
    assert isinstance(report.get("generated_at_utc"), str) and report["generated_at_utc"]
    assert report.get("status") in {"pass", "fail"}

    route_a = report.get("route_a")
    route_b = report.get("route_b")
    route_c = report.get("route_c")
    assert isinstance(route_a, dict), "route_a must be object"
    assert isinstance(route_b, dict), "route_b must be object"
    assert isinstance(route_c, dict), "route_c must be object"

    _assert_route(route_a, route_name="route_a")
    _assert_route(route_b, route_name="route_b")
    _assert_route(route_c, route_name="route_c")

    parity = route_c.get("from_arrow_large_offsets_parity")
    assert isinstance(parity, dict), "route_c.from_arrow_large_offsets_parity must be object"
    for key in (
        "mu_parquet",
        "mu_polars_arrow",
        "mu_duckdb_arrow",
        "abs_diff_polars",
        "abs_diff_duckdb",
    ):
        assert isinstance(parity.get(key), (float, int)), f"parity.{key} must be numeric"
    assert isinstance(parity.get("ok"), bool), "parity.ok must be bool"

