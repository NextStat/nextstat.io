#!/usr/bin/env python3
"""Smoke-check for Adoption Playbook routes (A/B/C).

Runs quickstart-like commands and compares generated artifacts with
`docs/guides/fixtures/*` references.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "docs" / "guides" / "fixtures"
TMP_DIR = REPO_ROOT / "tmp" / "guides"


@dataclass
class CompareResult:
    generated: str
    expected: str
    ok: bool
    reason: str | None = None


def run_cmd(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    cwd = cwd or REPO_ROOT
    cmd = " ".join(shlex.quote(x) for x in args)
    print(f"+ {cmd}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(args, cwd=str(cwd), env=merged_env, check=True)


def run_cli(ns_cli: Path, args: list[str]) -> None:
    run_cmd([str(ns_cli), *args], cwd=REPO_ROOT)


def write_scan_csv(scan_json: Path, scan_csv: Path) -> None:
    points = json.loads(scan_json.read_text(encoding="utf-8"))["points"]
    with scan_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mu", "q_mu", "nll_mu", "converged", "n_iter"],
        )
        writer.writeheader()
        for row in points:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})


def normalize_json_for_compare(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, inner in value.items():
            if key == "meta" and isinstance(inner, dict):
                inner = dict(inner)
                inner.pop("created_unix_ms", None)
                inner.pop("tool_version", None)
            out[key] = normalize_json_for_compare(inner)
        return out
    if isinstance(value, list):
        return [normalize_json_for_compare(v) for v in value]
    return value


def compare_json(generated: Path, expected: Path, *, strip_meta_ts: bool = False) -> CompareResult:
    lhs = json.loads(generated.read_text(encoding="utf-8"))
    rhs = json.loads(expected.read_text(encoding="utf-8"))
    if strip_meta_ts:
        lhs = normalize_json_for_compare(lhs)
        rhs = normalize_json_for_compare(rhs)
    ok = lhs == rhs
    return CompareResult(
        generated=str(generated.relative_to(REPO_ROOT)),
        expected=str(expected.relative_to(REPO_ROOT)),
        ok=ok,
        reason=None if ok else "json mismatch",
    )


def compare_text(generated: Path, expected: Path) -> CompareResult:
    ok = generated.read_text(encoding="utf-8") == expected.read_text(encoding="utf-8")
    return CompareResult(
        generated=str(generated.relative_to(REPO_ROOT)),
        expected=str(expected.relative_to(REPO_ROOT)),
        ok=ok,
        reason=None if ok else "text mismatch",
    )


def compare_pairs_json(pairs: list[tuple[Path, Path]], *, strip_meta_ts: bool = False) -> list[CompareResult]:
    return [compare_json(gen, exp, strip_meta_ts=strip_meta_ts) for gen, exp in pairs]


def compare_pairs_text(pairs: list[tuple[Path, Path]]) -> list[CompareResult]:
    return [compare_text(gen, exp) for gen, exp in pairs]


def route_a(ns_cli: Path) -> list[CompareResult]:
    out = TMP_DIR / "route_a"
    out.mkdir(parents=True, exist_ok=True)

    run_cli(
        ns_cli,
        [
            "fit",
            "--input",
            "tests/fixtures/simple_workspace.json",
            "--parity",
            "--threads",
            "1",
            "--output",
            str(out / "fit_result.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "hypotest",
            "--input",
            "tests/fixtures/simple_workspace.json",
            "--mu",
            "1.0",
            "--expected-set",
            "--threads",
            "1",
            "--output",
            str(out / "hypotest_mu1.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "upper-limit",
            "--input",
            "tests/fixtures/simple_workspace.json",
            "--expected",
            "--scan-start",
            "0",
            "--scan-stop",
            "5",
            "--scan-points",
            "101",
            "--threads",
            "1",
            "--output",
            str(out / "upper_limit_scan.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "scan",
            "--input",
            "tests/fixtures/simple_workspace.json",
            "--start",
            "0",
            "--stop",
            "5",
            "--points",
            "41",
            "--threads",
            "1",
            "--output",
            str(out / "scan_points.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "viz",
            "cls",
            "--input",
            "tests/fixtures/simple_workspace.json",
            "--scan-start",
            "0",
            "--scan-stop",
            "5",
            "--scan-points",
            "101",
            "--threads",
            "1",
            "--output",
            str(out / "cls_curve.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "viz",
            "pulls",
            "--input",
            "tests/fixtures/simple_workspace.json",
            "--fit",
            str(out / "fit_result.json"),
            "--threads",
            "1",
            "--output",
            str(out / "pulls.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "viz",
            "corr",
            "--input",
            "tests/fixtures/simple_workspace.json",
            "--fit",
            str(out / "fit_result.json"),
            "--threads",
            "1",
            "--output",
            str(out / "corr.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "hypotest-toys",
            "--input",
            "tests/fixtures/simple_workspace.json",
            "--mu",
            "1.0",
            "--n-toys",
            "200",
            "--seed",
            "42",
            "--expected-set",
            "--threads",
            "1",
            "--output",
            str(out / "hypotest_toys_mu1.json"),
        ],
    )
    write_scan_csv(out / "scan_points.json", out / "scan_points.csv")

    fixture = FIXTURES_DIR / "route_a"
    out_pairs_json = [
        (out / "fit_result.json", fixture / "fit_result.json"),
        (out / "hypotest_mu1.json", fixture / "hypotest_mu1.json"),
        (out / "upper_limit_scan.json", fixture / "upper_limit_scan.json"),
        (out / "scan_points.json", fixture / "scan_points.json"),
        (out / "cls_curve.json", fixture / "cls_curve.json"),
    ]
    out_pairs_json_ts = [
        (out / "pulls.json", fixture / "pulls.json"),
        (out / "corr.json", fixture / "corr.json"),
    ]
    out_pairs_text = [(out / "scan_points.csv", fixture / "scan_points.csv")]

    results: list[CompareResult] = []
    results.extend(compare_pairs_json(out_pairs_json))
    results.extend(compare_pairs_json(out_pairs_json_ts, strip_meta_ts=True))
    results.extend(compare_pairs_text(out_pairs_text))
    return results


def route_b(ns_cli: Path) -> list[CompareResult]:
    out = TMP_DIR / "route_b"
    out.mkdir(parents=True, exist_ok=True)

    run_cli(
        ns_cli,
        [
            "import",
            "trex-config",
            "--config",
            "docs/guides/fixtures/route_b/minimal_ntup_quickstart.config",
            "--base-dir",
            ".",
            "--output",
            str(out / "workspace_from_import_trex_config.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "build-hists",
            "--config",
            "docs/guides/fixtures/route_b/minimal_ntup_quickstart.config",
            "--base-dir",
            ".",
            "--out-dir",
            str(out / "build_hists"),
            "--overwrite",
        ],
    )
    run_cli(
        ns_cli,
        [
            "fit",
            "--input",
            str(out / "build_hists" / "workspace.json"),
            "--threads",
            "1",
            "--output",
            str(out / "fit_result.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "upper-limit",
            "--input",
            str(out / "build_hists" / "workspace.json"),
            "--expected",
            "--scan-start",
            "0",
            "--scan-stop",
            "3",
            "--scan-points",
            "81",
            "--threads",
            "1",
            "--output",
            str(out / "upper_limit_scan.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "scan",
            "--input",
            str(out / "build_hists" / "workspace.json"),
            "--start",
            "0",
            "--stop",
            "3",
            "--points",
            "31",
            "--threads",
            "1",
            "--output",
            str(out / "scan_points.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "viz",
            "cls",
            "--input",
            str(out / "build_hists" / "workspace.json"),
            "--scan-start",
            "0",
            "--scan-stop",
            "3",
            "--scan-points",
            "81",
            "--threads",
            "1",
            "--output",
            str(out / "cls_curve.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "viz",
            "pulls",
            "--input",
            str(out / "build_hists" / "workspace.json"),
            "--fit",
            str(out / "fit_result.json"),
            "--threads",
            "1",
            "--output",
            str(out / "pulls.json"),
        ],
    )
    run_cli(
        ns_cli,
        [
            "viz",
            "corr",
            "--input",
            str(out / "build_hists" / "workspace.json"),
            "--fit",
            str(out / "fit_result.json"),
            "--threads",
            "1",
            "--output",
            str(out / "corr.json"),
        ],
    )
    write_scan_csv(out / "scan_points.json", out / "scan_points.csv")

    fixture = FIXTURES_DIR / "route_b"
    out_pairs_json = [
        (out / "workspace_from_import_trex_config.json", fixture / "workspace_from_import_trex_config.json"),
        (out / "build_hists" / "workspace.json", fixture / "workspace_from_trex_config.json"),
        (out / "fit_result.json", fixture / "fit_result.json"),
        (out / "upper_limit_scan.json", fixture / "upper_limit_scan.json"),
        (out / "scan_points.json", fixture / "scan_points.json"),
        (out / "cls_curve.json", fixture / "cls_curve.json"),
    ]
    out_pairs_json_ts = [
        (out / "pulls.json", fixture / "pulls.json"),
        (out / "corr.json", fixture / "corr.json"),
    ]
    out_pairs_text = [(out / "scan_points.csv", fixture / "scan_points.csv")]

    results: list[CompareResult] = []
    results.extend(compare_pairs_json(out_pairs_json))
    results.extend(compare_pairs_json(out_pairs_json_ts, strip_meta_ts=True))
    results.extend(compare_pairs_text(out_pairs_text))
    return results


def route_c() -> tuple[list[CompareResult], dict[str, Any]]:
    out = TMP_DIR / "route_c"
    out.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, "docs/guides/fixtures/route_c/build_histograms_parquet_example.py"])

    try:
        import duckdb  # type: ignore
        import nextstat  # type: ignore
        import polars as pl  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime env guard
        raise RuntimeError(
            "Route C requires nextstat + pyarrow + polars + duckdb in the active Python environment"
        ) from exc

    observations = {"SR": [55.0, 65.0]}
    model = nextstat.from_parquet(
        "docs/guides/fixtures/route_c/histograms.parquet",
        poi="mu",
        observations=observations,
    )
    fit = nextstat.fit(model)
    mu_values = [i * 5.0 / 40.0 for i in range(41)]
    scan = nextstat.profile_scan(model, mu_values)
    cls = nextstat.cls_curve(model, mu_values)
    mu_up = nextstat.upper_limit(model)

    fit_out = {
        "bestfit": [float(x) for x in fit.bestfit],
        "parameters": [float(x) for x in fit.parameters],
        "parameter_names": list(model.parameter_names()),
        "uncertainties": [float(x) for x in fit.uncertainties],
        "nll": float(fit.nll),
        "converged": bool(fit.converged),
    }
    (out / "fit_result.json").write_text(
        json.dumps(fit_out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "scan_points.json").write_text(
        json.dumps(scan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "cls_curve.json").write_text(
        json.dumps(cls, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upper = {
        "alpha": float(cls["alpha"]),
        "exp_limits": [float(x) for x in cls["exp_limits"]],
        "mode": "scan",
        "mu_up": float(mu_up),
        "obs_limit": float(cls["obs_limit"]),
        "scan": {"start": 0.0, "stop": 5.0, "points": len(mu_values)},
    }
    (out / "upper_limit_scan.json").write_text(
        json.dumps(upper, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_scan_csv(out / "scan_points.json", out / "scan_points.csv")

    # from_arrow large-offset parity (no casts/normalization).
    pl_df = pl.read_parquet("docs/guides/fixtures/route_c/histograms.parquet")
    model_a = nextstat.from_arrow(pl_df.to_arrow(), poi="mu", observations=observations)
    fit_a = nextstat.fit(model_a)

    con = duckdb.connect()
    reader = con.execute(
        "SELECT * FROM 'docs/guides/fixtures/route_c/histograms.parquet'"
    ).arrow()
    model_b = nextstat.from_arrow(reader.read_all(), poi="mu", observations=observations)
    fit_b = nextstat.fit(model_b)

    mu_parquet = float(fit.bestfit[0])
    mu_polars = float(fit_a.bestfit[0])
    mu_duckdb = float(fit_b.bestfit[0])
    parity_ok = (
        abs(mu_parquet - mu_polars) < 1e-12 and abs(mu_parquet - mu_duckdb) < 1e-12
    )
    parity = {
        "mu_parquet": mu_parquet,
        "mu_polars_arrow": mu_polars,
        "mu_duckdb_arrow": mu_duckdb,
        "abs_diff_polars": abs(mu_parquet - mu_polars),
        "abs_diff_duckdb": abs(mu_parquet - mu_duckdb),
        "ok": parity_ok,
    }

    fixture = FIXTURES_DIR / "route_c"
    out_pairs_json = [
        (out / "fit_result.json", fixture / "fit_result.json"),
        (out / "upper_limit_scan.json", fixture / "upper_limit_scan.json"),
        (out / "scan_points.json", fixture / "scan_points.json"),
        (out / "cls_curve.json", fixture / "cls_curve.json"),
    ]
    out_pairs_text = [(out / "scan_points.csv", fixture / "scan_points.csv")]

    results: list[CompareResult] = []
    results.extend(compare_pairs_json(out_pairs_json))
    results.extend(compare_pairs_text(out_pairs_text))
    return results, parity


def results_to_dict(results: list[CompareResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in results:
        out.append(
            {
                "generated": item.generated,
                "expected": item.expected,
                "ok": item.ok,
                "reason": item.reason,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "tmp" / "reports" / "adoption_playbook_smoke_report.json"),
        help="Path to JSON report output",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `cargo build -p ns-cli` (expects binary already available)",
    )
    args = parser.parse_args()

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    ns_cli = REPO_ROOT / target_dir / "debug" / "nextstat"

    if not args.skip_build:
        run_cmd(["cargo", "build", "-p", "ns-cli"], env={"CARGO_TARGET_DIR": str(target_dir)})
    if not ns_cli.exists():
        raise FileNotFoundError(f"ns-cli binary not found: {ns_cli}")

    route_a_results = route_a(ns_cli)
    route_b_results = route_b(ns_cli)
    route_c_results, route_c_parity = route_c()

    all_results = route_a_results + route_b_results + route_c_results
    all_ok = all(r.ok for r in all_results) and route_c_parity["ok"]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if all_ok else "fail",
        "route_a": {
            "ok": all(r.ok for r in route_a_results),
            "comparisons": results_to_dict(route_a_results),
        },
        "route_b": {
            "ok": all(r.ok for r in route_b_results),
            "comparisons": results_to_dict(route_b_results),
        },
        "route_c": {
            "ok": all(r.ok for r in route_c_results) and route_c_parity["ok"],
            "comparisons": results_to_dict(route_c_results),
            "from_arrow_large_offsets_parity": route_c_parity,
        },
    }

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        report_display = str(report_path.relative_to(REPO_ROOT))
    except ValueError:
        report_display = str(report_path)
    print(f"report: {report_display}")
    print(f"status: {report['status']}")

    if not all_ok:
        print("failing checks:")
        for item in all_results:
            if not item.ok:
                print(f"- {item.generated} vs {item.expected}: {item.reason}")
        if not route_c_parity["ok"]:
            print("- route_c from_arrow parity mismatch")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
