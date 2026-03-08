#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _simplified_likelihood_export_benchmark import load_json
from _simplified_likelihood_export_public_validation import (
    DEFAULT_CATALOG_PATH,
    build_public_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-artifact", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    benchmark_artifact = args.benchmark_artifact.resolve()
    catalog_path = args.catalog.resolve()
    out_path = args.out.resolve()

    report = build_public_validation_report(
        benchmark_artifact_path=benchmark_artifact,
        benchmark=load_json(benchmark_artifact),
        catalog_path=catalog_path,
        catalog=load_json(catalog_path),
        deterministic=bool(args.deterministic),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Simplified-likelihood export public validation report:",
        f"status={report['status']}",
        f"public_cases={report['summary']['public_case_count']}",
        "outside_promoted_runtime_boundary="
        f"{report['summary']['cases_outside_promoted_stable_runtime_boundary']}",
        sep=" ",
    )
    print(f"Report written to {out_path}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
