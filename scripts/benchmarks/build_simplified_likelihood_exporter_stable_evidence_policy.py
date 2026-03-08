#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _simplified_likelihood_export_benchmark import load_json
from _simplified_likelihood_exporter_stable_evidence_policy import build_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-artifact", type=Path, required=True)
    parser.add_argument("--public-validation-report", type=Path, required=True)
    parser.add_argument("--stable-promotion-decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    benchmark_artifact = args.benchmark_artifact.resolve()
    public_validation_report = args.public_validation_report.resolve()
    stable_promotion_decision = args.stable_promotion_decision.resolve()
    out_path = args.out.resolve()

    payload = build_policy(
        benchmark_artifact_path=benchmark_artifact,
        benchmark=load_json(benchmark_artifact),
        public_validation_report_path=public_validation_report,
        public_validation_report=load_json(public_validation_report),
        stable_promotion_decision_path=stable_promotion_decision,
        stable_promotion_decision=load_json(stable_promotion_decision),
        deterministic=bool(args.deterministic),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = payload["current_evidence_summary"]
    print(
        "Exporter stable evidence policy:",
        f"status={payload['status']}",
        f"public_cases={summary['public_case_count']}",
        f"export_cases={summary['export_matrix_case_count']}",
        sep=" ",
    )
    print(f"Policy written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
