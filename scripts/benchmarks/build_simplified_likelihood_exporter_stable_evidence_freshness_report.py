#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _simplified_likelihood_export_benchmark import load_json
from _simplified_likelihood_exporter_stable_evidence_freshness import (
    build_freshness_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-report", type=Path, required=True)
    parser.add_argument("--public-validation-report", type=Path, required=True)
    parser.add_argument("--stable-evidence-policy", type=Path, required=True)
    parser.add_argument("--stable-promotion-decision", type=Path, required=True)
    parser.add_argument("--reference-date", type=str, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    snapshot_report = args.snapshot_report.resolve()
    public_validation_report = args.public_validation_report.resolve()
    stable_evidence_policy = args.stable_evidence_policy.resolve()
    stable_promotion_decision = args.stable_promotion_decision.resolve()
    out_path = args.out.resolve()

    payload = build_freshness_report(
        snapshot_report_path=snapshot_report,
        snapshot_report=load_json(snapshot_report),
        public_validation_report_path=public_validation_report,
        public_validation_report=load_json(public_validation_report),
        stable_evidence_policy_path=stable_evidence_policy,
        stable_evidence_policy=load_json(stable_evidence_policy),
        stable_promotion_decision_path=stable_promotion_decision,
        stable_promotion_decision=load_json(stable_promotion_decision),
        reference_date=args.reference_date,
        deterministic=bool(args.deterministic),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = payload["summary"]
    print(
        "Exporter stable evidence freshness:",
        f"status={payload['status']}",
        f"snapshot_age_days={summary['snapshot_age_days']}",
        f"max_snapshot_age_days={summary['max_snapshot_age_days']}",
        sep=" ",
    )
    print(f"Freshness report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
