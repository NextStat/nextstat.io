---
title: "Simplified Likelihood Artifacts"
status: stable
---

# Simplified Likelihood Artifacts

This repo now defines versioned JSON contracts for the first public
simplified-likelihood reinterpretation slice:

- `simplified_likelihood_v0` — input contract for reduced likelihood specs
- `simplified_likelihood_audit_v0` — audit artifact emitted by `nextstat audit --format json`
- `simplified_likelihood_derive_v0` — research-grade derive/export planning contract for reducing a fitted full workspace into a simplified-likelihood artifact
- `simplified_likelihood_export_report_v0` — research-grade machine-readable export report emitted by `nextstat simplify workspace --report ...`
- `simplified_likelihood_report_v0` — Apex2 fidelity/speedup verification artifact emitted by `tests/apex2_simplified_likelihood_report.py`
- `simplified_likelihood_public_fixture_catalog_v0` — Apex2 catalog for curated public-style simplified-likelihood consume fixtures
- `simplified_likelihood_export_public_case_catalog_v0` — Apex2 catalog for curated public-style `full -> derived -> reinterpret` exporter cases
- `simplified_likelihood_promotion_evidence_bundle_v0` — validator-facing evidence bundle built from the frozen Apex2 report and release-hardening docs
- `simplified_likelihood_promotion_evidence_check_v0` — machine-readable verification report for `promotion_evidence.json` integrity and promotion-readiness checks
- `simplified_likelihood_promotion_bundle_promotion_report_v0` — machine-readable promotion/persistence report for moving an accepted evidence bundle under the stable `benchmarks/artifacts/.../accepted` path
- `simplified_likelihood_export_benchmark_snapshot_report_v0` — machine-readable persistence report for copying a bench-host exporter Apex2 report under a committed `benchmarks/artifacts/.../current` path
- `simplified_likelihood_export_public_validation_report_v0` — machine-readable stable evidence report for the curated public reinterpretation-style exporter matrix embedded in the committed bench-host exporter snapshot
- `simplified_likelihood_exporter_promotion_evidence_bundle_v0` — machine-readable exporter promotion-readiness bundle built from the committed exporter snapshot and exporter governance docs
- `simplified_likelihood_exporter_promotion_evidence_check_v0` — machine-readable verification report for exporter `promotion_evidence.json` integrity and readiness checks
- `simplified_likelihood_exporter_promotion_bundle_promotion_report_v0` — machine-readable persistence report for moving an accepted exporter bundle under the committed exporter accepted path
- `simplified_likelihood_exporter_stable_review_assessment_v0` — machine-readable formal stable-review assessment built from the committed accepted exporter bundle while keeping the exporter support class at `research-grade`
- `simplified_likelihood_exporter_stable_evidence_policy_v0` — machine-readable release-facing admission policy and maintenance cadence for the accepted exporter stable-evidence floor
- `simplified_likelihood_exporter_stable_source_semantics_boundary_v0` — machine-readable published stable source-semantics boundary for the narrow exporter path
- `simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0` — machine-readable blocker matrix that states the remaining delta between the current exporter evidence package and any broader stable claim beyond the promoted narrow subset
- `simplified_likelihood_exporter_stable_candidate_review_packet_v0` — machine-readable validator-facing packet that merges accepted exporter evidence, blocker state, and maintainer recommendation
- `simplified_likelihood_exporter_stable_promotion_decision_v0` — machine-readable explicit release-facing decision that promotes the narrow exporter subset to `stable` while preserving research-grade fallback modes outside that boundary

Schema versions:

- `nextstat_simplified_likelihood_v0`
- `nextstat_simplified_likelihood_audit_v0`
- `nextstat_simplified_likelihood_derive_v0`
- `nextstat_simplified_likelihood_export_report_v0`
- `nextstat_apex2_simplified_likelihood_report_v0`
- `nextstat_simplified_likelihood_public_fixture_catalog_v0`
- `nextstat_simplified_likelihood_export_public_case_catalog_v0`
- `nextstat_simplified_likelihood_promotion_evidence_bundle_v0`
- `nextstat_simplified_likelihood_promotion_evidence_check_v0`
- `nextstat_simplified_likelihood_promotion_bundle_promotion_report_v0`
- `nextstat_simplified_likelihood_export_benchmark_snapshot_report_v0`
- `nextstat_simplified_likelihood_export_public_validation_report_v0`
- `nextstat_simplified_likelihood_exporter_promotion_evidence_bundle_v0`
- `nextstat_simplified_likelihood_exporter_promotion_evidence_check_v0`
- `nextstat_simplified_likelihood_exporter_promotion_bundle_promotion_report_v0`
- `nextstat_simplified_likelihood_exporter_stable_review_assessment_v0`
- `nextstat_simplified_likelihood_exporter_stable_evidence_policy_v0`
- `nextstat_simplified_likelihood_exporter_stable_source_semantics_boundary_v0`
- `nextstat_simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0`
- `nextstat_simplified_likelihood_exporter_stable_candidate_review_packet_v0`
- `nextstat_simplified_likelihood_exporter_stable_promotion_decision_v0`

Published schemas are available from the CLI:

```bash
nextstat config schema --name simplified_likelihood_v0
nextstat config schema --name simplified_likelihood_audit_v0
nextstat config schema --name simplified_likelihood_derive_v0
nextstat config schema --name simplified_likelihood_export_report_v0
nextstat config schema --name simplified_likelihood_promotion_evidence_bundle_v0
nextstat config schema --name simplified_likelihood_promotion_evidence_check_v0
nextstat config schema --name simplified_likelihood_promotion_bundle_promotion_report_v0
nextstat config schema --name simplified_likelihood_export_benchmark_snapshot_report_v0
nextstat config schema --name simplified_likelihood_export_public_validation_report_v0
nextstat config schema --name simplified_likelihood_exporter_promotion_evidence_bundle_v0
nextstat config schema --name simplified_likelihood_exporter_promotion_evidence_check_v0
nextstat config schema --name simplified_likelihood_exporter_promotion_bundle_promotion_report_v0
nextstat config schema --name simplified_likelihood_exporter_stable_review_assessment_v0
nextstat config schema --name simplified_likelihood_exporter_stable_evidence_policy_v0
nextstat config schema --name simplified_likelihood_exporter_stable_source_semantics_boundary_v0
nextstat config schema --name simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0
nextstat config schema --name simplified_likelihood_exporter_stable_candidate_review_packet_v0
nextstat config schema --name simplified_likelihood_exporter_stable_promotion_decision_v0
```

Example payloads live in:

- `docs/specs/hep/simplified_likelihood_v0.example.json`
- `docs/specs/hep/simplified_likelihood_covariance_public_v0.example.json`
- `docs/specs/hep/simplified_likelihood_derived_from_workspace_v0.example.json`
- `docs/specs/hep/simplified_likelihood_audit_v0.example.json`
- `docs/specs/hep/simplified_likelihood_derive_v0.example.json`
- `docs/specs/hep/simplified_likelihood_export_report_v0.example.json`
- `docs/specs/apex2_simplified_likelihood_report_v0.example.json`
- `docs/specs/apex2_simplified_likelihood_public_fixture_catalog_v0.example.json`
- `docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_check_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_export_public_validation_report_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.example.json`

Current CLI surface:

```bash
nextstat audit --input simplified.json
nextstat audit --input simplified.json --format json --output simplified_audit.json
nextstat fit --input simplified.json
nextstat hypotest --input simplified.json --mu 1.0
nextstat upper-limit --input simplified.json
nextstat scan --input simplified.json --points 21 --start 0.0 --stop 5.0
```

Current promoted narrow exporter runtime:

```bash
nextstat simplify workspace \
  --input workspace.json \
  --fit fit.json \
  --derive-config derive.json \
  --experiment ATLAS \
  --analysis-id analysis.sl.v0 \
  --reference internal-note \
  --report export_report.json \
  --output simplified.json
```

Key derive/export knob in March 2026:

- `reduction.constraint_covariance_source = "source_model_constraints"` reuses the source model's Gaussian nuisance widths when the selected nuisance surface is Gaussian-constrained; this is the bench-backed path for `full -> derived -> reinterpret`
- `reduction.constraint_covariance_source = "aligned_fit_covariance"` remains available as a compatibility fallback for sources with non-Gaussian or unconstrained nuisance terms, but it stays a lower-trust research-grade mode

Current Apex2 verification surface:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
  tests/apex2_simplified_likelihood_report.py \
  --suite ci \
  --out tmp/apex2_simplified_likelihood_report.json
```

Optional public-style runtime matrix:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
  tests/apex2_simplified_likelihood_report.py \
  --suite ci \
  --include-public-fixtures \
  --out tmp/apex2_simplified_likelihood_report_with_public_matrix.json
```

Canonical remote EPYC run:

```bash
BENCH_HOST=nextstat-bench \
BENCH_SUITE=bench \
bash scripts/benchmarks/apex2_simplified_likelihood_remote.sh
```

Current v0 scope:

- basis-form simplified likelihoods with explicit reduced nuisance components
- covariance-form simplified likelihoods with deterministic eigendecomposition
- reduced HistFactory compilation for `fit`, `hypotest`, `upper-limit`, and `scan`
- audit JSON with channel/bin counts, yield summaries, reduced nuisance count, and factorization diagnostics
- export report JSON with derive-config summary, output digest/size, retained-vs-full nuisance counts, and fidelity/factorization diagnostics
- explicit derive knob `reduction.constraint_covariance_source` so export semantics are machine-readable instead of hidden inside the runtime
- Apex2 report JSON with schema validation, factorization residuals, fidelity gates, and reinterpretation speedup summary
- curated public-style consume fixtures cataloged for basis, covariance, and derived simplified-likelihood examples
- optional Apex2 public-fixture matrix embedded in the report for runtime evidence on curated public-style fixtures
- validator-facing evidence bundle builder that copies the accepted docs/contracts plus the chosen Apex2 report into one handoff directory with `promotion_evidence.json`
- bundle generator: `scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py`

Current support class in March 2026:

- `stable` for `audit`, `fit`, `hypotest`, `upper-limit`, and `scan`
- `research-grade` for discovery-style outputs, toy CLs, ranking semantics, wider exporter fallback modes outside the published source boundary, covariance-only source semantics, and real-world external validation beyond the current promotion evidence
- the stable exporter source boundary is now published separately and remains narrow: `pyhf` source only, single-POI only, Gaussian-constrained `source_model_constraints`, and reduced-coordinate rather than source-level nuisance semantics

Stable-surface acceptance policy and current benchmark evidence:

- [Simplified Likelihood Stable-Surface Acceptance](/docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Support Matrix](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)
- [Simplified Likelihood Exporter Acceptance](/docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md)
- [Simplified Likelihood Exporter Runtime Gate](/docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md)
- [Simplified Likelihood Exporter Promotion Runbook](/docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Evidence Policy](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Evidence Freshness](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Source-Semantics Boundary](/docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md)

Curated public-style fixture catalog:

- schema: `docs/schemas/apex2/simplified_likelihood_public_fixture_catalog_v0.schema.json`
- catalog: `docs/specs/apex2_simplified_likelihood_public_fixture_catalog_v0.example.json`
- smoke test: `tests/python/test_simplified_likelihood_public_fixture_catalog_smoke.py`
- Apex2 report integration: `tests/apex2_simplified_likelihood_report.py --include-public-fixtures`
- boundary: the catalog now feeds an Apex2 public-style runtime matrix, but it is still not the bench-host promotion artifact and does not replace the paired synthetic fidelity/speedup matrix

Validator-facing evidence bundle:

- schema: `docs/schemas/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.schema.json`
- example: `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json`
- builder: `scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py`
- emitted bundle entrypoint: `promotion_evidence.json`
- purpose: copy the accepted stable-surface docs/contracts together with the selected Apex2 report into one hash-inventoried handoff directory for validator review
- boundary: this is frozen-artifact packaging only; it does not rerun fits, regenerate Apex2 numbers, or widen the promoted simplified-likelihood subset

Promotion evidence verification report:

- schema: `docs/schemas/benchmarks/simplified_likelihood_promotion_evidence_check_v0.schema.json`
- example: `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_check_v0.example.json`
- verifier: `scripts/benchmarks/verify_simplified_likelihood_promotion_evidence_bundle.py`
- emitted report: `promotion_evidence_check.json`
- purpose: validate bundle schema, copied-file inventory, SHA-256/byte counts, and promotion-ready claims before release/promotion review consumes the bundle
- boundary: this verifies a frozen bundle; it does not replace the canonical `nextstat-bench` run or generate new benchmark numbers

Accepted promotion bundle persistence:

- schema: `docs/schemas/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.schema.json`
- example: `docs/specs/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.example.json`
- promoter: `scripts/benchmarks/promote_simplified_likelihood_promotion_bundle.py`
- stable accepted bundle path:
  `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/`
- stable accepted bundle entrypoints:
  - `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json`
  - `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json`
  - `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/snapshot_index.json`
- stable archive path:
  `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/history/`
- purpose: move a promotion-ready frozen bundle from `tmp/` into a committed, hash-indexed accepted location without manual file copying
- boundary: this does not rerun Apex2 or recompute benchmark numbers; it only persists already verified evidence

Committed exporter benchmark persistence:

- schema: `docs/schemas/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.schema.json`
- example: `docs/specs/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.example.json`
- persistence script: `scripts/benchmarks/persist_simplified_likelihood_export_benchmark.py`
- stable committed current path:
  `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/`
- stable committed current entrypoints:
  - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json`
  - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json`
  - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json`
  - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json`
- history path:
  `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/history/`
- published history directories referenced by hygiene checks:
  - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/history/current_20260309T180800Z_previous`
  - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/history/current_20260310T000000Z_previous`
  - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/history/snapshot_20260310T000000Z_persisted`
  - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/history/snapshot_20260311T170606Z_persisted`
- purpose: persist the current bench-host `full -> derived -> reinterpret` exporter evidence under a committed audit-friendly path without claiming stable promotion of the exporter runtime itself
- boundary: this is research-grade benchmark persistence only; it does not widen the promoted stable subset and does not replace rerunning `nextstat-bench` when exporter math changes
- acceptance/gate docs:
  - `docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md`
  - `docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md`
- current March 11, 2026 committed exporter evidence:
  - `export_matrix_case_count = 11`
  - `export_matrix_public_reinterpretation_style_case_count = 9`
  - `export_matrix_min_net_end_to_end_upper_limit_speedup = 0.8777103422927768x`
  - `export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup = 2.1792428549098894x`
  - `export_matrix_public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup = 0.8777103422927768x`

Exporter public case catalog:

- schema: `docs/schemas/apex2/simplified_likelihood_export_public_case_catalog_v0.schema.json`
- catalog: `docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json`
- helper: `tests/python/_simplified_likelihood_export_public_case_catalog.py`
- Apex2 report integration:
  `tests/apex2_simplified_likelihood_report.py --include-export-matrix --include-export-public-cases`
- boundary: this catalog drives the research-grade exporter matrix only; it is
  committed `nextstat-bench` evidence for future stable review, not a stable
  exporter promotion by itself

Exporter public validation surface:

- schema: `docs/schemas/benchmarks/simplified_likelihood_export_public_validation_report_v0.schema.json`
- example: `docs/specs/benchmarks/simplified_likelihood_export_public_validation_report_v0.example.json`
- builder: `scripts/benchmarks/build_simplified_likelihood_export_public_validation_report.py`
- stable committed current entrypoint:
  `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json`
- surface note:
  `docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09.md`
- purpose: publish the curated public reinterpretation-style exporter matrix as a
  first-class stable evidence artifact instead of leaving it implicit inside the
  larger Apex2 exporter report
- boundary: this is a stable evidence surface, not a widened runtime support
  claim; the committed public case now stays inside the promoted runtime
  boundary on the `source_model_constraints` path, but this surface remains a
  separate fidelity-first evidence lane rather than the synthetic
  stable-review performance floor

Accepted exporter promotion-readiness bundle:

- schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json`
- example: `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json`
- builder: `scripts/benchmarks/build_simplified_likelihood_exporter_promotion_evidence_bundle.py`
- verifier schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.schema.json`
- verifier example: `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.example.json`
- verifier: `scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py`
- promotion report schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.schema.json`
- promotion report example: `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.example.json`
- promoter: `scripts/benchmarks/promote_simplified_likelihood_exporter_promotion_bundle.py`
- stable-review assessment schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.schema.json`
- stable-review assessment example: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.example.json`
- stable-review assessment generator: `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py`
- stable-evidence policy schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json`
- stable-evidence policy example: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json`
- stable-evidence policy generator: `scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py`
- stable-evidence freshness schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json`
- stable-evidence freshness example: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json`
- stable-evidence freshness generator: `scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py`
- stable-source-semantics boundary schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json`
- stable-source-semantics boundary example: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json`
- stable-source-semantics boundary generator: `scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py`
- stable-candidate blocker-matrix schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.schema.json`
- stable-candidate blocker-matrix example: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json`
- stable-candidate blocker-matrix generator: `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py`
- stable-candidate review-packet schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.schema.json`
- stable-candidate review-packet example: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json`
- stable-candidate review-packet generator: `scripts/benchmarks/build_simplified_likelihood_exporter_stable_candidate_review_packet.py`
- stable-promotion decision schema: `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json`
- stable-promotion decision example: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.example.json`
- stable-promotion decision generator: `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_promotion_decision.py`
- stable-review checklist:
  `docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md`
- stable-evidence policy note:
  `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md`
- stable-evidence freshness note:
  `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md`
- stable-source-semantics boundary note:
  `docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md`
- stable-candidate blocker-matrix note:
  `docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md`
- stable-candidate review-packet note:
  `docs/benchmarks/simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md`
- stable-promotion decision note:
  `docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md`
- exporter release PR checklist:
  `docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md`
- stable accepted bundle path:
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/`
- stable accepted bundle entrypoints:
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/snapshot_index.json`
- stable archive path:
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/history/`
- published accepted-history directories referenced by hygiene checks:
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/history/accepted_exporter-20260310T001421Z-public-floor-five_promoted`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/history/accepted_exporter-20260311T170606Z-public-floor-six_promoted`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/history/accepted_exporter-20260311T170606Z-public-floor-six-refresh_promoted`
  - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/history/accepted_exporter-20260311T170606Z-public-floor-six-refresh2_promoted`
- purpose: freeze exporter promotion-readiness evidence under a committed accepted path, derive the formal governance artifacts, and record the explicit release-facing stable decision for the narrow exporter subset
- boundary: this persists verified exporter evidence and the explicit stable decision for the narrow subset; it does not widen any research-grade fallback beyond the published source-semantics boundary
- current March 9, 2026 accepted blocker state:
  - `public_exporter_matrix_not_yet_part_of_stable_candidate_evidence = resolved`
  - `stable_source_semantics_boundary_not_yet_promoted = resolved`
  - `stable_candidate_review_packet_not_yet_published = resolved`
  - `stable_release_promotion_decision_not_yet_taken = resolved`
  - `stable_candidate.open_blocker_count = 0`

Important boundaries in March 2026:

- `nextstat audit` in the CLI accepts pyhf and simplified-likelihood JSON
- `nextstat.workspace_audit(...)` in Python accepts pyhf and simplified-likelihood JSON
- `nextstat_workspace_audit` in the server/tool surface accepts pyhf and simplified-likelihood JSON
- the promoted simplified-likelihood stable subset is `audit`, `fit`, `hypotest`, `upper-limit`, and `scan`
- the published derive/export contract is `stable` for the narrow exporter subset: `pyhf` source only, single-POI only, Gaussian-constrained `source_model_constraints`, explicit provenance, and reduced-coordinate semantics
- the matching export report contract is `stable` within that same narrow boundary: `nextstat_simplified_likelihood_export_report_v0` is versioned and emitted by `nextstat simplify workspace --report ...`
- derived simplified-likelihood artifacts use `metadata.source_format = "derived_from_workspace"` and must carry provenance plus fidelity diagnostics
- `nextstat simplify workspace` is currently `pyhf`-only on the source side and rejects partial per-channel bin selections explicitly instead of silently degrading semantics
- the published stable exporter claim is intentionally narrow: `pyhf` source only, single-POI only, and `reduction.constraint_covariance_source = "source_model_constraints"` for Gaussian-constrained nuisance sources only
- `reduction.constraint_covariance_source = "aligned_fit_covariance"` is retained as a research-grade fallback when source constraints are not Gaussian-constrained
- discovery, ranking, and toy CLs remain callable for simplified-likelihood inputs but stay outside the promoted stable subset
- ranking on simplified-likelihood operates on reduced nuisance coordinates from the compiled model; names and impacts must not be interpreted as source-level systematics
- covariance-form simplified-likelihoods expose synthetic reduced eigendirections only, so source-level ranking semantics are unavailable
- `derived_from_workspace` v0 artifacts do not preserve original nuisance identities through reduction, so ranking/impact outputs are not a source-level breakdown
- HS3 is rejected explicitly across all audit surfaces
- structural/typing guarantees are expressed in JSON Schema
- numeric invariants such as PSD checks, matrix symmetry tolerance, and vector-length agreement remain enforced by runtime validation

Acceptance criteria for the current stable subset:

- `abs(mu_hat_sl - mu_hat_full) / sigma_mu_full <= 0.05`
- `max_abs(q_mu_sl - q_mu_full) <= 0.1`
- `upper_limit_sl / upper_limit_full` in `[0.95, 1.05]`
- `reduced_nuisance_count / full_nuisance_count <= 0.25`
- `simplified_json_bytes / full_workspace_json_bytes <= 0.35`
- CI gate: end-to-end upper-limit speedup `>= 3x`
- bench-host promotion gate on `nextstat-bench`: end-to-end upper-limit speedup `>= 10x`

Operational gate surface:

```bash
make simplified-likelihood-stable-surface-gate
```

- workflow: `.github/workflows/simplified-likelihood-stable-surface.yml`
- release checklist: `docs/benchmarks/simplified-likelihood-release-pr-checklist-2026-03-08.md`
- promotion runbook: `docs/benchmarks/simplified-likelihood-promotion-runbook-2026-03-08.md`

Recommended operator flow:

1. Validate or generate `simplified_likelihood_v0`
2. Run `nextstat audit --format json` to produce `simplified_likelihood_audit_v0`
3. Use the same input with `nextstat fit` / `nextstat hypotest` / `nextstat upper-limit` / `nextstat scan`
4. Run `tests/apex2_simplified_likelihood_report.py` to benchmark reduced-vs-full fidelity and speedup
5. If external/public-style consume coverage matters for the change, rerun the same Apex2 report with `--include-public-fixtures`
6. If the change touches derive/export semantics, rerun the same Apex2 report with `--include-export-matrix`
7. Persist the bench-host exporter artifact with `persist_simplified_likelihood_export_benchmark.py`
8. Run `make simplified-likelihood-exporter-surface-gate` before claiming exporter promotion-readiness
9. Build the accepted exporter bundle from the committed exporter snapshot with `build_simplified_likelihood_exporter_promotion_evidence_bundle.py`
10. Verify the exporter bundle with `verify_simplified_likelihood_exporter_promotion_evidence_bundle.py`
11. Promote the exporter bundle under `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/`
12. Regenerate the formal stable-review assessment with `assess_simplified_likelihood_exporter_stable_review.py`
13. Regenerate the explicit delta-to-stable blocker matrix with `assess_simplified_likelihood_exporter_stable_candidate_blockers.py`
14. Build the validator-facing consume-path evidence bundle from the accepted Apex2 artifact
15. Verify that bundle with `verify_simplified_likelihood_promotion_evidence_bundle.py`
16. Promote the accepted consume-path bundle under `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/`
17. Archive the audit JSON, Apex2 report JSON, exporter benchmark snapshot report, exporter promotion evidence bundle, exporter promotion evidence check report, and accepted bundle promotion report alongside reinterpretation outputs for reproducibility

Promoted narrow derive/export surface:

```bash
nextstat config schema --name simplified_likelihood_derive_v0
nextstat config schema --name simplified_likelihood_export_report_v0
```

- this schema defines the intended derive/export request contract
- the matching runtime path is `nextstat simplify workspace`
- the matching machine-readable runtime report path is `nextstat simplify workspace --report export_report.json`
- that runtime path is versioned, tested, and promoted for the narrow exporter subset only
- broader research-grade fallback modes remain outside that promoted subset
