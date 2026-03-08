# NextStat Docs Index

This repo uses plain Markdown docs. Start here and then jump into the track that matches what you're doing.

## Roadmap

- Project roadmap, milestones, and known limitations: `docs/ROADMAP.md`

## Start Here

- Tutorials index (end-to-end workflows): `docs/tutorials/README.md`
- HEP GVM measurement-combination tutorial (stable-first core, advanced layers called out separately): `docs/tutorials/hep-gvm-measurement-combinations.md`
- Quickstarts (10 minutes to result): `docs/quickstarts/README.md`
- HEP GVM stable-first quickstart (5-minute first result from the committed example bundle; includes `make gvm-stable-first-example`): `docs/quickstarts/hep-gvm-stable-first.md`
- Adoption playbook (HEP routes A/B/C/D): `docs/guides/README.md`
- External validation kit for the stable-first GVM path: `docs/guides/gvm-external-validation-kit.md`
- External validator outreach pack for the stable-first GVM path: `docs/guides/gvm-external-validator-outreach-pack.md`
- External validation tracker template for the stable-first GVM path: `docs/guides/gvm-external-validation-tracker-template.md`
- Python reference: `docs/references/python-api.md`
- Python packaging (wheels/extras): `docs/references/python-packaging.md`
- Arrow / Parquet I/O (histogram tables): `docs/references/arrow-parquet-io.md`
- CLI reference: `docs/references/cli.md`
- Rust reference: `docs/references/rust-api.md`
- Terminology and style guide: `docs/references/terminology.md`
- Glossary (cross-domain term definitions): `docs/references/glossary.md`
- RNTuple effort estimate (minimal/converter/full): `docs/references/rntuple-minimal-reader-estimate.md`
- RNTuple compatibility matrix (verified rows + CI gates): `docs/references/rntuple-compatibility-matrix.md`
- RNTuple rollout/migration notes (v1 scope + limits): `docs/references/rntuple-rollout-v1.md`
- RNTuple benchmark note (`epyc-node`, `ns-root` vs ROOT): `docs/benchmarks/rntuple-epyc-node-2026-02-16.md`
- RNTuple mixed-layout verification addendum (`2,000,000` entries, release perf-gate): `docs/benchmarks/rntuple-epyc-node-2026-02-16.md`
- RNTuple reproducible comparison harness (`make rntuple-root-vs-nsroot`): `scripts/benchmarks/run_rntuple_root_vs_nsroot.sh`

## Demos

- Physics Assistant demo (ROOT -> anomaly scan -> p-values + plots): `docs/demos/physics-assistant.md`

## Benchmarks and Trust Artifacts

- Benchmarks hub: `docs/benchmarks.md`
- GVM benchmark snapshot (Apple M5 + AMD EPYC): `docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md`
- GVM NumericalPaper robustness snapshot (mixed literature + synthetic tiers): `docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md`
- GVM stable-surface readiness memo: `docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md`
- GVM stable-surface support policy: `docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md`
- GVM stable-first promotion decision: `docs/benchmarks/gvm-stable-first-decision-2026-03-07.md`
- GVM stable-first support matrix: `docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md`
- GVM stable-first release notes: `docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md`
- GVM stable-first release candidate (`v0.10.0`): `docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md`
- Public benchmark suites (seed repo): `benchmarks/nextstat-public-benchmarks/`
- Validation report (JSON/PDF contract): `docs/references/validation-report.md`
- ICH M15 reporting artifacts (schema contracts): `docs/references/m15-reporting.md`
- Bayesian trial design artifacts (schema contracts): `docs/references/bayesian-trial-design-artifacts.md`
- Bayesian design report acceptance criteria: `docs/specs/pharma/bayesian_design_report_acceptance_v0.md`
- Bayesian prior conflict diagnostic acceptance criteria: `docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md`
- Bayesian historical-control borrowing review acceptance criteria: `docs/specs/pharma/bayesian_historical_control_borrowing_review_acceptance_v0.md`
- Bayesian robust-mixture prior review acceptance criteria: `docs/specs/pharma/bayesian_robust_mixture_prior_review_acceptance_v0.md`
- Bayesian design regulatory appendix acceptance criteria: `docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md`
- Bayesian design appendix render acceptance criteria: `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`
- Bayesian design report bundle acceptance criteria: `docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md`
- Bayesian design validation-pack acceptance criteria: `docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md`
- Simplified likelihood artifacts (schema contracts): `docs/references/simplified-likelihood-artifacts.md`
- Simplified likelihood derive/export contract and export report (promoted narrow `stable` exporter subset with explicit `constraint_covariance_source` semantics and research-grade fallback outside the boundary): `docs/references/simplified-likelihood-artifacts.md`
- Simplified likelihood stable-surface acceptance: `docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md`
- Simplified likelihood stable-surface support matrix: `docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md`
- Simplified likelihood stable-surface release notes: `docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md`
- Simplified likelihood benchmark snapshot (`nextstat-bench`): `docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md`
- Simplified likelihood exporter acceptance (narrow `stable` exporter subset + explicit boundary): `docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md`
- Simplified likelihood exporter runtime gate (narrow `stable` subset + committed `nextstat-bench` evidence): `docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md`
- Simplified likelihood exporter promotion runbook (accepted bundle lifecycle for the promoted narrow subset): `docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md`
- Simplified likelihood exporter stable evidence policy (release-facing admission policy + maintenance cadence for the accepted `7 public / 9 total` floor): `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md`
- Simplified likelihood exporter stable evidence freshness (machine-readable 45-day freshness breach guard for the accepted exporter floor): `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md`
- Simplified likelihood exporter public validation surface (stable evidence for curated public exporter cases): `docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09.md`
- Simplified likelihood exporter stable-review checklist (historical review governance for the promoted narrow subset): `docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md`
- Simplified likelihood exporter stable source-semantics boundary (published narrow stable claim): `docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md`
- Simplified likelihood exporter stable-candidate blocker matrix (now zero-blocker governance artifact for the promoted narrow subset): `docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md`
- Simplified likelihood exporter stable-candidate review packet (validator-facing packet for the promoted narrow subset): `docs/benchmarks/simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md`
- Simplified likelihood exporter stable-promotion decision (explicit release-facing promotion decision): `docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md`
- Simplified likelihood exporter release PR checklist (release-facing governance for the promoted narrow subset): `docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md`
- Simplified likelihood exporter public case catalog (`research-grade`, Apex2 export matrix input): `docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json`
- Simplified likelihood exporter public validation report (`stable-evidence`): `docs/specs/benchmarks/simplified_likelihood_export_public_validation_report_v0.example.json`
- Simplified likelihood validator-facing promotion evidence bundle: `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json`
- Simplified likelihood promotion evidence verification report: `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_check_v0.example.json`
- Simplified likelihood promotion bundle persistence report: `docs/specs/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.example.json`
- Simplified likelihood exporter benchmark snapshot persistence report: `docs/specs/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.example.json`
- Simplified likelihood exporter promotion evidence bundle: `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json`
- Simplified likelihood exporter promotion evidence verification report: `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.example.json`
- Simplified likelihood exporter promotion bundle persistence report: `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.example.json`
- Simplified likelihood exporter stable-review assessment: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.example.json`
- Simplified likelihood exporter stable source-semantics boundary: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json`
- Simplified likelihood exporter stable-candidate blocker matrix: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json`
- Simplified likelihood exporter stable-candidate review packet: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json`
- Simplified likelihood exporter stable evidence policy: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json`
- Simplified likelihood exporter stable evidence freshness report: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json`
- Simplified likelihood exporter stable-promotion decision: `docs/specs/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.example.json`
- Simplified likelihood release PR checklist: `docs/benchmarks/simplified-likelihood-release-pr-checklist-2026-03-08.md`
- Simplified likelihood promotion runbook: `docs/benchmarks/simplified-likelihood-promotion-runbook-2026-03-08.md`
- Ads + Time Series stable-surface acceptance: `docs/benchmarks/ads-timeseries-stable-surface-acceptance-2026-03-08.md`
- Ads + Time Series stable-surface support matrix: `docs/benchmarks/ads-timeseries-support-matrix-2026-03-08.md`
- Ads + Time Series stable-surface release notes: `docs/benchmarks/ads-timeseries-release-notes-2026-03-08.md`
- Ads + Time Series runtime gate: `docs/benchmarks/ads-timeseries-runtime-gate.md`
- Ads + Time Series benchmark snapshot (`nextstat-bench`): `docs/benchmarks/ads-timeseries-benchmark-snapshot-2026-03-08.md`
- Ads + Time Series release PR checklist: `docs/benchmarks/ads-timeseries-release-pr-checklist-2026-03-08.md`
- Ads + Time Series promotion runbook: `docs/benchmarks/ads-timeseries-promotion-runbook-2026-03-08.md`
- Ads variance-reduction matrix runbook: `docs/benchmarks/ads-variance-reduction-runbook-2026-03-08.md`
- Ads variance-reduction benchmark note: `docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md`
- Ads variance-reduction stable-surface acceptance: `docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09.md`
- Ads variance-reduction runtime gate: `docs/benchmarks/ads-variance-reduction-runtime-gate.md`

## Whitepapers

- FDA Bayesian trial designs for drugs and biologics: `docs/whitepapers/fda-bayesian-trial-designs.md`
- NONMEM parity whitepaper: `docs/whitepapers/nonmem-parity.md`

## Tools and Server (LLM/Agent Integration)

- Tool API contract: `docs/references/tool-api.md`
- Server API (`/v1/tools/execute`, etc.): `docs/references/server-api.md`
- Plot artifacts (JSON): `docs/references/plot-artifacts.md`

## Neural Density Estimation

- Neural PDFs guide (FlowPdf, DcrSurrogate, training, ONNX): `docs/neural-density-estimation.md`
- Differentiable HistFactory (binned-likelihood workspace) layer for PyTorch: `docs/differentiable-layer.md`

## R Bindings

- R package reference (experimental): `docs/references/r-bindings.md`

## Arrow / Parquet

- Binned histogram Parquet schema (v2, with modifiers): `docs/references/binned-parquet-schema.md`
- Unbinned event-level Parquet schema (v1): `docs/references/unbinned-parquet-schema.md`

## Architecture Decisions (RFC/ADR)

- ADR-0001 RNTuple Support Policy: `docs/rfcs/rntuple-support-policy.md`
- RFC Research-Grade Measurement Combination Mode (GVM): `docs/rfcs/research-grade-measurement-combinations.md`
- RFC Simplified Likelihoods for HEP Reinterpretation: `docs/rfcs/simplified-likelihoods-reinterpretation.md`

## HPC / Cluster Deployment

- HTCondor & HPC cluster guide: `docs/guides/htcondor-hpc.md`
- HTCondor examples (.sub files, DAGMan): `docs/examples/htcondor/`
- Apptainer/Singularity containers: `containers/`

## GPU Support

- GPU contract and backend matrix: `docs/gpu-contract.md`

## Personas

These are navigation pages that map NextStat concepts and docs to non-HEP workflows.

- Data Scientists: `docs/personas/data-scientists.md`
- Quants: `docs/personas/quants.md`
- Biologists / Pharmacometricians: `docs/personas/biologists.md`

## Русскоязычная документация

- Индекс (RU): `docs/ru/README.md`
- Глоссарий (RU): `docs/ru/references/glossary.md`
