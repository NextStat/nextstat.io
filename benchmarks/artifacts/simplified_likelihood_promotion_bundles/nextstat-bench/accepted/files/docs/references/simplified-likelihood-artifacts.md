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
- `simplified_likelihood_report_v0` — Apex2 fidelity/speedup verification artifact emitted by `tests/apex2_simplified_likelihood_report.py`
- `simplified_likelihood_public_fixture_catalog_v0` — Apex2 catalog for curated public-style simplified-likelihood consume fixtures
- `simplified_likelihood_promotion_evidence_bundle_v0` — validator-facing evidence bundle built from the frozen Apex2 report and release-hardening docs

Schema versions:

- `nextstat_simplified_likelihood_v0`
- `nextstat_simplified_likelihood_audit_v0`
- `nextstat_simplified_likelihood_derive_v0`
- `nextstat_apex2_simplified_likelihood_report_v0`
- `nextstat_simplified_likelihood_public_fixture_catalog_v0`
- `nextstat_simplified_likelihood_promotion_evidence_bundle_v0`

Published schemas are available from the CLI:

```bash
nextstat config schema --name simplified_likelihood_v0
nextstat config schema --name simplified_likelihood_audit_v0
nextstat config schema --name simplified_likelihood_derive_v0
nextstat config schema --name simplified_likelihood_promotion_evidence_bundle_v0
```

Example payloads live in:

- `docs/specs/hep/simplified_likelihood_v0.example.json`
- `docs/specs/hep/simplified_likelihood_covariance_public_v0.example.json`
- `docs/specs/hep/simplified_likelihood_derived_from_workspace_v0.example.json`
- `docs/specs/hep/simplified_likelihood_audit_v0.example.json`
- `docs/specs/hep/simplified_likelihood_derive_v0.example.json`
- `docs/specs/apex2_simplified_likelihood_report_v0.example.json`
- `docs/specs/apex2_simplified_likelihood_public_fixture_catalog_v0.example.json`
- `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json`

Current CLI surface:

```bash
nextstat audit --input simplified.json
nextstat audit --input simplified.json --format json --output simplified_audit.json
nextstat fit --input simplified.json
nextstat hypotest --input simplified.json --mu 1.0
nextstat upper-limit --input simplified.json
nextstat scan --input simplified.json --points 21 --start 0.0 --stop 5.0
```

Current research-grade derive/export runtime:

```bash
nextstat simplify workspace \
  --input workspace.json \
  --fit fit.json \
  --derive-config derive.json \
  --experiment ATLAS \
  --analysis-id analysis.sl.v0 \
  --reference internal-note \
  --output simplified.json
```

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
- Apex2 report JSON with schema validation, factorization residuals, fidelity gates, and reinterpretation speedup summary
- curated public-style consume fixtures cataloged for basis, covariance, and derived simplified-likelihood examples
- optional Apex2 public-fixture matrix embedded in the report for runtime evidence on curated public-style fixtures
- validator-facing evidence bundle builder that copies the accepted docs/contracts plus the chosen Apex2 report into one handoff directory with `promotion_evidence.json`
- bundle generator: `scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py`

Current support class in March 2026:

- `stable` for `audit`, `fit`, `hypotest`, `upper-limit`, and `scan`
- `research-grade` for discovery-style outputs, toy CLs, ranking semantics, derive/export runtime workflows, covariance-only source semantics, and real-world external validation beyond the current promotion evidence

Stable-surface acceptance policy and current benchmark evidence:

- [Simplified Likelihood Stable-Surface Acceptance](/docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Support Matrix](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)

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

Important boundaries in March 2026:

- `nextstat audit` in the CLI accepts pyhf and simplified-likelihood JSON
- `nextstat.workspace_audit(...)` in Python accepts pyhf and simplified-likelihood JSON
- `nextstat_workspace_audit` in the server/tool surface accepts pyhf and simplified-likelihood JSON
- the promoted simplified-likelihood stable subset is `audit`, `fit`, `hypotest`, `upper-limit`, and `scan`
- the published derive/export contract is `research-grade`: schema, examples, and the `nextstat simplify workspace` runtime path are versioned and tested, but they are not promoted into the stable subset
- derived simplified-likelihood artifacts use `metadata.source_format = "derived_from_workspace"` and must carry provenance plus fidelity diagnostics
- `nextstat simplify workspace` is currently `pyhf`-only on the source side and rejects partial per-channel bin selections explicitly instead of silently degrading semantics
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
6. Build the validator-facing evidence bundle from the accepted Apex2 artifact
7. Archive the audit JSON, Apex2 report JSON, and promotion evidence bundle alongside reinterpretation outputs for reproducibility

Research-grade derive/export planning surface:

```bash
nextstat config schema --name simplified_likelihood_derive_v0
```

- this schema defines the intended derive/export request contract
- the matching runtime path is `nextstat simplify workspace`
- that runtime path is versioned and tested, but it is not promoted
  to the stable subset
