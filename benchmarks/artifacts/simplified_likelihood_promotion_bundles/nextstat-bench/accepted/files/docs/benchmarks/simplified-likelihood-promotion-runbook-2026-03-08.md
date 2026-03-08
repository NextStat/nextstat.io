# Simplified Likelihood Promotion Runbook

**Date**: 2026-03-08  
**Status**: promotion runbook  
**Scope**: `nextstat-bench` promotion evidence for the stable simplified-likelihood surface

## Purpose

This runbook defines the canonical promotion workflow for the simplified-likelihood
stable subset after the CI gate is green.

It answers one operational question:

- how do maintainers produce and review the bench-host evidence required for the
  `~10x reinterpretation speedup` stable-surface claim?

## Ownership

Promotion requires three explicit roles:

- release owner: decides whether the stable claim is made in the release PR
- bench operator: runs the canonical `nextstat-bench` benchmark artifact
- reviewer: confirms the artifact satisfies the March 8, 2026 acceptance policy

For small releases, the same maintainer may hold all three roles, but the
artifact review still needs to be explicit in the PR.

## When this runbook is required

Run this promotion workflow when a change affects any of:

- simplified-likelihood schema contracts
- validation/runtime rules in `ns-translate` for simplified likelihoods
- simplified-likelihood CLI consume/audit paths
- Python or server simplified-likelihood audit dispatch
- Apex2 simplified-likelihood report logic or case matrix
- benchmark claims or public docs that restate the performance envelope

Docs-only edits that do not change claims or acceptance wording do not require a
new bench-host run.

## Artifact contract

The promotion artifact is the machine-readable Apex2 JSON report:

- Schema: `docs/schemas/apex2/simplified_likelihood_report_v0.schema.json`
- Example: `docs/specs/apex2_simplified_likelihood_report_v0.example.json`
- Runner: `tests/apex2_simplified_likelihood_report.py`
- Remote runner: `scripts/benchmarks/apex2_simplified_likelihood_remote.sh`

The validator-facing handoff bundle is:

- Schema: `docs/schemas/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.schema.json`
- Example: `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json`
- Builder: `scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py`
- Emitted bundle entrypoint: `promotion_evidence.json`

The validator-facing promotion evidence bundle is a derived companion artifact:

- Schema: `docs/schemas/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.schema.json`
- Example: `docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json`
- Builder: `scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py`
- Output summary: `promotion_evidence.json`

The current published evidence note is:

- `docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md`
- `docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md`
- `docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md`

## Precondition

Before touching `nextstat-bench`, the CI/local gate must already be green:

```bash
make simplified-likelihood-stable-surface-gate
```

This ensures the bench-host run is promotion evidence, not a substitute for the
standard correctness gate.

## Canonical nextstat-bench run

Recommended command:

```bash
BENCH_HOST=nextstat-bench \
BENCH_SUITE=bench \
bash scripts/benchmarks/apex2_simplified_likelihood_remote.sh
```

The canonical remote runner now includes the curated public-style fixture matrix
by default. Override only for legacy comparisons:

```bash
BENCH_INCLUDE_PUBLIC_FIXTURES=0 \
BENCH_HOST=nextstat-bench \
BENCH_SUITE=bench \
bash scripts/benchmarks/apex2_simplified_likelihood_remote.sh
```

If the local SSH alias is not configured correctly, pass explicit overrides:

```bash
BENCH_HOST=<host> \
BENCH_SSH_USER=<user> \
BENCH_SSH_PORT=<port> \
BENCH_SSH_KEY=/path/to/key \
BENCH_SUITE=bench \
bash scripts/benchmarks/apex2_simplified_likelihood_remote.sh
```

Recommended preserved output layout:

- local synced artifact dir: `tmp/apex2_simplified_likelihood_<timestamp>/<host>/`
- report JSON: `apex2_simplified_likelihood_report.json`
- if public fixtures are enabled, the same report JSON also carries
  `public_fixture_matrix` with runtime evidence for curated public-style
  basis/covariance/derived examples

Recommended validator bundle layout after the benchmark passes:

- bundle dir: `tmp/simplified_likelihood_promotion_evidence_bundle_<timestamp>/<host>/`
- entry JSON: `promotion_evidence.json`

## Pass conditions

Promotion passes only if all of the following are true in the bench-host artifact:

- `summary.status == "ok"`
- `summary.all_schema_valid == true`
- `summary.all_fidelity_gates_pass == true`
- `summary.all_performance_gates_pass == true`
- `summary.bench.min_speedup_end_to_end_upper_limit >= 10.0`
- if `public_fixture_matrix` is present:
  - `public_fixture_matrix.summary.status == "ok"`
  - `public_fixture_matrix.summary.all_runtime_gates_pass == true`
  - `public_fixture_matrix.summary.all_derived_fidelity_gates_pass == true`

This is stricter than the CI gate, which only requires `>= 3x`.

## Review checklist

The reviewer must confirm:

- the artifact came from `nextstat-bench`
- the artifact path is linked from the benchmark snapshot note
- the published snapshot note quotes the artifact, not terminal-only output
- the support matrix and release note do not widen the claim beyond the promoted subset
- no public release note widens the claim beyond the current stable subset

## Promotion output

Once the run passes:

1. update or confirm the benchmark snapshot note
2. build the promotion evidence bundle from the accepted Apex2 artifact
3. link both the Apex2 artifact and `promotion_evidence.json` from the release PR
4. state whether the `~10x` speedup claim is still justified
5. note any deviation from the previous accepted artifact

Canonical bundle build command:

```bash
python3 scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py \
  --benchmark-artifact tmp/apex2_simplified_likelihood_<timestamp>/<host>/apex2_simplified_likelihood_report.json \
  --bundle-dir tmp/simplified_likelihood_promotion_evidence_bundle_<timestamp>/<host> \
  --deterministic
```

## Failure handling

If the bench-host run fails:

- do not make the `~10x reinterpretation speedup` stable-surface claim
- keep the surface at its current support class
- record the failed artifact path in the PR for traceability
- open or update follow-up tasks before promotion proceeds
