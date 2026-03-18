# Validation and Release Discipline

**Document ID:** NS-ENG-VAL-001
**Version:** 1.1.0
**Status:** Controlled public engineering reference
**Last updated:** 2026-03-18

This page is the canonical public overview of how NextStat turns a large
multi-surface repository into a governed, release-grade product.

It is intentionally not a replacement for the surface-specific support
matrices, acceptance contracts, benchmark snapshots, or release runbooks.
Instead, it explains how those pieces fit together.

For the maintainer-only shipping path, see the
[Release Runbook](../releases/release-runbook.md).

## Design Principle

NextStat does not treat "code exists" as evidence that a surface is ready.
A surface is considered trustworthy only after it passes the appropriate
combination of:

- deterministic correctness tests
- parity or reference checks
- stable-surface support and acceptance contracts
- benchmark or evidence capture
- prerelease governance checks
- release workflow integration

The acceptance criterion is therefore validation, not provenance.

In practice this means:

- automation and tooling may accelerate implementation and validation work
- provenance alone does not make a surface trustworthy
- the trust boundary is the documented validation and release discipline

## Control Planes

The system is easiest to understand as five cooperating control planes.

| Control Plane | Source of Truth | Purpose |
| --- | --- | --- |
| Implementation | Rust crates, Python bindings, CLI, scripts | Runtime behavior |
| Test and parity | `cargo test`, `pytest`, parity runners, schema checks | Correctness and numerical contract enforcement |
| Governance | support matrices, acceptance docs, release/HEP matrices | Defines what is publicly stable and what blocks release |
| Release | prerelease gate, release manifest, full-fidelity simulation, CI workflows | Ensures build, staging, and publish integrity |
| Evidence | benchmark snapshots, validation bundles, published release assets | Captures the accepted state for audit and review |

The key design choice is that no single control plane is allowed to silently
define release readiness on its own. A surface becomes stable only when the
implementation, governance, release, and evidence planes all line up.

## Repository Structure and Scope

The repository contains several different kinds of material:

- runtime implementation across Rust, Python bindings, CLI, and scripts
- test and fixture surface
- benchmark harnesses and comparison runners
- schemas and example artifacts
- release manifests, candidate bundles, and published evidence

The repository is intentionally larger than the runtime alone because it
stores the contracts and evidence needed to make stable-surface claims.

The result is a repository that contains both product code and the quality
system needed to govern that product code.

## Current Public Evidence Snapshot

Several important governance facts are already machine-tracked and publicly
documented:

- HEP stable inventory: `141/141 stable`, `0 research`
- required release surfaces: `6`
- HEP owner slices:
  - HistFactory: `48`
  - GVM: `47`
  - Infrastructure: `12`
  - Unbinned: `11`
  - Viz: `11`
  - Import/Export: `6`
  - Simplified Likelihood: `2`
  - HEPData: `2`
  - Preprocess: `2`
- current public HEP layer split:
  - CLI: `64`
  - Python: `52`
  - Server: `16`
  - Tool: `9`

For regulated pharma validation, the public controlled protocol is:

- [IQ/OQ/PQ Validation Protocol](../validation/iq-oq-pq-protocol.md)

That protocol currently states:

- `Document ID: NS-VAL-001`
- `Version: 2.0.0`
- Appendix B traceability matrix linking requirements to test cases
- public qualification inventory visible in the protocol text:
  - `21` IQ IDs
  - `79` OQ IDs
  - `25` PQ IDs
- the v2.0.0 change log explicitly records an expanded OQ traceability matrix
  with `85 test cases`

## Quality Layers

The quality model is layered. Each layer answers a different question.

| Layer | Question | Typical Output | Release Impact |
| --- | --- | --- | --- |
| Unit and integration tests | Does the implementation behave correctly at function/module/API level? | `cargo test`, `pytest`, CLI/tool smoke tests | Blocking |
| Deterministic parity and contract tests | Does NextStat match a trusted reference or a frozen contract where one exists? | pyhf parity, ROOT/TREx parity, golden fixtures, schema checks | Blocking for promoted surfaces |
| Stable-surface governance | Is a public surface explicitly covered by support docs, acceptance rules, and release wiring? | support matrix, acceptance doc, release surface matrix, HEP surface matrix | Blocking |
| Benchmark and evidence capture | Is the promoted surface operating inside its declared envelope? | benchmark snapshots, evidence bundles, comparison reports | Blocking or advisory depending on policy |
| Prerelease simulation | Will the release pipeline actually stage, validate, and publish the required assets? | prerelease summary, release manifest, full-fidelity simulation report | Blocking |
| Publish workflows | Are all release assets built, staged, and published correctly across platforms? | GitHub Release, wheels, sdists, crates, validation artifacts | Blocking |

This layered model is deliberate:

- tests catch local behavioral bugs
- parity and contracts keep numerical meaning stable
- governance prevents undocumented public drift
- release simulation catches workflow-only failures
- publish workflows prove that the release surface is actually shippable

## Test Layers

### 1. Rust and Python correctness

The first layer is conventional software testing:

- `cargo test --workspace`
- targeted `pytest` suites for Python bindings, CLI, server, tools, and
  release/process regression checks
- schema and example validation for JSON-facing surfaces

This layer catches implementation bugs but does not by itself justify
numerical trust claims.

### 2. Deterministic parity and frozen contracts

Where a trusted external reference exists, NextStat uses it.

Examples:

- HistFactory and simplified-likelihood parity against pyhf
- ROOT/TRExFitter parity for explicitly bounded comparison paths
- domain-specific frozen examples and golden fixtures for report and bundle
  surfaces

Where an external reference does not exist, the contract is made explicit
through:

- support matrices
- acceptance documents
- schema-validated examples
- deterministic committed evidence

This is the layer that turns "implementation" into "governed public
behavior".

### Deterministic Reference Guarantees

For stable numerical surfaces, NextStat explicitly distinguishes between:

- a deterministic reference path
- an optimized production path

The deterministic path is the trust anchor. Public docs already define this
for major surfaces:

- HistFactory / pyhf parity uses deterministic CPU evaluation with fixed
  summation order, fixed seeds, and controlled threading
- the pyhf parity contract uses pyhf's NumPy backend (`f64`, deterministic) as
  the canonical oracle
- the HistFactory stable subset promises deterministic CPU parity and
  bit-reproducible results on the same input across releases
- the pharma qualification protocol includes bit-for-bit reproducibility tests
  such as `OQ-SAEM-007` and `PQ-REPR-001`

The practical rule is:

- optimize freely on the fast path
- do not break the deterministic reference contract

## Canonical Commands and Outputs

The public quality system is intentionally executable. The core commands are:

| Command | Layer | Primary Outputs |
| --- | --- | --- |
| `cargo test --workspace` | correctness | Rust unit/integration/doctest results |
| `pytest ...` | correctness/contracts | Python, CLI, server, schema, and release-process checks |
| `python3 -m scripts.release_surface_matrix --check` | governance | validates the release inventory |
| `python3 -m scripts.hep_surface_matrix --check` | governance | validates the HEP maturity inventory |
| `python3 -m scripts.hep_validation_bundle --check` | governance/evidence | validates the canonical HEP bundle contract |
| `make validation-pack` | regulated evidence | deterministic validation report and qualification-facing artifacts |
| `make apex2-pre-release-gate` | prerelease | governance/perf summary, release reports, local dry-run |
| `python3 -m scripts.release_full_fidelity_simulation` | release simulation | local workflow-faithful staging report |

The canonical prerelease outputs are:

| Artifact | Meaning |
| --- | --- |
| `tmp/apex2_pre_release_gate_summary.json` | machine-readable overall prerelease outcome |
| `tmp/apex2_pre_release_gate_summary.md` | maintainer-readable summary |
| `tmp/release_surface_matrix_report.json` | required/advisory/manual surface coverage |
| `tmp/release_manifest.json` | candidate publish contract |
| `tmp/release_full_fidelity_simulation_report.json` | local release staging result |
| `tmp/release_candidate_bundle/` | candidate asset bundle for review |
| `tmp/hep_validation_bundle.json` | canonical HEP quality state |
| `tmp/hep_validation_bundle.md` | human-readable HEP bundle summary |
| `artifacts_jsononly/pharma_validation.json` | canonical pharma validation evidence reused by release-grade packaging paths |

The public validation/reporting layer also includes regulated artifacts such as
the validation pack and compliance mappings:

- [Validation Report Reference](validation-report.md)
- [IQ/OQ/PQ Validation Protocol](../validation/iq-oq-pq-protocol.md)
- [21 CFR Part 11 Compliance Documentation](../validation/21cfr-part11-compliance.md)

## Stable-Surface Governance

Stable-surface governance exists to prevent "silent maturity drift":
a surface must not become publicly important without explicit contracts.

For a promoted stable surface, the following are expected:

- a support matrix describing what is covered
- an acceptance or fidelity contract describing success criteria
- a runtime gate or release gate
- machine-readable wiring in the release surface inventory
- release artifacts or evidence where applicable

Examples of surface-specific public references:

- [HEP Stable Surface](hep-stable-surface.md)
- [Simplified Likelihood Stable-Surface Acceptance](../benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md)
- [HistFactory Support Matrix](../benchmarks/histfactory-support-matrix-2026-03-17.md)
- [HEPData Import Runtime Gate](../benchmarks/hepdata-import-runtime-gate.md)

## Machine-Readable Sources of Truth

NextStat uses machine-readable inventories to prevent documentation drift.

### Release surface matrix

The canonical release inventory is:

- `scripts/release_surface_matrix_v1.json`

It records:

- which surfaces are required for every release
- which CI jobs cover those surfaces
- which local `make` targets reproduce the checks
- which docs are canonical for each surface

Validation command:

```bash
python3 -m scripts.release_surface_matrix --check
```

### HEP surface matrix

The canonical HEP inventory is:

- `hep_surface_matrix_v1.json`

It records every public HEP surface with:

- `name`
- `layer`
- `maturity_class`
- `owner_slice`
- `support_matrix_ref`

Validation commands:

```bash
python3 -m scripts.hep_surface_matrix --check
python3 -m scripts.hep_validation_bundle --check
```

The HEP validation bundle is the single canonical artifact summarizing the
current HEP quality state.

## Required Release Surfaces

The current required-for-release surfaces are defined in
`scripts/release_surface_matrix_v1.json`. At the time of writing they are:

| Surface | Local Gate | CI Job | Canonical Public Reference |
| --- | --- | --- | --- |
| `gvm_stable_first` | `gvm-stable-first-gate` | `gvm-stable-first` | `docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md` |
| `simplified_likelihood_stable_surface` | `simplified-likelihood-stable-surface-gate` | `simplified-likelihood-stable-surface` | `docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md` |
| `simplified_likelihood_exporter_surface` | `simplified-likelihood-exporter-surface-gate` | `simplified-likelihood-exporter-surface` | `docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09.md` |
| `m15_reporting_stable_surface` | `m15-reporting-stable-surface-gate` | `m15-reporting-stable-surface` | `docs/references/m15-reporting.md` |
| `hepdata_import_stable_surface` | `hepdata-import-stable-surface-gate` | `hepdata-import-stable-surface` | `docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md` |
| `histfactory_stable_surface` | `histfactory-stable-surface-gate` | `histfactory-stable-surface` | `docs/benchmarks/histfactory-support-matrix-2026-03-17.md` |

If a public stable surface is not represented in the relevant machine-readable
inventory, it is not fully governed.

## Benchmark and Evidence Layers

NextStat deliberately separates benchmarks from correctness:

- correctness answers "is this contract still true?"
- benchmarks answer "is this still inside the expected operating envelope?"

The repository keeps both micro-benchmarks and end-to-end benchmark surfaces:

- Rust micro-benchmarks via Criterion
- Apex2 end-to-end benchmark and validation runners
- stable-surface snapshots and evidence bundles for promoted domains

Canonical public entry point:

- [Benchmarks](../benchmarks.md)

Important policy distinction:

- committed git artifacts are reserved for contracts and minimal accepted
  evidence
- CI artifacts hold transient validation bundles and rerun outputs
- GitHub Release assets hold publish-grade release evidence

See:

- [Benchmark Artifact Policy](../releases/benchmark-artifact-policy.md)
- [Release Runbook](../releases/release-runbook.md)

## Artifact Placement Rules

Artifact placement is part of the release contract. The same output should not
float ambiguously between git, CI, and release assets.

| Location | What belongs there | What does not belong there |
| --- | --- | --- |
| Git-tracked docs/specs/examples | contracts, schemas, support matrices, minimal accepted evidence | raw rerun histories, transient CI outputs, bulk benchmark dumps |
| CI artifacts | transient rerun bundles, validation outputs, workflow-scoped reports | canonical long-term published evidence |
| GitHub Release assets | publish-grade evidence, wheels, sdists, released validation bundles | internal scratch outputs and local-only reruns |
| Local `tmp/` | prerelease working state and review artifacts | committed source-of-truth contracts |

This placement discipline is what keeps the repository auditable instead of
turning it into a cache of opaque generated output.

## CI and Release Topology

The automation layer is intentionally split by purpose rather than hidden
behind a single monolithic CI status.

### Pull request and push validation

Representative always-on or path-sensitive workflows include:

- `rust-tests.yml`
- `python-tests.yml`
- `pyhf-parity.yml`
- `unbinned-toy-parity.yml`
- stable-surface workflows such as:
  - `gvm-stable-first.yml`
  - `simplified-likelihood-stable-surface.yml`
  - `simplified-likelihood-exporter-surface.yml`
  - `m15-reporting-stable-surface.yml`
- review and supply-chain workflows such as:
  - `dependency-audit.yml`
  - `codeql.yml`
  - `secret-scan.yml`

### Prerelease and publish workflows

The release path is distinct and explicit:

- `prepare-release.yml` for manual candidate preparation
- `release-candidate.yml` for build, wheel, gate, and manifest generation
- `release.yml` for GitHub Release plus crates.io/PyPI publication

### Scheduled and nightly workflows

Longer-running or environment-specific checks are separated out:

- `apex2-nightly-slow.yml`
- `slow-regressions.yml`
- `bench.yml`
- `coverage.yml`
- `trex-baseline-refresh.yml`
- `ns-root-external-bench.yml`

The governance narrative matters because different checks block at different
times:

- PR/push checks catch implementation and surface drift early
- prerelease gates block shipping
- nightly and scheduled jobs harden the baseline without overloading PR paths

## Governance vs Performance

The prerelease process distinguishes between release correctness and
machine-dependent performance drift.

The local prerelease gate is:

```bash
make apex2-pre-release-gate
```

It emits:

- `tmp/apex2_pre_release_gate_summary.json`
- `tmp/apex2_pre_release_gate_summary.md`
- `tmp/release_surface_matrix_report.json`
- `tmp/release_manifest.json`
- `tmp/release_full_fidelity_simulation_report.json`

The gate uses separate outcomes for:

- governance / correctness failures
- performance / baseline failures
- infrastructure / baseline-state failures

Interpretation:

- governance failure blocks the release
- performance on a non-canonical host may be advisory
- performance on a canonical enforced runner must be reviewed before changing
  expectations
- infrastructure failure means the prerelease environment itself is not in a
  valid state

This split prevents a machine-specific slowdown from being confused with a
release-integrity failure.

### Decision Semantics

The intended operator behavior is:

| Outcome | Meaning | Action |
| --- | --- | --- |
| `success` | correctness and release governance are green | proceed |
| `governance` | a stable-surface, manifest, matrix, simulation, or contract failure exists | block release and fix |
| `performance` | performance contract failed on an enforced/canonical runner | review baseline, hardware, or contract before release |
| `infrastructure` | prerelease environment is not valid enough to interpret results | fix environment first |
| `advisory` perf on non-canonical host | informative drift outside canonical hardware | do not treat as release corruption by itself |

This is the practical difference between "the release is broken" and "the
release is correct, but the local machine is not the canonical perf host."

## Human Review and Change Control

Automation is not the only release control.

The project also relies on explicit human review:

- maintainers review and comment on changes
- approval is required before merge to `main`
- DCO sign-off is required on all commits
- release notes and prerelease outputs are reviewed before tag push

Public references:

- [CONTRIBUTING.md](../../CONTRIBUTING.md)
- [.github/pull_request_template.md](../../.github/pull_request_template.md)

This matters because numerical software needs both:

- mechanical enforcement of contracts
- human judgment around scope, interpretation, and release decisions

## Full-Fidelity Local Release Simulation

NextStat does not rely only on GitHub Actions to discover release-pipeline
problems.

The local prerelease gate includes a workflow-faithful release simulation:

- it reconstructs the artifact layout expected from the candidate workflow
- stages release assets using the same staging logic as the publish path
- verifies that release assets can be validated and attached before tag push

Canonical implementation:

- `scripts/release_full_fidelity_simulation.py`

This is how release-only bugs such as artifact layout mismatches, missing
manifest entries, or staging collisions are caught before a real tag is cut.

## Stable Promotion Model

A surface is not promoted to stable because it exists or because it has tests.
Promotion requires explicit closure of the governance loop.

Promotion usually lands these pieces together:

| Piece | Purpose |
| --- | --- |
| Support matrix | Defines covered scope |
| Acceptance/fidelity doc | Defines success and tolerances |
| Runtime or release gate | Enforces the surface in CI and prerelease |
| Machine-readable inventory entry | Prevents silent drift |
| Evidence bundle or benchmark snapshot | Captures the current accepted state |
| Public docs / quickstart | Makes the stable boundary legible to users |

This is why the repository contains many examples, schema files, and evidence
artifacts: they are part of the product contract, not incidental clutter.

### Promotion Lifecycle

The intended stable promotion lifecycle is:

1. a surface exists as implementation
2. its public boundary is described with support and acceptance docs
3. the surface is wired into machine-readable governance
4. runtime or release gates are added
5. evidence is captured and reviewed
6. the surface is promoted to stable
7. every future release revalidates that stable claim

In other words, a stable surface is not a label added at the end of
development. It is the result of closing the whole governance loop.

### What "Stable" Means Here

In this system, `stable` means:

- the public boundary is explicit
- the expected behavior is documented
- the governing checks are machine-enforced
- the release path knows about the surface
- the accepted evidence state can be reproduced or revalidated

It does **not** mean:

- infinite backward compatibility for every internal detail
- zero future optimization or implementation change
- zero performance movement across non-canonical machines

## Maintainer Workflow

At a high level, maintainers should think in this order:

1. Update versions and release notes.
2. Run the canonical prerelease gate.
3. Review governance summaries and required surface reports.
4. Review performance reports in the correct policy context.
5. Confirm that required stable-surface gates are green.
6. Use tag push as the canonical publish path.

The release process should therefore be read as:

1. prove the inventories are still valid
2. prove the required surfaces are still governed
3. prove the candidate assets still stage correctly
4. prove the publish path still has the required payload
5. publish only after those proofs are green

Detailed procedure:

- [Release Runbook](../releases/release-runbook.md)
- [CONTRIBUTING.md](../../CONTRIBUTING.md)

## Practical Reading Guide

If you are trying to understand whether a surface is trustworthy, read in this
order:

1. the public support matrix
2. the acceptance or parity contract
3. the runtime or release gate doc
4. the benchmark snapshot, validation bundle, or qualification protocol
5. the release runbook for the global shipping path

If one of those layers is missing, the surface is not fully described.

## Related Documents

- [Release Runbook](../releases/release-runbook.md)
- [Benchmarks](../benchmarks.md)
- [HEP Stable Surface](hep-stable-surface.md)
- [Validation Report Reference](validation-report.md)
- [White Paper](../WHITEPAPER.md)
- [Benchmark Artifact Policy](../releases/benchmark-artifact-policy.md)
- [IQ/OQ/PQ Validation Protocol](../validation/iq-oq-pq-protocol.md)
- [21 CFR Part 11 Compliance Documentation](../validation/21cfr-part11-compliance.md)
