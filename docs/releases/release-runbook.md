# Release Runbook

This is the canonical maintainer runbook for shipping a NextStat GitHub release with:

- Rust crates
- Python wheels and sdists
- CLI wheels and binaries
- validation-pack artifacts
- stable-surface benchmark evidence

Canonical prepare workflow:
- [.github/workflows/prepare-release.yml](.github/workflows/prepare-release.yml)

Canonical candidate workflow:
- [.github/workflows/release-candidate.yml](.github/workflows/release-candidate.yml)

Canonical publish workflow:
- [.github/workflows/release.yml](.github/workflows/release.yml)

Canonical local gate:
- [scripts/apex2/pre_release_gate.sh](scripts/apex2/pre_release_gate.sh)

Canonical local release dry-run:
- [scripts/release_full_fidelity_simulation.py](scripts/release_full_fidelity_simulation.py)

Canonical surface matrix:
- [scripts/release_surface_matrix_v1.json](scripts/release_surface_matrix_v1.json)

Canonical release manifest schema:
- [release_manifest_v1.schema.json](docs/schemas/releases/release_manifest_v1.schema.json)

Benchmark artifact policy:
- [benchmark-artifact-policy.md](docs/releases/benchmark-artifact-policy.md)

Pharma release evidence policy:
- [pharma-release-evidence-policy.md](docs/releases/pharma-release-evidence-policy.md)

## Maintainer path

1. Ensure versions are aligned in:
   - [Cargo.toml](Cargo.toml)
   - [bindings/ns-py/pyproject.toml](bindings/ns-py/pyproject.toml)
   - [bindings/ns-cli-py/pyproject.toml](bindings/ns-cli-py/pyproject.toml)
2. Update [CHANGELOG.md](CHANGELOG.md).
3. Run the canonical pre-release gate:
   ```bash
   make apex2-pre-release-gate
   ```
   Notes:
   - local/dev runs use `APEX2_PERF_POLICY=auto` by default
   - `auto` enforces performance only when `APEX2_CANONICAL_PERF_RUNNER=1`
   - otherwise perf drift is reported as an advisory while governance remains hard-gated
4. Review:
   - `tmp/apex2_pre_release_gate_summary.json`
   - `tmp/apex2_pre_release_gate_summary.md`
   - `tmp/baseline_compare_report.json`
   - `tmp/trex_analysis_spec_compare_report.json` when applicable
   - `tmp/root_suite_compare_report.json` when applicable
   - `tmp/release_surface_matrix_report.json`
   - `tmp/release_surface_matrix_report.md`
   - `tmp/sota_claim_matrix_report.json`
   - `tmp/sota_claim_matrix_report.md`
   - `tmp/public_sota_bundle.json`
   - `tmp/public_sota_bundle.md`
   - `tmp/v1_sota_policy_report.json`
   - `tmp/v1_sota_policy_report.md`
   - `tmp/release_manifest.json`
   - `tmp/release_manifest.md`
   - `tmp/release_full_fidelity_simulation_report.json`
   - `tmp/release_full_fidelity_simulation_report.md`
   - `tmp/release_candidate_bundle/`
   - canonical Linux pharma evidence and any downstream reuse of `pharma_validation.json`
5. Confirm that every required stable-surface gate listed in the release surface report is green.
6. Interpret the pre-release exit code before taking action:
   - `20`: governance / correctness failure; release is blocked
   - `21`: performance / baseline failure on an enforced/canonical perf runner; review before changing contracts
   - `22`: infrastructure / baseline-state failure; fix the prerelease environment first
7. Use tag push as the canonical publish path:
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`

## Release surface matrix

The release surface matrix is the machine-readable source of truth for:

- which stable surfaces are required for every release
- which workflow jobs correspond to each surface
- which `make` targets maintainers should run locally
- which docs are canonical for each stable surface

The pre-release gate emits:
- `tmp/apex2_pre_release_gate_summary.json`
- `tmp/apex2_pre_release_gate_summary.md`
- `tmp/release_surface_matrix_report.json`
- `tmp/release_surface_matrix_report.md`
- `tmp/sota_claim_matrix_report.json`
- `tmp/sota_claim_matrix_report.md`
- `tmp/public_sota_bundle.json`
- `tmp/public_sota_bundle.md`
- `tmp/v1_sota_policy_report.json`
- `tmp/v1_sota_policy_report.md`
- `tmp/release_full_fidelity_simulation_report.json`
- `tmp/release_full_fidelity_simulation_report.md`

Interpretation:
- `required_release_surfaces`: every release must satisfy these gates
- `advisory_touched_surfaces`: surfaces inferred from changes since the latest reachable `v*` tag
- `optional_manual_surfaces`: parity/cluster surfaces that remain manual or environment-bound

## Current stable release surfaces

- `gvm_stable_first`
- `simplified_likelihood_stable_surface`
- `simplified_likelihood_exporter_surface`
- `m15_reporting_stable_surface`
- `hepdata_import_stable_surface`
- `histfactory_stable_surface`

Current optional manual surface:
- `root_trexfitter_parity`

## Workflow dispatch contract

[prepare-release.yml](.github/workflows/prepare-release.yml) is the canonical manual prepare-only path.

Required input:
- `release_tag` in `vX.Y.Z` form

Manual dispatch is for:
- validating versions
- building candidate artifacts
- emitting a machine-readable release manifest
- emitting a canonical release candidate bundle
- exercising the release matrix without publishing

Manual prerelease validation is also expected to use local build artifacts for the
Python surface; it must not depend on PyPI availability of `nextstat-cli`.

Manual dispatch does not publish:
- crates.io
- GitHub Release
- PyPI

Canonical production publish path remains tag push through [release.yml](.github/workflows/release.yml). Both paths share the reusable candidate workflow in [release-candidate.yml](.github/workflows/release-candidate.yml).

## Full-fidelity local release simulation

The local prerelease gate now runs a synthetic but workflow-faithful release simulation that:

- parses upload-artifact steps from [release-candidate.yml](.github/workflows/release-candidate.yml)
- reconstructs the downloaded artifact layout under a local `dist/`
- runs [release_stage_assets.py](scripts/release_stage_assets.py)
- verifies that GitHub Release asset staging succeeds before retag/push

This dry-run is intended to catch CI-only asset layout and release staging bugs before the real
tagged publish workflow runs.

## Artifact handling

Do not use the git repository as a dumping ground for raw benchmark histories or transient release outputs.

Use the benchmark artifact policy:
- committed git artifacts only for canonical contracts and minimal accepted evidence
- CI artifacts for transient validation bundles
- GitHub Release assets for published release evidence

Use the pharma release evidence policy:
- canonical Linux `pharma_validation.json` is release-grade evidence
- deterministic rerender paths may reuse that evidence
- cross-platform SAEM compatibility is an acceptance-envelope surface, not an exact snapshot surface

## Related references

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/references/validation-and-release-discipline.md](docs/references/validation-and-release-discipline.md)
- [docs/benchmarks.md](docs/benchmarks.md)
- [docs/references/m15-reporting.md](docs/references/m15-reporting.md)
