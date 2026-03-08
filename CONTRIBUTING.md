# Contributing to NextStat

Thanks for your interest in NextStat. We welcome all contributions, from typo fixes to new features.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [DCO Sign-off](#dco-sign-off)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

We aim to build an open and welcoming community. Please be respectful and constructive in all discussions.

## Getting Started

### Find Something to Work On

1. Browse GitHub issues labeled `good first issue` or `help wanted`
2. Read `docs/WHITEPAPER.md` for architectural and validation context
3. If you have a new idea, open an issue first to discuss scope and approach

### Environment Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/your-username/nextstat.io.git
cd nextstat.io

# 3. Add upstream remote
git remote add upstream https://github.com/NextStat/nextstat.io.git

# 4. Build and run tests
cargo build --workspace
cargo test --workspace

# Optional: include feature-gated backends (CUDA requires nvcc)
# cargo test --workspace --all-features

# 5. (Optional) install pre-commit hooks
# Planned in Phase 0
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming convention:

- `feature/` - new functionality
- `bugfix/` - bug fix
- `docs/` - documentation-only changes
- `refactor/` - refactoring without API changes

### 2. Follow TDD (Test-Driven Development)

Required for code changes:

1. Write a failing test
   ```bash
   cargo test -p ns-core test_name -- --nocapture
   # should FAIL
   ```
2. Implement the minimal fix
3. Run the test again
   ```bash
   cargo test -p ns-core test_name
   # should PASS
   ```
4. Refactor if needed
   ```bash
   cargo test --workspace
   ```
5. Commit with DCO sign-off
   ```bash
   git add .
   git commit -s -m "feat(ns-core): add new functionality"
   ```

## Coding Standards

### Rust

- Style: run `cargo fmt` before committing
- Linting: fix all `cargo clippy` warnings (CI treats warnings as errors)
- Documentation: public APIs must have doc comments
- Tests: add coverage for new functionality (aim for 80%+ for new modules)
- Error handling: prefer `Result<T, Error>`, avoid `panic!` in library code

### Python

- Style: PEP 8, format with `ruff format` (CI checks formatting)
- Type hints: required for public functions
- Keep public surface area stable and tested (API contracts + parity tests)

## Pull Request Process

### 1. Before Opening a PR

- [ ] Tests pass: `cargo test --workspace`
- [ ] No clippy warnings: `cargo clippy --workspace -- -D warnings`
- [ ] Code is formatted: `cargo fmt --check`
- [ ] All commits include DCO sign-off
- [ ] Docs updated if behavior changed
- [ ] Tests added for new behavior

### 1a. Additional Gate for M15 / Regulated Reporting Changes

If your PR changes the public M15 surface, at minimum:

- `crates/ns-cli/src/m15.rs`
- `docs/schemas/validation/m15_*`
- `docs/specs/m15_*`
- `validation-pack/render_validation_pack.sh`
- `.github/workflows/python-tests.yml`
- `.github/workflows/release.yml`
- `docs/references/m15-reporting.md`

Then before opening the PR, run:

```bash
pytest -q \
  tests/python/test_m15_artifact_schema_smoke.py \
  tests/python/test_python_tests_workflow_m15_smoke.py \
  tests/python/test_release_workflow_m15_smoke.py \
  tests/python/test_validation_pack_script_smoke.py

cargo test -p ns-cli \
  --test cli_m15_assessment_table \
  --test cli_m15_map \
  --test cli_m15_mar \
  --test cli_m15_bundle
```

And confirm all of the following:

- Schemas, examples, and `docs/references/m15-reporting.md` stay in sync.
- Deterministic rerender remains intact for the M15 artifact chain.
- No hidden model execution was introduced into the M15 render/bundle path.
- CI/release workflow parity for M15 validation-pack assets remains intact.

### 1b. Additional Gate for Bayesian Design Stable Surface Changes

If your PR changes the public Bayesian design stable surface, at minimum:

- `crates/ns-inference/src/bayes_design.rs`
- `bindings/ns-py/src/lib.rs`
- `bindings/ns-py/python/nextstat/bayes_design.py`
- `validation-pack/render_validation_pack.sh`
- `docs/schemas/pharma/*design*`
- `docs/schemas/pharma/bayesian_*`
- `docs/specs/pharma/*design*`
- `docs/specs/pharma/bayesian_*`
- `docs/benchmarks/bayesian-design-*`
- `docs/references/bayesian-trial-design-artifacts.md`
- `docs/references/validation-report.md`
- `docs/specs/pharma/bayesian_design_report_acceptance_v0.md`
- `docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md`
- `docs/specs/pharma/bayesian_historical_control_borrowing_review_acceptance_v0.md`
- `docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_acceptance_v0.md`
- `docs/specs/pharma/bayesian_robust_mixture_prior_review_acceptance_v0.md`
- `docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md`

Then before opening the PR, run:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q \
  tests/python/test_bayes_design_module_api.py \
  tests/python/test_bayes_design_contract.py \
  tests/python/test_bayes_design_schema_smoke.py \
  tests/python/test_bayes_design_stable_surface_regression.py \
  tests/python/test_validation_pack_script_smoke.py \
  tests/python/test_validation_pack_execution_regression.py \
  tests/python/test_bayesian_design_report_bundle_performance_budget.py \
  tests/python/test_bayesian_design_report_bundle_benchmark_smoke.py \
  tests/python/test_bayes_design_checklists_smoke.py

python3 scripts/docs/terminology_lint.py --check
```

If the change also affects CLI schema publication, run:

```bash
cargo test -p ns-cli config_schema_can_emit_beta_binomial_design_schemas --test cli_config_schema
```

If the change affects the published design-report bundle runtime behavior or public performance wording, run the canonical packaging benchmark gate on `nextstat-bench`:

```bash
ssh nextstat-bench
cd /path/to/nextstat.io
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
  scripts/benchmarks/bench_bayesian_design_report_bundle.py \
  --deterministic \
  --out bench_results/bayesian_design_report_bundle/summary.json
```

And confirm all of the following:

- Schemas, examples, acceptance docs, and `docs/references/bayesian-trial-design-artifacts.md` stay in sync.
- Backward-compatible ingress remains intact for the published report, appendix, and bundle wrappers.
- Backward-compatible ingress remains intact for the published borrowing and robust-mixture extension operating-characteristics wrappers.
- Seeded determinism remains intact for the published borrowing and robust-mixture extension operating-characteristics wrappers.
- No hidden execution was introduced into frozen Bayesian report render/bundle paths.
- No hidden execution was introduced into frozen Bayesian prior conflict diagnostic paths.
- No hidden execution was introduced into frozen Bayesian historical-control borrowing review paths.
- No hidden execution was introduced into frozen Bayesian robust-mixture prior review paths.
- No hidden execution was introduced into frozen Bayesian regulatory appendix paths.
- No hidden execution was introduced into frozen Bayesian regulatory appendix render paths.
- No hidden execution was introduced into the frozen Bayesian validation-pack appendix path.
- `nextstat-bench` promotion evidence is linked when runtime-affecting packaging behavior changes.

### 2. Open the Pull Request

1. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a PR on GitHub (`base: main` <- `compare: your-branch`)
3. Fill out the PR template

### 3. Code Review

- Maintainers will review and comment
- Address requested changes (or explain tradeoffs)
- Pushing updates to your branch will update the PR automatically

### 4. Merge

After approval, a maintainer will merge your PR into `main`.

## Release Checklist (Maintainers)

- [ ] Ensure git working tree is clean (or set `APEX2_ALLOW_DIRTY=1` only if you understand the risk)
- [ ] Run the Apex2 pre-release gate: `make apex2-pre-release-gate`
- [ ] If the baseline is stale (expected perf change), re-record: `make apex2-baseline-record`
- [ ] Review `tmp/baseline_compare_report.json` for any slowdowns/flags
- [ ] Follow the canonical release runbook: `docs/releases/release-runbook.md`
- [ ] Review `tmp/release_surface_matrix_report.json` and `tmp/release_surface_matrix_report.md`
- [ ] Review `tmp/release_manifest.json` and `tmp/release_manifest.md`
- [ ] Review `tmp/release_candidate_bundle/`
- [ ] Optional (cluster): run ROOT/TRExFitter parity and archive artifacts (see `docs/tutorials/root-trexfitter-parity.md`)
- [ ] If the release includes M15 surface changes, run the M15 PR/release gates from `docs/references/m15-reporting.md` and verify M15 validation-pack assets are published (`m15_bundle_manifest.json`, schema, `.sha256`, `.sha256.bin`, `m15_snapshot_index.json`)
- [ ] If the release includes Bayesian design stable-surface changes, run the Bayesian PR/release gates from `docs/references/bayesian-trial-design-artifacts.md` and verify the current `nextstat-bench` packaging artifact is linked when runtime-affecting packaging behavior changed
- [ ] Follow the benchmark artifact policy: `docs/releases/benchmark-artifact-policy.md`

## DCO Sign-off

All commits must be signed off with DCO (Developer Certificate of Origin).

What it means: by signing off, you certify you have the right to contribute the code under the project's license.

See `DCO.md` for the full text.

Sign off automatically:

```bash
git commit -s -m "your commit message"
```

If you forgot:

```bash
# last commit
git commit --amend --signoff

# multiple commits
git rebase --signoff HEAD~3
git push --force-with-lease origin your-branch
```

## Testing

### Types of Tests

- Unit tests: small, isolated checks of functions/modules
- Integration tests: behavior across module boundaries (including CLI smoke tests)
- Doc tests: examples in Rust documentation

### Running Tests

```bash
# All Rust tests
cargo test --workspace

# Optional: include feature-gated backends (CUDA requires nvcc)
# cargo test --workspace --all-features

# A specific crate
cargo test -p ns-core

# A specific test
cargo test -p ns-core test_name

# With output
cargo test -p ns-core -- --nocapture

# Doctests only
cargo test --doc
```

Python tests (use the repo venv):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q -m "not slow" tests/python
```

## Documentation

Documentation types:

1. Code docs: required for public APIs
2. User docs: update `README.md` and relevant pages under `docs/`
3. Architecture/design docs: add or update docs under `docs/` (or create an RFC if needed)

## Questions

- Open a GitHub issue with label `question`
- Email: dev@nextstat.io
