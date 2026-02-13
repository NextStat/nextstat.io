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
- [ ] Follow the canonical runbook: `docs/tutorials/release-gates.md`
- [ ] Optional (cluster): run ROOT/TRExFitter parity and archive artifacts (see `docs/tutorials/root-trexfitter-parity.md`)

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
