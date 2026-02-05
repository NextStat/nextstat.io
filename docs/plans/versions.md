# NextStat Version Baseline (validated)

Validated: **2026-02-05**

Goal: keep plans executable on current toolchain and dependency versions. If you update versions, update them in Phase 0 bootstrap docs and keep the master plan in sync.

## Languages / Toolchain

- Rust toolchain: `1.93.0` (stable)
- Rust edition: `2024`
- Python: `>=3.11` (minimum). Recommended: test against the latest stable Python as well.

## Rust workspace dependencies (minimums)

Source of truth: `Cargo.toml` -> `[workspace.dependencies]` (plans must stay in sync with the repo)

- `ndarray = "0.17"`
- `num-traits = "0.2"`
- `approx = "0.5"`
- `statrs = "0.18"`
- `rayon = "1.11"`
- `nalgebra = "0.34"`
- `argmin = "0.11"`
- `argmin-math = { version = "0.5", features = ["ndarray_latest"] }`
- `serde = "1"`
- `serde_json = "1"`
- `serde_yaml_ng = "0.10"` (prefer this over `serde_yaml` due to deprecation status)
- `tokio = "1"`
- `clap = "4.5"`
- `pyo3 = "0.28"`
- `numpy = "0.27"`
- `proptest = "1.10"`
- `criterion = "0.8"`
- `tracing = "0.1"`
- `tracing-subscriber = "0.3"`
- `wide = "1.1"` (SIMD)
- `rand = "0.9"` (Phase 3 NUTS/HMC)
- `rand_distr = "0.5"` (Phase 3 NUTS/HMC)

GPU-related (Phase 2C):

- `metal = "0.33"`
- `cudarc = "0.19"` (see feature strategy in `docs/plans/phase-2c-gpu-backends.md`)

## Python tooling / validation deps (minimums)

Source of truth: `bindings/ns-py/pyproject.toml` (plans must stay in sync with the repo)

- `numpy>=2.0`
- `pytest>=9.0`
- `pytest-cov>=7.0`
- `ruff>=0.15`
- `mypy>=1.19`
- `maturin>=1.11,<2.0`
- `pyhf>=0.7.6` (validation extra)

Bayesian (Phase 3, optional extras):

- `arviz>=0.23.4`
- `emcee>=3.1.6`

## CI / Automation pins

Source of truth: `.github/workflows/*.yml`

- `actions/checkout@v6`
- `actions/setup-python@v6`
- `dtolnay/rust-toolchain@v1`
- `Swatinem/rust-cache@v2`
- `github/codeql-action/*@v4`
- `gitleaks/gitleaks-action@v2`
- `taiki-e/install-action@v2`
- `actions/upload-artifact@v6`
- `actions/download-artifact@v7`
- `PyO3/maturin-action@v1`
- `softprops/action-gh-release@v2`

## Pre-commit hook pins

Source of truth: `.pre-commit-config.yaml`

- `astral-sh/ruff-pre-commit` -> `rev: v0.15.0`
- `pre-commit/pre-commit-hooks` -> `rev: v6.0.0`
- `compilerla/conventional-pre-commit` -> `rev: v4.3.0`

## How to re-validate

- Rust crates: Dependabot + periodic checks against crates.io
- Python deps: Dependabot (pip) + `pip-audit`
- Pre-commit: `pre-commit autoupdate` (pin `rev:` in PRs)
- Quick snapshot (crates/PyPI/actions): `python3 scripts/versions_audit.py`
