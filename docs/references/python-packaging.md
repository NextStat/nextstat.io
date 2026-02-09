---
title: "Python Packaging (Wheels, Extras, Reproducibility)"
status: stable
---

# Python Packaging (Wheels, Extras, Reproducibility)

## Install

Base install (native extension):

```bash
pip install nextstat
```

Optional extras:

```bash
# Validation + parity tooling (pyhf + ROOT XML/IO helpers)
pip install "nextstat[validation]"

# Arrow/Parquet authoring from Python (PyArrow)
pip install "nextstat[io]"

# Plotting helpers (matplotlib)
pip install "nextstat[viz]"

# Bayesian helpers (emcee/arviz)
pip install "nextstat[bayes]"

# Pure-Python HTTP client for nextstat-server (nextstat.remote)
pip install "nextstat[remote]"

# Convenience set for demo/agent scripts (ROOT ingest + schema validation + remote client)
pip install "nextstat[agent]"
```

## Reproducible Wheel Builds (Notes)

NextStat uses `maturin` and a pinned Rust toolchain:

- Rust toolchain pin: `rust-toolchain.toml` (repo root)
- Python build backend: `bindings/ns-py/pyproject.toml` (`maturin>=1.11,<2.0`)

Recommended local build:

```bash
cd bindings/ns-py
python -m pip install --upgrade pip
pip install "maturin>=1.11,<2.0"
maturin build --release
```

Reproducibility guidelines:

- Use the pinned Rust toolchain (`rust-toolchain.toml`).
- Build with a locked dependency graph (commit `Cargo.lock` and keep it unchanged).
- Prefer CI wheels for distribution (GitHub Actions matrix in `.github/workflows/release.yml`).

## Compatibility

- Python: `>=3.11` (see `bindings/ns-py/pyproject.toml`)
- Platforms: macOS + Linux wheels are built in CI; other platforms may require source builds.
