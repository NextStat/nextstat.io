---
title: "Python Packaging (Wheels, Extras, Reproducibility)"
status: stable
---

# Python Packaging (Wheels, Extras, Reproducibility)

## Install

Base install (native extension + CLI binary):

```bash
pip install nextstat              # installs library + CLI
nextstat version                  # CLI works immediately
python -m nextstat version        # also works
```

The `nextstat` package automatically pulls in `nextstat-cli`, which places the
`nextstat` binary in your virtualenv's `bin/` (or `Scripts\` on Windows).
To install only the CLI without the Python library:

```bash
pip install nextstat-cli
```

Optional extras:

```bash
# Bayesian diagnostics (ArviZ, emcee, xarray)
pip install "nextstat[bayes]"

# Plotting helpers (matplotlib)
pip install "nextstat[viz]"

# Arrow/Parquet authoring from Python (PyArrow)
pip install "nextstat[io]"

# PyTorch differentiable layer and GPU tensor interop
pip install "nextstat[torch]"

# LangChain tool integration for AI agents
pip install "nextstat[langchain]"

# Pure-Python HTTP client for nextstat-server (nextstat.remote)
pip install "nextstat[remote]"

# Everything above combined
pip install "nextstat[all]"
```

Internal/development extras (not included in `all`):

```bash
# Validation + parity tooling (pyhf + ROOT XML/IO helpers)
pip install "nextstat[validation]"

# Convenience set for demo/agent scripts (ROOT ingest + schema validation + remote client)
pip install "nextstat[agent]"
```

## CLI Package (`nextstat-cli`)

The CLI is distributed as a separate PyPI package `nextstat-cli`.  It uses
`maturin` with `bindings = "bin"`, producing `py3-none-{platform}` wheels
(one per platform, not per Python version).

- Source: `bindings/ns-cli-py/pyproject.toml`
- Rust crate: `crates/ns-cli/`
- Binary name: `nextstat`

The main `nextstat` package pins `nextstat-cli=={version}` as a dependency,
so version upgrades stay in lockstep.

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

## HPC Deployment

For deploying on HTCondor, Slurm, or other batch clusters, see `docs/guides/htcondor-hpc.md`.
