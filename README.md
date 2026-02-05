# NextStat

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.93%2B-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)

NextStat is a high-performance statistical fitting toolkit for High Energy Physics (HEP), implemented in Rust with Python bindings.

## What You Get

- pyhf JSON compatibility (HistFactory-style workspaces)
- Negative log-likelihood (Poisson + constraints), including Barlow-Beeston auxiliary terms
- Maximum Likelihood Estimation (L-BFGS-B) with uncertainties via Hessian-based covariance
- SIMD kernels and Rayon parallelism where it matters
- Rust library, Python package (PyO3/maturin), and a CLI

## Quickstart

### Install (Rust)

```bash
cargo add ns-core ns-inference ns-compute
```

### Install (Python)

```bash
pip install nextstat
```

### Build From Source

```bash
git clone https://github.com/nextstat/nextstat.git
cd nextstat

# Rust workspace
cargo build --release

# Python bindings (editable dev install)
cd bindings/ns-py
maturin develop --release
```

## Usage

### Rust API

```rust
use ns_inference::mle::MaximumLikelihoodEstimator;
use ns_translate::pyhf::{HistFactoryModel, Workspace};

let json = std::fs::read_to_string("workspace.json")?;
let workspace: Workspace = serde_json::from_str(&json)?;
let model = HistFactoryModel::from_workspace(&workspace)?;

let mle = MaximumLikelihoodEstimator::new();
let result = mle.fit(&model)?;

println!("Best-fit params: {:?}", result.parameters);
println!("NLL at minimum: {}", result.nll);
```

### Python API

```python
import json

import nextstat

workspace = json.loads(open("workspace.json").read())
model = nextstat.from_pyhf(json.dumps(workspace))
result = nextstat.fit(model)

poi_idx = model.poi_index()
print("POI index:", poi_idx)
print("Best-fit POI:", result.bestfit[poi_idx])
print("Uncertainty:", result.uncertainties[poi_idx])
```

### CLI

```bash
nextstat fit --input workspace.json
nextstat version
```

## Architecture

NextStat follows a "clean architecture" style: inference depends on stable abstractions, not on specific execution backends.

```
┌─────────────────────────────────────────────────────────────────┐
│                        HIGH-LEVEL LOGIC                          │
│  ns-inference (MLE, Profile Likelihood, Hypothesis Tests, ...)   │
│  - depends on core types and model interfaces                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ depends on abstractions
┌─────────────────────────┴───────────────────────────────────────┐
│                      ns-core (interfaces)                        │
│  - error types, FitResult, traits                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │ implemented by
┌─────────────────────────┴───────────────────────────────────────┐
│                    LOW-LEVEL IMPLEMENTATIONS                     │
│  ns-translate (pyhf model + NLL/grad)  ns-compute (SIMD kernels)  │
│  ns-ad (dual/tape AD)                  optional GPU backends      │
└─────────────────────────────────────────────────────────────────┘
```

## Project Layout

```
nextstat/
├── crates/
│   ├── ns-core/
│   ├── ns-compute/
│   ├── ns-ad/
│   ├── ns-inference/
│   ├── ns-translate/
│   ├── ns-viz/
│   └── ns-cli/
├── bindings/
│   └── ns-py/
├── docs/
│   ├── legal/
│   ├── plans/
│   └── references/
└── tests/
```

## Development

### Requirements

- Rust 1.93+ (edition 2024)
- Python 3.11+ (for bindings)
- maturin (for Python bindings)

### Build and Test

```bash
# Build
cargo build --workspace

# Tests (including feature-gated backends)
cargo test --workspace --all-features

# Opt-in slow Rust tests (toys, etc.)
cargo test -p ns-inference -- --ignored

# Format and lint
cargo fmt --check
cargo clippy --workspace -- -D warnings
```

### Python Tests

Local test runs should use the repo venv (it pins a Python version compatible with the built extension).

```bash
# Run fast Python tests (parity + API contracts)
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python -m pytest -q -m "not slow" tests/python

# Run slow toy regression tests (opt-in)
PYTHONPATH=bindings/ns-py/python NS_RUN_SLOW=1 NS_TOYS=200 NS_SEED=0 ./.venv/bin/python -m pytest -q -m slow tests/python
```

### Benchmarks

```bash
# Compile and run all benches
cargo bench --workspace

# Common entry points
cargo bench -p ns-translate --bench model_benchmark
cargo bench -p ns-translate --bench nll_benchmark
cargo bench -p ns-compute --bench simd_benchmark
cargo bench -p ns-inference --bench mle_benchmark
cargo bench -p ns-inference --bench hypotest_benchmark
cargo bench -p ns-ad --bench ad_benchmark
cargo bench -p ns-core --bench core_benchmark
```

## Documentation

- White paper: `docs/WHITEPAPER.md`
- Internal plans/design notes are maintained outside this public repository.

## Contributing

See `CONTRIBUTING.md`. All commits must include DCO sign-off (`git commit -s`).

## License

NextStat uses a dual-licensing model:

- Open Source: `LICENSE` (AGPL-3.0-or-later)
- Commercial: `LICENSE-COMMERCIAL`

## Contact

- Website: https://nextstat.io
- GitHub: https://github.com/nextstat/nextstat
