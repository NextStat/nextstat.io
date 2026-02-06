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

### Bayesian (NUTS) + ArviZ

Install optional deps:

```bash
pip install "nextstat[bayes]"
```

Run sampling and get an ArviZ `InferenceData`:

```python
import json
from pathlib import Path

import nextstat

workspace = json.loads(Path("workspace.json").read_text())
model = nextstat.from_pyhf(json.dumps(workspace))

idata = nextstat.bayes.sample(
    model,
    n_chains=2,
    n_warmup=500,
    n_samples=1000,
    seed=42,
    target_accept=0.8,
)

print(idata)
```

### Viz (CLs Brazil bands, profile scans)

Install optional deps:

```bash
pip install "nextstat[viz]"
```

Compute artifacts and plot (matplotlib):

```python
import json
import numpy as np
from pathlib import Path

import nextstat

workspace = json.loads(Path("workspace.json").read_text())
model = nextstat.from_pyhf(json.dumps(workspace))

scan = np.linspace(0.0, 5.0, 101)
cls_art = nextstat.viz.cls_curve(model, scan, alpha=0.05)
nextstat.viz.plot_cls_curve(cls_art, title="CLs Brazil band")

mu = [0.0, 0.5, 1.0, 2.0]
prof_art = nextstat.viz.profile_curve(model, mu)
nextstat.viz.plot_profile_curve(prof_art, title="Profile likelihood scan")
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

Details (quick mode, baselines, CI workflows): `docs/benchmarks.md`.

## Documentation

- White paper (Markdown): `docs/WHITEPAPER.md`
- White paper (PDF): built by `python3 scripts/build_whitepaper.py` and attached to GitHub Releases on tags (`v*`)
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
