# NextStat

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.93%2B-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)

NextStat is a high-performance statistical fitting toolkit for High Energy Physics (HEP), implemented in Rust with Python bindings.

## The God Run (toy-based CLs)

- **Model:** S+B HistFactory (synthetic), 50 channels × 4 bins, 201 parameters (mu + 200 nuisances)
- **Task:** CLs via toy-based q~_mu
- **Load:** 10,000 toys (b-only) + 10,000 toys (s+b)
- **Machine:** Apple M5 (arm64), macOS-26.2-arm64-arm-64bit-Mach-O
- **Versions:** nextstat 0.1.0, pyhf 0.7.6, Python 3.13.11
- **Recorded:** 2026-02-07 (UTC)
- **Commit:** 88d57856

| Tool | Wall time | Speedup |
|---|---:|---:|
| NextStat (Rayon) | 3.47 s | 1.0× |
| pyhf (multiprocessing, 10 procs) | 50m 11.7s | 868.0× |

Reproduce:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python scripts/god_run_benchmark.py --n-toys 10000
```

## What You Get

- pyhf JSON compatibility (HistFactory-style workspaces)
- Native HS3 (HEP Statistics Serialization Standard) v0.2 support — load ROOT 6.37+ HS3 JSON directly, auto-detected alongside pyhf
- Native ROOT TTree reader with mmap I/O, rayon-parallel basket decompression, and columnar extraction — no ROOT C++ dependency
- Ntuple-to-workspace pipeline: ROOT ntuples → histograms → HistFactory workspace (TRExFitter replacement)
- Expression engine for string-based selections and weights (`"njet >= 4 && pt > 25.0"`)
- Negative log-likelihood (Poisson + constraints), including Barlow-Beeston auxiliary terms
- Maximum Likelihood Estimation (L-BFGS-B) with uncertainties via (damped) Hessian-based covariance + diagonal fallback
- NUTS sampling surface (generic `Posterior` API) + optional ArviZ integration
- SIMD kernels, Rayon parallelism, Apple Accelerate (vDSP/vForce), and optional GPU acceleration (CUDA for NVIDIA, Metal for Apple Silicon)
- Rust library, Python package (PyO3/maturin), and a CLI
- Implemented packs: regression/GLM, hierarchical models, time series (Kalman/EM/forecast), econometrics/causal helpers, and PK/NLME baselines

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

### Try the Playground (WASM)

Run asymptotic CLs upper limits (Brazil bands) in the browser (no Python, no server) using a pyhf-style `workspace.json`.

From the repo root:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.108

make playground-build-wasm
make playground-serve
```

Open `http://localhost:8000/` and drag & drop a `workspace.json` (example: `playground/examples/simple_workspace.json`).

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

### Ntuple → Workspace (TRExFitter replacement)

```rust
use ns_translate::NtupleWorkspaceBuilder;

let ws = NtupleWorkspaceBuilder::new()
    .ntuple_path("ntuples/")
    .tree_name("events")
    .measurement("meas", "mu")
    .add_channel("SR", |ch| {
        ch.variable("mbb")
          .binning(&[0., 50., 100., 150., 200., 300.])
          .selection("njet >= 4 && pt > 25.0")
          .data_file("data.root")
          .add_sample("signal", |s| {
              s.file("ttH.root")
               .weight("weight_mc * weight_sf")
               .normfactor("mu")
          })
          .add_sample("background", |s| {
              s.file("ttbar.root")
               .weight("weight_mc * weight_sf")
               .normsys("bkg_norm", 0.9, 1.1)
               .weight_sys("jes", "weight_jes_up", "weight_jes_down")
               .tree_sys("jer", "jer_up.root", "jer_down.root")
               .staterror()
          })
    })
    .build()?;  // → Workspace (same type as pyhf JSON path)
```

No ROOT C++ dependency. ~8.5x faster than uproot+numpy on the full pipeline.

### Low-level TTree access

```rust
use ns_root::RootFile;

let file = RootFile::open("data.root")?;
let tree = file.get_tree("events")?;

// Columnar access
let pt: Vec<f64> = file.branch_data(&tree, "pt")?;
let eta: Vec<f64> = file.branch_data(&tree, "eta")?;

// Expression engine
let expr = ns_root::CompiledExpr::compile("pt > 25.0 && abs(eta) < 2.5")?;
```

### CLI

```bash
nextstat fit --input workspace.json
nextstat hypotest --input workspace.json --mu 1.0 --expected-set
nextstat hypotest-toys --input workspace.json --mu 1.0 --n-toys 10000 --seed 42 --threads 0
nextstat hypotest-toys --input workspace.json --mu 1.0 --n-toys 10000 --gpu cuda   # NVIDIA GPU (f64)
nextstat hypotest-toys --input workspace.json --mu 1.0 --n-toys 10000 --gpu metal  # Apple Silicon GPU (f32)
nextstat upper-limit --input workspace.json --expected --scan-start 0 --scan-stop 5 --scan-points 201
nextstat version
```

## Documentation

- Tutorial index: `docs/tutorials/README.md`
- Python API reference: `docs/references/python-api.md`
- Rust API reference: `docs/references/rust-api.md`
- CLI reference: `docs/references/cli.md`
- Playground (browser/WASM): `docs/references/playground.md` (and `playground/README.md`)

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
│  ns-translate (pyhf + ntuple → Workspace)  ns-compute (SIMD/CUDA/Metal) │
│  ns-ad (dual/tape AD)  ns-root (ROOT I/O, TTree, expressions)    │
└─────────────────────────────────────────────────────────────────┘
```

## Project Layout

```
nextstat/
├── crates/
│   ├── ns-core/         # Core types, traits, error handling
│   ├── ns-compute/      # SIMD kernels, Apple Accelerate, CUDA/Metal batch NLL+grad
│   ├── ns-ad/           # Automatic differentiation (dual/tape)
│   ├── ns-root/         # Native ROOT file reader (TH1, TTree, expressions, filler)
│   ├── ns-translate/    # Format translators (pyhf, HS3, HistFactory XML, ntuple builder)
│   ├── ns-inference/    # MLE, NUTS, CLs, GLM, time series, PK/NLME
│   ├── ns-viz/          # Visualization artifacts
│   └── ns-cli/          # CLI binary
├── bindings/
│   └── ns-py/           # Python bindings (PyO3/maturin)
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

# Build with CUDA support (requires nvcc)
cargo build --workspace --features cuda

# Build with Metal support (Apple Silicon, macOS)
cargo build --workspace --features metal

# Tests (including feature-gated backends)
cargo test --workspace --all-features

# Opt-in slow Rust tests (toys, SBC, NUTS quality gates)
make rust-slow-tests

# Very slow (release) regression check
make rust-very-slow-tests

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

### Apex2 Baselines (pyhf + P6 GLM)

Record a reference baseline (writes JSON under `tmp/baselines/` with a full environment fingerprint):

```bash
make apex2-baseline-record
```

Compare current HEAD vs the latest recorded baseline (writes `tmp/baseline_compare_report.json`):

```bash
make apex2-baseline-compare
```

Pre-release gate runbook: `docs/tutorials/release-gates.md`.

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
