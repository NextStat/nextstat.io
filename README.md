# NextStat

[![HEP Stable Surfaces](https://img.shields.io/badge/HEP_stable_surfaces-141%2F141-brightgreen)](docs/references/hep-stable-surface.md)
[![Release Gates](https://img.shields.io/badge/release_gates-6_required-blue)](docs/releases/release-runbook.md)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

NextStat is a statistical inference engine built in Rust with bindings for
Python, R, CLI, WASM, and a self-hosted server. It lets you run HistFactory
fits, hypothesis tests, upper limits, and measurement combinations against the
same pyhf JSON workspaces you already use — but orders of magnitude faster, with
deterministic results, and without a C++ ROOT dependency.

Beyond HEP, NextStat covers pharmacometrics (FOCE, SAEM, VPC, ICH M15),
Bayesian sampling (NUTS), GLMs, survival analysis, econometrics, and time
series — all from one engine, one install, one validation stack.

**What makes it different:** every public surface ships with a parity contract
against a reference implementation, a support matrix documenting what is
covered, and a CI gate that blocks the release if the contract breaks. The
result is an engine you can audit before you trust — not a library where
correctness is assumed from reputation.

The engine runs on two paths:

- a **deterministic reference path** for parity, validation, and release gates
- an **optimized production path** using SIMD, parallelism, and optional GPU
  acceleration (CUDA, Metal)

## Why Trust It

The key principle is simple:

> validation, not provenance

The repository is large because it includes the quality system required to make
stable-surface claims: tests, parity contracts, support matrices, benchmark
evidence, validation bundles, schemas, examples, release manifests, and release
artifacts.

Current trust anchors:

| Signal | Current State | Canonical Reference |
| --- | --- | --- |
| HEP stable surface | `141/141 stable`, `0 research` | [docs/references/hep-stable-surface.md](docs/references/hep-stable-surface.md) |
| Required release surfaces | `6` required-for-release stable surfaces | [docs/references/validation-and-release-discipline.md](docs/references/validation-and-release-discipline.md) |
| HEP governance bundle | machine-checked HEP matrix + validation bundle | [docs/references/hep-stable-surface.md](docs/references/hep-stable-surface.md) |
| Pharma qualification | `NS-VAL-001 v2.0.0`, Appendix B traceability matrix, public IQ/OQ/PQ protocol | [docs/validation/iq-oq-pq-protocol.md](docs/validation/iq-oq-pq-protocol.md) |
| Release discipline | prerelease gate, release manifest, full-fidelity local release simulation | [docs/releases/release-runbook.md](docs/releases/release-runbook.md) |
| Benchmark governance | committed evidence policy + promotion rules | [docs/releases/benchmark-artifact-policy.md](docs/releases/benchmark-artifact-policy.md) |

Concrete public evidence in the repo:

- **HEP**: `141` stable surfaces across `9` slices (HistFactory 48, GVM 47, Infrastructure 12, Unbinned 11, Viz 11, Import/Export 6, Simplified Likelihood 2, HEPData 2, Preprocess 2)
- **Pharma**: `21` IQ, `79` OQ, `25` PQ qualification IDs in the controlled protocol with bit-for-bit reproducibility checks across FOCE, SAEM, and simulation paths
- **Deterministic guarantees**: pyhf NumPy f64 is the canonical oracle for HistFactory parity; stable surfaces promise deterministic CPU parity and bit-reproducible results on the same input

If you want the engineering model behind the repository, start here:

- [Validation and Release Discipline](docs/references/validation-and-release-discipline.md)

## Status and Maturity

NextStat is production-ready for the documented stable surfaces and release
gates linked from this README. Today that means:

- HEP stable surface: `141/141 stable`, `0 research`
- `6` required-for-release surfaces blocked by machine-checked CI gates
- regulated pharma validation exposed through the public IQ/OQ/PQ protocol and validation-pack references

The maturity rule is explicit: if a surface has a support matrix, acceptance
contract, runtime gate, and release evidence, it is stable. If those contracts
are not published for a given path, do not treat the presence of code alone as
the maturity signal.

## What You Can Do

### HEP

- fit, scan, rank, hypotest, significance, and upper limits on HistFactory / pyhf JSON workspaces
- import HEPData and materialize deterministic local workspaces
- run stable scalar measurement combinations (GVM) plus advanced scenario and calibration studies
- derive and export simplified-likelihood reduced models with explicit stable/research boundaries
- build ntuple-to-workspace pipelines without a ROOT C++ dependency
- build and evaluate unbinned likelihood workflows
- generate deterministic report and validation artifacts

### Pharmacometrics and regulated reporting

- run PK/NLME workflows (FOCE, SAEM, VPC, GOF, SCM, MAP, simulation)
- produce deterministic validation-pack and report artifacts
- generate ICH M15 JSON bundle surfaces on top of the validation stack
- reuse machine-readable provenance and schema-checked output contracts

### General statistics and adjacent domains

- sample posteriors with NUTS and Bayesian helper workflows
- fit GLMs and hierarchical models
- estimate and forecast with state-space / Kalman workflows
- run econometrics and causal-inference helpers
- fit survival models

> These surfaces do not yet carry published support matrices or release gates.
> Treat them as functional but without the same stability contract as HEP or pharma.

## Quickstart

### Install (Python)

```bash
pip install nextstat
nextstat version

pip install "nextstat[bayes]"   # ArviZ / Bayesian helpers
pip install "nextstat[viz]"     # plotting extras
pip install "nextstat[all]"     # everything
```

### Try HistFactory in 30 seconds

```bash
nextstat fit --input playground/examples/simple_workspace.json
nextstat upper-limit --input playground/examples/simple_workspace.json --expected
```

Or from Python:

```python
import nextstat, json

workspace = json.load(open("playground/examples/simple_workspace.json"))
model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
result = nextstat.fit(model)

print(result.parameters)   # MLE parameter values
print(result.twice_nll)     # 2 * NLL (pyhf convention)
print(result.converged)     # True
```

More:

- [CLI Reference](docs/references/cli.md)
- [Python API](docs/references/python-api.md)
- [HistFactory stable support matrix](docs/benchmarks/histfactory-support-matrix-2026-03-17.md)

### Stable-first GVM in 5 minutes

```bash
nextstat combine-measurements-build-spec \
  --manifest docs/examples/gvm-stable-first/manifest.yaml \
  --output /tmp/gvm-spec.json

nextstat combine-measurements \
  --input /tmp/gvm-spec.json \
  --output /tmp/gvm-result.json \
  --solver auto \
  --threads 1
```

Continue with:

- `nextstat combine-measurements-calibrate`
- `nextstat combine-measurements-calibrate-study`
- `make gvm-stable-first-example`

Walkthrough:

- [HEP GVM stable-first quickstart](docs/quickstarts/hep-gvm-stable-first.md)
- [HEP GVM measurement-combination tutorial](docs/tutorials/hep-gvm-measurement-combinations.md)

### HEPData import

```bash
nextstat import hepdata --list
nextstat import hepdata \
  --dataset hepdata.116034.v1.r34 \
  --out-dir /tmp/hepdata-workspaces
```

References:

- [HEPData Import support matrix](docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md)
- [HEPData Import runtime gate](docs/benchmarks/hepdata-import-runtime-gate.md)

### Validation pack

```bash
make validation-pack
```

Key references:

- [Validation report reference](docs/references/validation-report.md)
- [IQ/OQ/PQ validation protocol](docs/validation/iq-oq-pq-protocol.md)
- [21 CFR Part 11 compliance documentation](docs/validation/21cfr-part11-compliance.md)

## Documentation Map

| Area | References |
| --- | --- |
| **Start here** | [Docs index](docs/README.md) · [Validation and release discipline](docs/references/validation-and-release-discipline.md) · [Tutorials](docs/tutorials/README.md) · [Benchmarks hub](docs/benchmarks.md) |
| **API surfaces** | [CLI](docs/references/cli.md) · [Python](docs/references/python-api.md) · [Rust](docs/references/rust-api.md) · [R](docs/references/r-bindings.md) · [Tool](docs/references/tool-api.md) · [Server](docs/references/server-api.md) |
| **Trust & releases** | [HEP stable surface](docs/references/hep-stable-surface.md) · [M15 reporting](docs/references/m15-reporting.md) · [Validation report](docs/references/validation-report.md) · [Benchmark policy](docs/releases/benchmark-artifact-policy.md) · [Release runbook](docs/releases/release-runbook.md) |
| **Tutorials & demos** | [Frequentist HEP](docs/tutorials/phase-3.1-frequentist.md) · [ROOT/TRExFitter parity](docs/tutorials/root-trexfitter-parity.md) · [Physics assistant](docs/demos/physics-assistant.md) |

## Representative Benchmark Snapshot

This repository keeps benchmark evidence in governed form rather than as
marketing-only claims. One representative example:

**Toy-based CLs “God Run”**

- Model: synthetic S+B HistFactory
- Shape: `50 channels × 4 bins`, `201 parameters`
- Load: `10,000` b-only toys + `10,000` s+b toys
- Machine: Apple M5
- Recorded: `2026-02-07`
- Snapshot commit: `88d57856`

| Tool | Wall Time | Speedup |
| --- | ---: | ---: |
| NextStat (Rayon) | `3.47 s` | `1.0×` |
| pyhf (multiprocessing, 10 procs) | `50m 11.7s` | `868.0×` |

Reproduce:

```bash
PYTHONPATH=bindings/ns-py/python python \
  scripts/god_run_benchmark.py \
  --n-toys 10000
```

For governed benchmark evidence and promotion rules, see:

- [Benchmarks hub](docs/benchmarks.md)
- [Benchmark artifact policy](docs/releases/benchmark-artifact-policy.md)

## Architecture

NextStat follows a layered architecture: inference logic depends on stable
interfaces, while translation, compute backends, and I/O stay modular.

```text
High-level logic
  ns-inference  ns-viz  ns-server  ns-cli
        |
        v
Core abstractions
  ns-core  ns-prob  ns-ad
        |
        v
Implementations and ingress
  ns-compute  ns-translate  ns-root  ns-unbinned
```

Key crates:

- `ns-core` — shared types and interfaces
- `ns-compute` — SIMD / Accelerate / CUDA / Metal kernels
- `ns-translate` — pyhf, HS3, HistFactory XML, ntuple/workspace translation
- `ns-root` — native ROOT I/O
- `ns-unbinned` — event-level likelihood components
- `ns-inference` — fits, scans, hypotests, NUTS, GLM, time series, PK/NLME
- `ns-viz` — plotting artifacts
- `ns-server` — self-hosted inference API
- `ns-cli` — command-line surface

Bindings:

- `bindings/ns-py`
- `bindings/ns-r`
- `bindings/ns-wasm`

For the deeper rationale, see:

- [White paper](docs/WHITEPAPER.md)

## Development

### Requirements

- Rust `1.93+`
- Python `3.11+`
- `maturin`
- optional: R `4.1+`, CUDA toolkit, Apple Metal-capable macOS host

### Build and test

```bash
# Build
cargo build --workspace

# Rust tests
cargo test --workspace

# Python tests (fast path)
PYTHONPATH=bindings/ns-py/python python -m pytest -q -m "not slow" tests/python

# Format and lint
cargo fmt --check
cargo clippy --workspace -- -D warnings
```

### Prerelease gate

```bash
make apex2-pre-release-gate
```

This is the canonical local prerelease entry point. It validates:

- release surface matrix
- HEP surface matrix
- HEP validation bundle
- release manifest
- full-fidelity local release simulation
- separate governance and performance outcomes with distinct summaries and exit codes

References:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Release Runbook](docs/releases/release-runbook.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

All commits must include DCO sign-off:

```bash
git commit -s
```

## License

NextStat uses a dual-licensing model:

- open source: [LICENSE](LICENSE) (`AGPL-3.0-or-later`)
- commercial alternative: [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL)

## Contact

- Website: https://nextstat.io
- GitHub: https://github.com/NextStat/nextstat.io
