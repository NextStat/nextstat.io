# Installation & Quickstart

## Installation

### Step 0: Prerequisites

| Tool   | Minimum Version | How to Check         |
|--------|----------------|----------------------|
| Python | 3.11           | `python3 --version`  |
| pip    | 23.0           | `pip --version`      |

If you don't have Python yet:

```bash
# macOS (Homebrew)
brew install python@3.12

# Ubuntu / Debian
sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip

# Windows — download from https://python.org/downloads
# During install, check "Add Python to PATH"
```

### Step 1: Install via pip (recommended)

Pre-built binary wheels are available for:
- Linux x86_64 and aarch64
- macOS Apple Silicon (arm64)
- macOS Intel (x86_64)
- Windows x86_64

#### 1a. Create a virtual environment (strongly recommended)

A virtual environment keeps NextStat and its dependencies isolated from your system Python.

```bash
# Create a new virtual environment called "ns-env"
python3 -m venv ns-env

# Activate it
# macOS / Linux:
source ns-env/bin/activate
# Windows (PowerShell):
ns-env\Scripts\Activate.ps1
# Windows (cmd):
ns-env\Scripts\activate.bat

# Your prompt should now show (ns-env) at the beginning
```

#### 1b. Install the package

```bash
pip install nextstat
```

This downloads a pre-compiled wheel (~15 MB) and installs it. No Rust compiler needed.

#### 1c. Verify the installation

```bash
python3 -c "import nextstat; print(nextstat.__version__)"
```

Expected output:

```
0.9.5
```

#### 1d. Install optional extras

| Command                          | What it adds                                        |
|----------------------------------|-----------------------------------------------------|
| `pip install "nextstat[bayes]"`  | ArviZ, xarray — for Bayesian diagnostics and traces |
| `pip install "nextstat[viz]"`    | matplotlib — for built-in plotting functions         |
| `pip install "nextstat[all]"`    | Everything above combined                            |

### Alternative: Install as a Rust crate

If you are building a Rust application and want to use NextStat as a library:

```bash
# Make sure you have Rust 1.93+
rustup update stable
rustc --version   # should print 1.93.0 or higher

# Add NextStat crates to your project
cargo add ns-core ns-inference ns-compute
```

Then in your `main.rs` or `lib.rs`:

```rust
use ns_inference::mle::MaximumLikelihoodEstimator;
// You're ready to go!
```

### Alternative: Build from source

Only needed if you want to modify NextStat itself, or if no pre-built wheel exists for your platform.

#### Requirements

| Tool    | Version | Install                                                          |
|---------|---------|------------------------------------------------------------------|
| Rust    | 1.93+   | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Python  | 3.11+   | see above                                                        |
| maturin | 1.0+    | `pip install maturin`                                            |
| git     | any     | `brew install git` / `apt install git`                           |

#### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/NextStat/nextstat.io.git
cd nextstat.io

# 2. Build the entire Rust workspace (takes 2-5 minutes on first build)
cargo build --release

# 3. Create a virtual environment for Python
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 4. Install maturin (the Rust-Python build tool)
pip install maturin

# 5. Build and install the Python bindings in development mode
cd bindings/ns-py
maturin develop --release

# 6. Verify
python3 -c "import nextstat; print(nextstat.__version__)"
# Expected: 0.9.4
```

### GPU Acceleration (optional)

NextStat supports GPU-accelerated toy fitting and batch NLL evaluation.
This is optional — everything works on CPU by default.

| GPU            | Feature Flag | Requirement                        |
|----------------|-------------|-------------------------------------|
| NVIDIA         | `cuda`      | CUDA Toolkit 12.0+ and nvcc in PATH |
| Apple Silicon  | `metal`     | macOS 14+ (Sonoma or later)         |

```bash
# Build from source with CUDA support
cargo build --workspace --features cuda

# Build from source with Metal support (Apple Silicon)
cargo build --workspace --features metal
```

### Troubleshooting

**"No matching distribution found"**

This means there is no pre-built wheel for your platform. Check your Python version
(`python3 --version`) — you need 3.11 or higher. If you are on an unusual architecture,
build from source (see above).

**"pip: command not found"**

Try `pip3` instead of `pip`. On some systems, `pip` is only available inside a virtual
environment. Create one first (Step 1a above).

**"error: linker 'cc' not found" (building from source)**

You need a C compiler. Install build tools:

```bash
# macOS
xcode-select --install

# Ubuntu / Debian
sudo apt install build-essential

# Fedora
sudo dnf install gcc
```

---

## Quickstart

### Step 1: Get a sample workspace

A "workspace" is a JSON file that describes your statistical model: the observed data,
signal and background expectations, and systematic uncertainties. NextStat uses the same
JSON format as [pyhf](https://github.com/scikit-hep/pyhf).

Create a minimal workspace with one signal region:

```bash
cat > workspace.json << 'EOF'
{
  "channels": [
    {
      "name": "singlechannel",
      "samples": [
        {
          "name": "signal",
          "data": [5.0, 10.0],
          "modifiers": [
            { "name": "mu", "type": "normfactor", "data": null }
          ]
        },
        {
          "name": "background",
          "data": [50.0, 60.0],
          "modifiers": [
            { "name": "uncorr_bkguncrt", "type": "shapesys", "data": [5.0, 12.0] }
          ]
        }
      ]
    }
  ],
  "observations": [
    { "name": "singlechannel", "data": [55.0, 65.0] }
  ],
  "measurements": [
    { "name": "Measurement", "config": { "poi": "mu", "parameters": [] } }
  ],
  "version": "1.0.0"
}
EOF
```

### Step 2: Run your first fit (Python)

Create a script called `my_first_fit.py`:

```python
import json
import nextstat

# 1. Load the workspace JSON
with open("workspace.json") as f:
    workspace = json.load(f)

# 2. Build a model from the workspace
model = nextstat.from_pyhf(json.dumps(workspace))

# 3. Run a maximum-likelihood fit
result = nextstat.fit(model)

# 4. Print the results
poi_idx = model.poi_index()
print("Signal strength (mu):", result.bestfit[poi_idx])
print("Uncertainty:         ", result.uncertainties[poi_idx])
print("All parameters:      ", result.bestfit)
print("All uncertainties:   ", result.uncertainties)
```

Run it:

```bash
python3 my_first_fit.py
```

Expected output (values may differ slightly):

```
Signal strength (mu): 0.9999...
Uncertainty:          0.3...
All parameters:       [0.999..., 1.0..., 0.99...]
All uncertainties:    [0.3..., 0.08..., 0.15...]
```

**What just happened?** NextStat found the parameter values that best describe your
observed data. The signal strength `mu ≈ 1.0` means the observed data is consistent
with the signal+background hypothesis. The other parameters are nuisance parameters
(systematic uncertainties) that were profiled during the fit.

### Step 3: Test a hypothesis

A hypothesis test tells you whether the data is compatible with a given signal strength.
The CLs method is the standard approach in particle physics:

```python
import json
import nextstat

with open("workspace.json") as f:
    workspace = json.load(f)

model = nextstat.from_pyhf(json.dumps(workspace))

# Test the hypothesis mu=1.0 (signal exists at nominal strength)
result = nextstat.hypotest(model, mu_test=1.0)

print("CLs value:     ", result.cls)
print("CLs+b value:   ", result.clsb)
print("CLb value:     ", result.clb)
print("Excluded (95%)?", "YES" if result.cls < 0.05 else "NO")
```

Expected output:

```
CLs value:      0.43...
CLs+b value:    0.24...
CLb value:      0.55...
Excluded (95%)? NO
```

**What does this mean?** CLs > 0.05 means we cannot exclude the signal hypothesis at
95% confidence level. The signal is compatible with the data.

### Step 4: Compute upper limits (Brazil band)

An upper limit scan finds the maximum signal strength that is still compatible with the data:

```python
import json
import nextstat

with open("workspace.json") as f:
    workspace = json.load(f)

model = nextstat.from_pyhf(json.dumps(workspace))

# Compute expected and observed upper limits
limits = nextstat.upper_limit(model)

print("Observed upper limit:", limits.observed)
print("Expected -2σ:       ", limits.expected_minus2)
print("Expected -1σ:       ", limits.expected_minus1)
print("Expected median:    ", limits.expected)
print("Expected +1σ:       ", limits.expected_plus1)
print("Expected +2σ:       ", limits.expected_plus2)
```

### Step 5: Use the command-line interface

NextStat also ships a CLI binary. All the same operations are available from the terminal:

```bash
# Fit a workspace
nextstat fit --input workspace.json

# Hypothesis test (asymptotic CLs)
nextstat hypotest --input workspace.json --mu 1.0

# Hypothesis test with expected bands
nextstat hypotest --input workspace.json --mu 1.0 --expected-set

# Upper limit scan (201 points from mu=0 to mu=5)
nextstat upper-limit --input workspace.json \
  --expected --scan-start 0 --scan-stop 5 --scan-points 201

# Toy-based hypothesis test (10k toys, all CPU cores)
nextstat hypotest-toys --input workspace.json \
  --mu 1.0 --n-toys 10000 --seed 42 --threads 0

# GPU-accelerated toys (NVIDIA)
nextstat hypotest-toys --input workspace.json \
  --mu 1.0 --n-toys 10000 --gpu cuda

# GPU-accelerated toys (Apple Silicon)
nextstat hypotest-toys --input workspace.json \
  --mu 1.0 --n-toys 10000 --gpu metal
```

### Step 6: Use from Rust (optional)

If you are a Rust developer, you can use NextStat as a library in your own project:

```rust
use ns_inference::mle::MaximumLikelihoodEstimator;
use ns_translate::pyhf::{HistFactoryModel, Workspace};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the workspace
    let json = std::fs::read_to_string("workspace.json")?;
    let workspace: Workspace = serde_json::from_str(&json)?;

    // 2. Build the model
    let model = HistFactoryModel::from_workspace(&workspace)?;

    // 3. Fit
    let mle = MaximumLikelihoodEstimator::new();
    let result = mle.fit(&model)?;

    // 4. Print results
    println!("Best-fit parameters: {:?}", result.parameters);
    println!("NLL at minimum:      {}", result.nll);
    Ok(())
}
```

### Try it in your browser (no install needed)

Don't want to install anything yet? Try the [WASM Playground](https://nextstat.io/playground) —
it runs NextStat entirely in your browser via WebAssembly. No server, no Python, no setup.

### What to explore next

- [Architecture](/docs/architecture) — understand how NextStat is built
- [Python API reference](/docs/python-api) — all functions and classes
- [Bayesian sampling (NUTS)](/docs/bayesian) — posterior inference with MCMC
- [Regression & GLM](/docs/regression) — linear, logistic, Poisson models
- [Survival analysis](/docs/survival) — Kaplan-Meier, Cox PH, parametric models
- [GPU acceleration](/docs/gpu) — CUDA and Metal for batch fitting
- [CLI reference](/docs/cli) — all command-line options
