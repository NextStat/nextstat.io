#!/usr/bin/env bash
# PF3.1-OPT2 multi-GPU benchmark setup for Vast.ai (4× A40)
# Usage: scp this to the instance, then: bash setup_vastai_multi_gpu.sh
set -euo pipefail

echo "=== PF3.1 Multi-GPU Benchmark Setup ==="
echo "Host: $(hostname)"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Verify GPUs
echo ""
echo "=== GPU inventory ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $N_GPU GPUs"

# 2. Install Rust (if not present)
if ! command -v rustup &>/dev/null; then
    echo ""
    echo "=== Installing Rust ==="
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi
source "$HOME/.cargo/env" 2>/dev/null || true
rustc --version
cargo --version

# 3. Install system deps
echo ""
echo "=== System deps ==="
apt-get update -qq && apt-get install -y -qq build-essential pkg-config libssl-dev git cmake 2>/dev/null || true

# 4. Clone repo
REPO_DIR="$HOME/nextstat.io"
if [ ! -d "$REPO_DIR" ]; then
    echo ""
    echo "=== Cloning nextstat.io ==="
    git clone --depth 1 https://github.com/NextStat/nextstat.io.git "$REPO_DIR"
else
    echo ""
    echo "=== Updating nextstat.io ==="
    cd "$REPO_DIR" && git pull origin main
fi

# 5. Build with CUDA
echo ""
echo "=== Building (release, CUDA) ==="
cd "$REPO_DIR"
# Set CUDA path if needed
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version || echo "WARNING: nvcc not found, build may fail"

cargo build -p ns-cli --features cuda --release 2>&1 | tail -5
echo "Build complete."

# 6. Quick smoke test
echo ""
echo "=== Smoke test (1 GPU, 5 toys) ==="
cd "$REPO_DIR"
# Create minimal analytical spec for testing
mkdir -p /tmp/pf31_bench
python3 -c "
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
rng = np.random.default_rng(42)
x = rng.normal(0, 1, size=10000)
x = x[(x >= -6.0) & (x <= 6.0)]
table = pa.table({'x': x})
pq.write_table(table, '/tmp/pf31_bench/data.parquet')
print(f'Generated {len(x)} events')
" 2>/dev/null || echo "pyarrow not available, skipping data gen"

cat > /tmp/pf31_bench/spec.json << 'SPEC'
{
  "schema_version": "nextstat_unbinned_spec_v0",
  "model": {
    "poi": "mu",
    "parameters": [
      {"name": "mu", "init": 0.0, "bounds": [-3.0, 3.0]},
      {"name": "sigma", "init": 1.0, "bounds": [0.1, 5.0]},
      {"name": "lam", "init": 0.5, "bounds": [0.01, 2.0]},
      {"name": "n_sig", "init": 5000.0, "bounds": [0.0, 15000.0]},
      {"name": "n_bkg", "init": 5000.0, "bounds": [0.0, 15000.0]}
    ]
  },
  "channels": [
    {
      "name": "SR",
      "data": {"file": "data.parquet"},
      "observables": [{"name": "x", "bounds": [-6.0, 6.0]}],
      "processes": [
        {
          "name": "signal",
          "pdf": {"type": "gaussian", "observable": "x", "params": ["mu", "sigma"]},
          "yield": {"type": "parameter", "name": "n_sig"}
        },
        {
          "name": "background",
          "pdf": {"type": "exponential", "observable": "x", "params": ["lam"]},
          "yield": {"type": "parameter", "name": "n_bkg"}
        }
      ]
    }
  ]
}
SPEC

target/release/nextstat unbinned-fit-toys --config /tmp/pf31_bench/spec.json \
    --n-toys 5 --gpu cuda --seed 42 2>&1 | python3 -c "
import json,sys
d = json.load(sys.stdin)
r = d['results']
print(f'Smoke: {r[\"n_toys\"]} toys, {r[\"n_converged\"]} converged, {r[\"n_error\"]} errors')
" || echo "Smoke test output above"

echo ""
echo "=== Setup complete. Ready for benchmark matrix. ==="
echo "Run: bash scripts/benchmarks/pf31_multi_gpu_matrix.sh"
