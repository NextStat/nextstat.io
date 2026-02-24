#!/usr/bin/env bash
set -euo pipefail

# Apex2 Phase 0 remote runner (EPYC/V100/etc).
#
# Runs:
# - builds + installs NextStat wheel from the *current working tree* (rsync snapshot)
# - installs benchmark harness deps (pinned where available)
# - runs scripts/benchmarks/apex2_phase0.sh remotely
# - syncs artifacts back locally
#
# Usage:
#   APEX2_HOST=nextstat-bench bash scripts/benchmarks/apex2_phase0_remote.sh
#   APEX2_HOST=v100          bash scripts/benchmarks/apex2_phase0_remote.sh
#
# Optional overrides:
#   APEX2_HOST, APEX2_USER, APEX2_PORT, APEX2_KEY
#   APEX2_REMOTE_REPO (default: /root/nextstat.io)
#   APEX2_REMOTE_OUT_ROOT (default: /tmp/apex2_phase0_<STAMP>_<host>)
#   APEX2_LOCAL_OUT_ROOT (default: benchmarks/artifacts/apex2_phase0_<STAMP>/<host>)
#   APEX2_BUILD_FEATURES (default: auto: cuda if nvidia-smi exists else "")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

APEX2_HOST="${APEX2_HOST:-nextstat-bench}"
APEX2_REMOTE_REPO="${APEX2_REMOTE_REPO:-/root/nextstat.io}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_OUT="${APEX2_REMOTE_OUT_ROOT:-/tmp/apex2_phase0_${STAMP}_${APEX2_HOST}}"
LOCAL_OUT="${APEX2_LOCAL_OUT_ROOT:-$ROOT_DIR/benchmarks/artifacts/apex2_phase0_${STAMP}/${APEX2_HOST}}"
REMOTE_VENV="${APEX2_REMOTE_VENV:-/tmp/apex2_phase0_venv_${STAMP}_${APEX2_HOST}}"

# Use the user's SSH config host aliases (recommended). This avoids hardcoding keys/users/ports
# and matches the existing benchmark ops runbooks.
SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new "$APEX2_HOST")

mkdir -p "$LOCAL_OUT"

echo "[apex2-remote] host=${APEX2_HOST}"
echo "[apex2-remote] remote_repo=${APEX2_REMOTE_REPO}"
echo "[apex2-remote] remote_out=${REMOTE_OUT}"
echo "[apex2-remote] remote_venv=${REMOTE_VENV}"
echo "[apex2-remote] local_out=${LOCAL_OUT}"

echo "[apex2-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version || true"

echo "[apex2-remote] ensure remote repo dir exists..."
"${SSH_BASE[@]}" "mkdir -p '${APEX2_REMOTE_REPO}'"

echo "[apex2-remote] rsync working tree snapshot..."
# Keep sync reasonably small: exclude build outputs and large local artifact bundles.
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude '.nextstat-cargo-target/' \
  --exclude 'node_modules/' \
  --exclude 'benchmarks/artifacts/' \
  --exclude 'benchmarks/unbinned/artifacts/' \
  --exclude 'benchmarks/nextstat-public-benchmarks/.venv/' \
  --exclude '.venv*/' \
  --exclude 'tmp/' \
  --exclude 'tmp*/' \
  --exclude 'bindings/nextstat-nlp/' \
  --exclude 'docs/blog/artifacts/' \
  --exclude '**/__pycache__/' \
  --exclude '.DS_Store' \
  "$ROOT_DIR/" \
  "${APEX2_HOST}:${APEX2_REMOTE_REPO}/"

echo "[apex2-remote] setup venv for public benchmark harness..."
"${SSH_BASE[@]}" bash -s -- "$APEX2_REMOTE_REPO" "$REMOTE_VENV" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
: "${REPO:?missing REPO arg}"
: "${VENV:?missing VENV arg}"
cd "$REPO/benchmarks/nextstat-public-benchmarks"

# Always build an ephemeral Linux venv (do NOT reuse any synced venv dirs).
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools >/dev/null
"$VENV/bin/python" -m pip install -r env/python/requirements.txt >/dev/null
"$VENV/bin/python" -m pip install -r env/python/requirements-econometrics-baselines.txt >/dev/null

# Phase 0 extras (timeseries + GLM competitors)
"$VENV/bin/python" -m pip install arch pykalman scikit-learn glum >/dev/null
EOS

echo "[apex2-remote] build + install nextstat wheel (maturin develop)..."
"${SSH_BASE[@]}" bash -s -- "$APEX2_REMOTE_REPO" "$REMOTE_VENV" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
: "${REPO:?missing REPO arg}"
: "${VENV:?missing VENV arg}"
cd "$REPO/bindings/ns-py"

# Ensure no stale/broken repo-root venvs interfere with maturin's venv detection.
rm -rf "$REPO/.venv" "$REPO/.venv_epyc" "$REPO/.venv_v100" >/dev/null 2>&1 || true

FEATURES=""
if command -v nvidia-smi >/dev/null 2>&1; then
  FEATURES="--features cuda"
fi

# CUDA 13+ may drop offline PTX compilation support for older architectures (e.g. sm_70 on V100).
# Prefer a CUDA 12.x nvcc if present so we can still build Volta PTX.
if [[ -n "$FEATURES" ]]; then
  for cuda_root in /usr/local/cuda-12.9 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [[ -x "${cuda_root}/bin/nvcc" ]]; then
      export CUDA_HOME="${cuda_root}"
      export CUDA_PATH="${cuda_root}"
      export PATH="${cuda_root}/bin:${PATH}"
      echo "[apex2-remote] using CUDA_HOME=${CUDA_HOME} (nvcc_release=$(nvcc --version 2>/dev/null | grep -m1 -E \"release\" || true))"
      break
    fi
  done
fi

#
# NOTE: Many modern Linux distros enforce PEP 668 for the system Python. Install
# build tooling inside our ephemeral venv instead.
"$VENV/bin/python" -m pip install -U maturin >/dev/null

# Make maturin detect the intended venv (it requires VIRTUAL_ENV or .venv).
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

if [[ -n "$FEATURES" ]]; then
  "$VENV/bin/python" -m maturin develop --release $FEATURES --pip-path "$VENV/bin/pip"
else
  "$VENV/bin/python" -m maturin develop --release --pip-path "$VENV/bin/pip"
fi
EOS

echo "[apex2-remote] run apex2 phase0..."
"${SSH_BASE[@]}" bash -s -- "$APEX2_REMOTE_REPO" "$REMOTE_OUT" "$REMOTE_VENV" <<'EOS'
set -euo pipefail
REPO="$1"
OUT="$2"
VENV="$3"
: "${REPO:?missing REPO arg}"
: "${OUT:?missing OUT arg}"
: "${VENV:?missing VENV arg}"
cd "$REPO"
export APEX2_PYTHON_BIN="$VENV/bin/python"
if command -v nvidia-smi >/dev/null 2>&1; then
  # GPU stands are used primarily for LAPS. CPU benchmark suites can be extremely slow
  # on many GPU instances, so default to GPU-only unless overridden.
  export APEX2_GPU_ONLY="${APEX2_GPU_ONLY:-1}"
fi
mkdir -p "$OUT"
scripts/benchmarks/apex2_phase0.sh "$OUT" "42,123,777"
EOS

echo "[apex2-remote] sync artifacts back..."
rsync -az "${APEX2_HOST}:${REMOTE_OUT}/" "$LOCAL_OUT/"

echo "[apex2-remote] done: $LOCAL_OUT"
