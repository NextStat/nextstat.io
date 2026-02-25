#!/usr/bin/env bash
set -euo pipefail

# Remote runner for nextstat-nlp workflow verification (EPYC/V100/etc).
#
# Runs (on remote host):
# - rsync snapshot of current working tree (including bindings/nextstat-nlp)
# - builds + installs nextstat wheel (maturin develop)
# - installs nextstat-nlp (onnx backend) in the same ephemeral venv
# - runs 3x reproducible workflow matrix:
#     python bindings/nextstat-nlp/tools/run_workflow_matrix.py --backends heuristic onnx --n-repeats 3
# - syncs artifacts back locally
#
# Usage:
#   APEX2_HOST=nextstat-bench bash scripts/nlp/nlp_workflow_remote.sh
#   APEX2_HOST=v100          bash scripts/nlp/nlp_workflow_remote.sh
#
# Optional overrides:
#   APEX2_HOST
#   NLP_REMOTE_REPO (default: /root/nextstat.io)
#   NLP_REMOTE_OUT_ROOT (default: /tmp/nlp_workflow_<STAMP>_<host>)
#   NLP_LOCAL_OUT_ROOT (default: benchmarks/artifacts/nlp_workflow_<STAMP>/<host>)
#   NLP_REMOTE_VENV (default: /tmp/nlp_workflow_venv_<STAMP>_<host>)
#   NLP_BUILD_FEATURES (default: empty; set to '--features cuda' if you explicitly want CUDA build)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

APEX2_HOST="${APEX2_HOST:-nextstat-bench}"
NLP_REMOTE_REPO="${NLP_REMOTE_REPO:-/root/nextstat.io}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_OUT="${NLP_REMOTE_OUT_ROOT:-/tmp/nlp_workflow_${STAMP}_${APEX2_HOST}}"
LOCAL_OUT="${NLP_LOCAL_OUT_ROOT:-$ROOT_DIR/benchmarks/artifacts/nlp_workflow_${STAMP}/${APEX2_HOST}}"
REMOTE_VENV="${NLP_REMOTE_VENV:-/tmp/nlp_workflow_venv_${STAMP}_${APEX2_HOST}}"
NLP_BUILD_FEATURES="${NLP_BUILD_FEATURES:-}"

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new "$APEX2_HOST")

mkdir -p "$LOCAL_OUT"

echo "[nlp-remote] host=${APEX2_HOST}"
echo "[nlp-remote] remote_repo=${NLP_REMOTE_REPO}"
echo "[nlp-remote] remote_out=${REMOTE_OUT}"
echo "[nlp-remote] remote_venv=${REMOTE_VENV}"
echo "[nlp-remote] local_out=${LOCAL_OUT}"
if [[ -n "$NLP_BUILD_FEATURES" ]]; then
  echo "[nlp-remote] build_features=${NLP_BUILD_FEATURES}"
fi

echo "[nlp-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version || true"

echo "[nlp-remote] ensure remote repo dir exists..."
"${SSH_BASE[@]}" "mkdir -p '${NLP_REMOTE_REPO}'"

echo "[nlp-remote] rsync working tree snapshot..."
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
  --exclude 'docs/blog/artifacts/' \
  --exclude '**/__pycache__/' \
  --exclude '.DS_Store' \
  "$ROOT_DIR/" \
  "${APEX2_HOST}:${NLP_REMOTE_REPO}/"

echo "[nlp-remote] setup venv + install deps..."
"${SSH_BASE[@]}" bash -s -- "$NLP_REMOTE_REPO" "$REMOTE_VENV" "$NLP_BUILD_FEATURES" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
BUILD_FEATURES="${3-}"
: "${REPO:?missing REPO arg}"
: "${VENV:?missing VENV arg}"

echo "[nlp-remote] disk check..."
df -h / | head -n 2 || true
avail_k="$(df -Pk / | tail -n 1 | awk '{print $4}')"
if [[ -n "$avail_k" && "$avail_k" -lt 2097152 ]]; then
  echo "ERROR: Not enough free space on / (${avail_k}K available). Clean up the host and retry." >&2
  exit 2
fi

rm -rf "$VENV"
python3 -m venv --without-pip "$VENV"
if [[ ! -x "$VENV/bin/pip" ]]; then
  # Some minimal images ship python3 without ensurepip.
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  "$VENV/bin/python" /tmp/get-pip.py >/dev/null
fi
"$VENV/bin/python" -m pip install -U pip wheel setuptools >/dev/null

# Build tooling (PEP 668 safe).
"$VENV/bin/python" -m pip install -U maturin >/dev/null

cd "$REPO/bindings/ns-py"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

# Some ephemeral hosts occasionally crash `ld.lld` during large shared-library links.
# Prefer bfd + single-job builds for stability; this workflow is not performance-sensitive.
if command -v ld.bfd >/dev/null 2>&1; then
  export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-fuse-ld=bfd"
fi
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

if [[ -n "$BUILD_FEATURES" ]]; then
  # If building with CUDA on older GPUs (e.g. V100 sm_70), prefer CUDA 12.x if present.
  if command -v nvidia-smi >/dev/null 2>&1; then
    for cuda_root in /usr/local/cuda-12.9 /usr/local/cuda-12.5 /usr/local/cuda; do
      if [[ -x "${cuda_root}/bin/nvcc" ]]; then
        export CUDA_HOME="${cuda_root}"
        export CUDA_PATH="${cuda_root}"
        export PATH="${cuda_root}/bin:${PATH}"
        break
      fi
    done
  fi
  "$VENV/bin/python" -m maturin develop --release $BUILD_FEATURES --pip-path "$VENV/bin/pip"
else
  # Default: CPU-only build, even on GPU instances (NLP workflow doesn't need CUDA).
  "$VENV/bin/python" -m maturin develop --release --pip-path "$VENV/bin/pip"
fi

cd "$REPO"
"$VENV/bin/python" -m pip install -e "bindings/nextstat-nlp[onnx]" requests >/dev/null
EOS

echo "[nlp-remote] run workflow matrix (3x)..."
"${SSH_BASE[@]}" bash -s -- "$NLP_REMOTE_REPO" "$REMOTE_OUT" "$REMOTE_VENV" <<'EOS'
set -euo pipefail
REPO="$1"
OUT="$2"
VENV="$3"
: "${REPO:?missing REPO arg}"
: "${OUT:?missing OUT arg}"
: "${VENV:?missing VENV arg}"

mkdir -p "$OUT"
cd "$REPO"

"$VENV/bin/python" bindings/nextstat-nlp/tools/run_workflow_matrix.py \
  --backends heuristic onnx \
  --n-repeats 3 \
  --num-threads 8 \
  --out-dir "$OUT"
EOS

echo "[nlp-remote] sync artifacts back..."
rsync -az "${APEX2_HOST}:${REMOTE_OUT}/" "$LOCAL_OUT/"

echo "[nlp-remote] done: $LOCAL_OUT"
