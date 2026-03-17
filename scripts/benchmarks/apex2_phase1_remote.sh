#!/usr/bin/env bash
set -euo pipefail

# Apex2 Phase 1 remote runner (EPYC canonical).
#
# Runs:
# - builds + installs NextStat wheel from the *current working tree* (rsync snapshot)
# - installs benchmark harness deps (Python + optional R baselines)
# - runs scripts/benchmarks/apex2_phase1.sh remotely
# - syncs artifacts back locally
#
# Usage:
#   APEX2_HOST=epyc-node bash scripts/benchmarks/apex2_phase1_remote.sh
#
# Optional overrides:
#   APEX2_HOST
#   APEX2_REMOTE_REPO (default: /workspace/nextstat.io)
#   APEX2_REMOTE_OUT_ROOT (default: /tmp/apex2_phase1_<STAMP>_<host>)
#   APEX2_LOCAL_OUT_ROOT (default: benchmarks/artifacts/apex2_phase1_<STAMP>/<host>)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

APEX2_HOST="${APEX2_HOST:-epyc-node}"
APEX2_REMOTE_REPO="${APEX2_REMOTE_REPO:-/workspace/nextstat.io}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_OUT="${APEX2_REMOTE_OUT_ROOT:-/tmp/apex2_phase1_${STAMP}_${APEX2_HOST}}"
LOCAL_OUT="${APEX2_LOCAL_OUT_ROOT:-$ROOT_DIR/benchmarks/artifacts/apex2_phase1_${STAMP}/${APEX2_HOST}}"
REMOTE_VENV="${APEX2_REMOTE_VENV:-/tmp/apex2_phase1_venv_${STAMP}_${APEX2_HOST}}"
SUITES_FILTER="${APEX2_PHASE1_SUITES:-}"

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new "$APEX2_HOST")

mkdir -p "$LOCAL_OUT"

echo "[apex2-remote] host=${APEX2_HOST}"
echo "[apex2-remote] remote_repo=${APEX2_REMOTE_REPO}"
echo "[apex2-remote] remote_out=${REMOTE_OUT}"
echo "[apex2-remote] remote_venv=${REMOTE_VENV}"
echo "[apex2-remote] local_out=${LOCAL_OUT}"
if [[ -n "${SUITES_FILTER}" ]]; then
  echo "[apex2-remote] suites_filter=${SUITES_FILTER}"
fi

echo "[apex2-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version || true; Rscript --version 2>/dev/null | head -n 1 || true"

echo "[apex2-remote] ensure remote repo dir exists..."
"${SSH_BASE[@]}" "mkdir -p '${APEX2_REMOTE_REPO}'"

echo "[apex2-remote] rsync working tree snapshot..."
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

rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools >/dev/null
"$VENV/bin/python" -m pip install -r env/python/requirements.txt >/dev/null
"$VENV/bin/python" -m pip install -r env/python/requirements-econometrics-baselines.txt >/dev/null

# Phase 1 baseline deps (idempotent)
# Keep scipy compatible with pyfixest (installed via econometrics baseline reqs on many hosts).
# pyfixest requires scipy<1.16 as of 0.40.x.
"$VENV/bin/python" -m pip install "scipy<1.16" statsmodels lifelines arch scikit-learn glum >/dev/null
EOS

echo "[apex2-remote] ensure R baseline deps (cmprsk) if R is present..."
"${SSH_BASE[@]}" bash -s -- <<'EOS'
set -euo pipefail
if ! command -v Rscript >/dev/null 2>&1; then
  echo "[apex2-remote] Rscript not found; skipping R baselines"
  exit 0
fi

Rscript -e 'ok <- suppressWarnings(requireNamespace("cmprsk", quietly=TRUE)); if (!ok) { message("installing cmprsk..."); install.packages("cmprsk", repos="https://cloud.r-project.org"); }'
EOS

echo "[apex2-remote] build + install nextstat wheel (maturin develop)..."
"${SSH_BASE[@]}" bash -s -- "$APEX2_REMOTE_REPO" "$REMOTE_VENV" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
: "${REPO:?missing REPO arg}"
: "${VENV:?missing VENV arg}"
cd "$REPO/bindings/ns-py"

rm -rf "$REPO/.venv" "$REPO/.venv_epyc" "$REPO/.venv_v100" >/dev/null 2>&1 || true

"$VENV/bin/python" -m pip install -U maturin >/dev/null
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

bash "$APEX2_REMOTE_REPO/benchmarks/nextstat-public-benchmarks/scripts/install_local_nextstat_python.sh" \
  "$APEX2_REMOTE_REPO" "$VENV" "$CARGO_TARGET_DIR"
EOS

echo "[apex2-remote] run apex2 phase1..."
"${SSH_BASE[@]}" bash -s -- "$APEX2_REMOTE_REPO" "$REMOTE_OUT" "$REMOTE_VENV" "$SUITES_FILTER" <<'EOS'
set -euo pipefail
REPO="$1"
OUT="$2"
VENV="$3"
FILTER="$4"
: "${REPO:?missing REPO arg}"
: "${OUT:?missing OUT arg}"
: "${VENV:?missing VENV arg}"
cd "$REPO"
export APEX2_PYTHON_BIN="$VENV/bin/python"
export APEX2_PHASE1_SUITES="${FILTER:-}"
mkdir -p "$OUT"
rc=0
scripts/benchmarks/apex2_phase1.sh "$OUT" "42,123,777" || rc=$?
echo "$rc" > "$OUT/apex2_phase1_exit_code.txt"
# Do not fail fast; we still want strict validation + artifact sync for debugging.
exit 0
EOS

echo "[apex2-remote] validate artifacts (strict) x2..."
"${SSH_BASE[@]}" bash -s -- "$APEX2_REMOTE_REPO" "$REMOTE_OUT" "$REMOTE_VENV" <<'EOS'
set -euo pipefail
REPO="$1"
OUT="$2"
VENV="$3"
: "${REPO:?missing REPO arg}"
: "${OUT:?missing OUT arg}"
: "${VENV:?missing VENV arg}"
cd "$REPO/benchmarks/nextstat-public-benchmarks"
"$VENV/bin/python" scripts/validate_artifacts.py --strict "$OUT"
"$VENV/bin/python" scripts/validate_artifacts.py --strict "$OUT"
EOS

echo "[apex2-remote] sync artifacts back..."
rsync -az "${APEX2_HOST}:${REMOTE_OUT}/" "$LOCAL_OUT/"

echo "[apex2-remote] done: $LOCAL_OUT"
