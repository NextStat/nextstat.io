#!/usr/bin/env bash
set -euo pipefail

# Remote EPYC runner for the generic NextStat sampler-matrix benchmark harness.
#
# Uses an rsynced snapshot of the current working tree, builds an ephemeral wheel
# via `maturin develop` with a tmp-backed cargo target, runs the benchmark, and
# syncs the JSON/Markdown artifacts back locally.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"

BENCH_HOST="${BENCH_HOST:-nextstat-bench}"
BENCH_SSH_USER="${BENCH_SSH_USER:-}"
BENCH_SSH_PORT="${BENCH_SSH_PORT:-}"
BENCH_SSH_KEY="${BENCH_SSH_KEY:-}"
BENCH_SKIP_BUILD="${BENCH_SKIP_BUILD:-0}"
BENCH_SCRIPT="${BENCH_SCRIPT:-scripts/benchmarks/bench_sampler_matrix.py}"

BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/nextstat_sampler_matrix_bench_repo_${STAMP}}"
BENCH_REMOTE_VENV="${BENCH_REMOTE_VENV:-/tmp/nextstat_sampler_matrix_bench_venv_${STAMP}}"
BENCH_REMOTE_OUT="${BENCH_REMOTE_OUT:-/tmp/bench_sampler_matrix_${STAMP}_${BENCH_HOST}}"
BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/nextstat_sampler_matrix_bench_target_${STAMP}}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-$ROOT_DIR/tmp/bench_sampler_matrix_${STAMP}/${BENCH_HOST}}"

BENCH_SEEDS="${BENCH_SEEDS:-42,123,777}"
BENCH_MODELS="${BENCH_MODELS:-std_normal_10d,eight_schools,glm_logistic,funnel_ncp_10d}"
BENCH_METHODS="${BENCH_METHODS:-nuts,walnuts,mams}"
BENCH_CHAINS="${BENCH_CHAINS:-4}"
BENCH_WARMUP="${BENCH_WARMUP:-1000}"
BENCH_SAMPLES="${BENCH_SAMPLES:-1000}"
BENCH_GLM_N="${BENCH_GLM_N:-1000}"
BENCH_GLM_P="${BENCH_GLM_P:-10}"
BENCH_METRIC="${BENCH_METRIC:-diagonal}"
BENCH_TARGET_ACCEPT="${BENCH_TARGET_ACCEPT:-0.8}"
BENCH_MAX_TREEDEPTH="${BENCH_MAX_TREEDEPTH:-10}"
BENCH_MAMS_MAX_LEAPFROG="${BENCH_MAMS_MAX_LEAPFROG:-1024}"
BENCH_MAMS_DIAGONAL_PRECOND="${BENCH_MAMS_DIAGONAL_PRECOND:-1}"
BENCH_RAYON_THREADS="${BENCH_RAYON_THREADS:-${BENCH_CHAINS}}"

REMOTE_SPEC="${BENCH_HOST}"
if [[ -n "${BENCH_SSH_USER}" ]]; then
  REMOTE_SPEC="${BENCH_SSH_USER}@${BENCH_HOST}"
fi

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new)
RSYNC_RSH=(ssh -o StrictHostKeyChecking=accept-new)
if [[ -n "${BENCH_SSH_KEY}" ]]; then
  SSH_BASE+=(-i "${BENCH_SSH_KEY}")
  RSYNC_RSH+=(-i "${BENCH_SSH_KEY}")
fi
if [[ -n "${BENCH_SSH_PORT}" ]]; then
  SSH_BASE+=(-p "${BENCH_SSH_PORT}")
  RSYNC_RSH+=(-p "${BENCH_SSH_PORT}")
fi
SSH_BASE+=("${REMOTE_SPEC}")

mkdir -p "${BENCH_LOCAL_OUT}"

echo "[sampler-matrix-remote] host=${REMOTE_SPEC}"
echo "[sampler-matrix-remote] script=${BENCH_SCRIPT}"
echo "[sampler-matrix-remote] remote_repo=${BENCH_REMOTE_REPO}"
echo "[sampler-matrix-remote] remote_venv=${BENCH_REMOTE_VENV}"
echo "[sampler-matrix-remote] remote_out=${BENCH_REMOTE_OUT}"
echo "[sampler-matrix-remote] remote_target=${BENCH_REMOTE_TARGET}"
echo "[sampler-matrix-remote] local_out=${BENCH_LOCAL_OUT}"
echo "[sampler-matrix-remote] skip_build=${BENCH_SKIP_BUILD}"
echo "[sampler-matrix-remote] metric=${BENCH_METRIC}"

echo "[sampler-matrix-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version"

echo "[sampler-matrix-remote] create remote directories..."
"${SSH_BASE[@]}" "mkdir -p '${BENCH_REMOTE_REPO}' '${BENCH_REMOTE_OUT}'"

echo "[sampler-matrix-remote] rsync snapshot..."
RSYNC_RSH_CMD="${RSYNC_RSH[*]}"
rsync -az \
  --rsh="${RSYNC_RSH_CMD}" \
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
  "${ROOT_DIR}/" \
  "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/"

if [[ "${BENCH_SKIP_BUILD}" != "1" ]]; then
  echo "[sampler-matrix-remote] create venv and install build toolchain..."
  "${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_REPO}" "${BENCH_REMOTE_VENV}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
cd "$REPO/bindings/ns-py"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools maturin >/dev/null
EOS

  echo "[sampler-matrix-remote] build local nextstat into venv..."
  "${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_REPO}" "${BENCH_REMOTE_VENV}" "${BENCH_REMOTE_TARGET}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
cd "$REPO/bindings/ns-py"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
"$VENV/bin/python" -m maturin develop --release --pip-path "$VENV/bin/pip"
EOS
else
  echo "[sampler-matrix-remote] skipping build; reusing existing repo/venv/target"
fi

echo "[sampler-matrix-remote] run benchmark..."
"${SSH_BASE[@]}" bash -s -- \
  "${BENCH_REMOTE_REPO}" \
  "${BENCH_REMOTE_VENV}" \
  "${BENCH_REMOTE_OUT}" \
  "${BENCH_REMOTE_TARGET}" \
  "${BENCH_SCRIPT}" \
  "${BENCH_SEEDS}" \
  "${BENCH_MODELS}" \
  "${BENCH_METHODS}" \
  "${BENCH_CHAINS}" \
  "${BENCH_WARMUP}" \
  "${BENCH_SAMPLES}" \
  "${BENCH_GLM_N}" \
  "${BENCH_GLM_P}" \
  "${BENCH_METRIC}" \
  "${BENCH_TARGET_ACCEPT}" \
  "${BENCH_MAX_TREEDEPTH}" \
  "${BENCH_MAMS_MAX_LEAPFROG}" \
  "${BENCH_MAMS_DIAGONAL_PRECOND}" \
  "${BENCH_RAYON_THREADS}" \
  "${LOCAL_GIT_COMMIT}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
OUT="$3"
TARGET="$4"
SCRIPT_PATH="$5"
SEEDS="$6"
MODELS="$7"
METHODS="$8"
CHAINS="$9"
WARMUP="${10}"
SAMPLES="${11}"
GLM_N="${12}"
GLM_P="${13}"
METRIC="${14}"
TARGET_ACCEPT="${15}"
MAX_TREEDEPTH="${16}"
MAMS_MAX_LEAPFROG="${17}"
MAMS_DIAGONAL_PRECOND="${18}"
RAYON_THREADS="${19}"
GIT_COMMIT="${20}"
cd "$REPO"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS="$RAYON_THREADS"
export NEXTSTAT_BENCH_GIT_COMMIT="$GIT_COMMIT"
export NEXTSTAT_BENCH_HOST_POLICY="${BENCH_HOST_POLICY:-${BENCH_HOST}}"
export NEXTSTAT_BENCH_SUBMIT_HOST="${BENCH_HOST}"
export NEXTSTAT_BENCH_EXECUTE_HOST="${BENCH_HOST}"
export NEXTSTAT_BENCH_SCHEDULER="${BENCH_SCHEDULER:-ssh}"
CMD=(
  "$VENV/bin/python" "$SCRIPT_PATH"
  --out-dir "$OUT"
  --seeds "$SEEDS"
  --models "$MODELS"
  --methods "$METHODS"
  --n-chains "$CHAINS"
  --n-warmup "$WARMUP"
  --n-samples "$SAMPLES"
  --glm-n "$GLM_N"
  --glm-p "$GLM_P"
  --metric "$METRIC"
  --target-accept "$TARGET_ACCEPT"
  --max-treedepth "$MAX_TREEDEPTH"
  --mams-max-leapfrog "$MAMS_MAX_LEAPFROG"
)
if [[ "$MAMS_DIAGONAL_PRECOND" == "1" ]]; then
  CMD+=(--mams-diagonal-precond)
else
  CMD+=(--no-mams-diagonal-precond)
fi
"${CMD[@]}"
EOS

echo "[sampler-matrix-remote] sync artifacts..."
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_OUT}/" "${BENCH_LOCAL_OUT}/"

echo "[sampler-matrix-remote] done: ${BENCH_LOCAL_OUT}"
