#!/usr/bin/env bash
set -euo pipefail

# Canonical remote runner for MAMS multi-seed repeatability evidence on nextstat-bench.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"

BENCH_HOST="${BENCH_HOST:-nextstat-bench}"
BENCH_SSH_USER="${BENCH_SSH_USER:-}"
BENCH_SSH_PORT="${BENCH_SSH_PORT:-}"
BENCH_SSH_KEY="${BENCH_SSH_KEY:-}"

REMOTE_SPEC="${BENCH_HOST}"
if [[ -n "${BENCH_SSH_USER}" ]]; then
  REMOTE_SPEC="${BENCH_SSH_USER}@${BENCH_HOST}"
fi

SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectionAttempts=3)
RSYNC_RSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectionAttempts=3)
if [[ -n "${BENCH_SSH_KEY}" ]]; then
  SSH_BASE+=(-i "${BENCH_SSH_KEY}")
  RSYNC_RSH+=(-i "${BENCH_SSH_KEY}")
fi
if [[ -n "${BENCH_SSH_PORT}" ]]; then
  SSH_BASE+=(-p "${BENCH_SSH_PORT}")
  RSYNC_RSH+=(-p "${BENCH_SSH_PORT}")
fi
SSH_BASE+=("${REMOTE_SPEC}")

BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/nextstat_mams_multiseed_repo_${STAMP}}"
BENCH_REMOTE_VENV="${BENCH_REMOTE_VENV:-/tmp/nextstat_mams_multiseed_venv_${STAMP}}"
BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/nextstat_mams_multiseed_target_${STAMP}}"
BENCH_REMOTE_OUT="${BENCH_REMOTE_OUT:-/tmp/nextstat_mams_multiseed_${STAMP}_${BENCH_HOST}}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-$ROOT_DIR/benchmarks/artifacts/mams_multiseed_validation_${STAMP}/${BENCH_HOST}}"

BENCH_BACKENDS="${BENCH_BACKENDS:-nextstat_mams,nextstat_nuts}"
BENCH_CHAINS="${BENCH_CHAINS:-4}"
BENCH_WARMUP="${BENCH_WARMUP:-3500}"
BENCH_SAMPLES="${BENCH_SAMPLES:-2000}"
BENCH_SEEDS="${BENCH_SEEDS:-42,0,123}"
BENCH_DATASET_SEED="${BENCH_DATASET_SEED:-12345}"
BENCH_TARGET_ACCEPT="${BENCH_TARGET_ACCEPT:-0.985}"
BENCH_RUN_TIMEOUT_S="${BENCH_RUN_TIMEOUT_S:-300}"
BENCH_PARITY_WARN_Z="${BENCH_PARITY_WARN_Z:-8}"
BENCH_PARITY_FAIL_Z="${BENCH_PARITY_FAIL_Z:-12}"
BENCH_DETERMINISTIC="${BENCH_DETERMINISTIC:-1}"
BENCH_SSH_RETRIES="${BENCH_SSH_RETRIES:-3}"
BENCH_RSYNC_RETRIES="${BENCH_RSYNC_RETRIES:-3}"
BENCH_RETRY_SLEEP_S="${BENCH_RETRY_SLEEP_S:-2}"

ssh_run() {
  local attempt=1
  local rc=0
  while (( attempt <= BENCH_SSH_RETRIES )); do
    set +e
    "${SSH_BASE[@]}" "$@"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
    if [[ "${attempt}" -lt BENCH_SSH_RETRIES && "${rc}" -eq 255 ]]; then
      echo "[mams-multiseed-remote] ssh attempt ${attempt}/${BENCH_SSH_RETRIES} failed with rc=${rc}; retrying in ${BENCH_RETRY_SLEEP_S}s..."
      sleep "${BENCH_RETRY_SLEEP_S}"
      attempt=$((attempt + 1))
      continue
    fi
    return "${rc}"
  done
  return "${rc}"
}

rsync_run() {
  local attempt=1
  local rc=0
  while (( attempt <= BENCH_RSYNC_RETRIES )); do
    set +e
    rsync "$@"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 || "${rc}" -eq 24 ]]; then
      return "${rc}"
    fi
    if [[ "${attempt}" -lt BENCH_RSYNC_RETRIES && ( "${rc}" -eq 12 || "${rc}" -eq 20 || "${rc}" -eq 30 || "${rc}" -eq 35 || "${rc}" -eq 255 ) ]]; then
      echo "[mams-multiseed-remote] rsync attempt ${attempt}/${BENCH_RSYNC_RETRIES} failed with rc=${rc}; retrying in ${BENCH_RETRY_SLEEP_S}s..."
      sleep "${BENCH_RETRY_SLEEP_S}"
      attempt=$((attempt + 1))
      continue
    fi
    return "${rc}"
  done
  return "${rc}"
}

mkdir -p "${BENCH_LOCAL_OUT}"

echo "[mams-multiseed-remote] host=${REMOTE_SPEC}"
echo "[mams-multiseed-remote] remote_repo=${BENCH_REMOTE_REPO}"
echo "[mams-multiseed-remote] remote_venv=${BENCH_REMOTE_VENV}"
echo "[mams-multiseed-remote] remote_target=${BENCH_REMOTE_TARGET}"
echo "[mams-multiseed-remote] remote_out=${BENCH_REMOTE_OUT}"
echo "[mams-multiseed-remote] local_out=${BENCH_LOCAL_OUT}"
echo "[mams-multiseed-remote] backends=${BENCH_BACKENDS}"
echo "[mams-multiseed-remote] seeds=${BENCH_SEEDS}"
echo "[mams-multiseed-remote] dataset_seed=${BENCH_DATASET_SEED}"

echo "[mams-multiseed-remote] probe remote..."
ssh_run "set -euo pipefail; hostname; uname -a; python3 --version; cargo --version"

echo "[mams-multiseed-remote] create remote directories..."
ssh_run "mkdir -p '${BENCH_REMOTE_REPO}' '${BENCH_REMOTE_REPO}/benchmarks' '${BENCH_REMOTE_REPO}/bindings' '${BENCH_REMOTE_REPO}/crates' '${BENCH_REMOTE_OUT}'"

echo "[mams-multiseed-remote] rsync snapshot..."
RSYNC_RSH_CMD="${RSYNC_RSH[*]}"
set +e
rsync_run -az \
  --rsh="${RSYNC_RSH_CMD}" \
  "${ROOT_DIR}/Cargo.toml" \
  "${ROOT_DIR}/Cargo.lock" \
  "${ROOT_DIR}/rust-toolchain.toml" \
  "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/"
rsync_rc=$?
if [[ "${rsync_rc}" -eq 0 ]]; then
  rsync_run -az \
    --rsh="${RSYNC_RSH_CMD}" \
    --exclude '.venv*/' \
    --exclude 'target/' \
    --exclude '.nextstat-cargo-target/' \
    --exclude '**/__pycache__/' \
    --exclude '.DS_Store' \
    "${ROOT_DIR}/bindings/" \
    "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/bindings/"
  rsync_rc=$?
fi
if [[ "${rsync_rc}" -eq 0 ]]; then
  rsync_run -az \
    --rsh="${RSYNC_RSH_CMD}" \
    --exclude 'target/' \
    --exclude '.nextstat-cargo-target/' \
    --exclude '**/__pycache__/' \
    --exclude '.DS_Store' \
    "${ROOT_DIR}/crates/" \
    "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/crates/"
  rsync_rc=$?
fi
if [[ "${rsync_rc}" -eq 0 ]]; then
  rsync_run -az \
    --rsh="${RSYNC_RSH_CMD}" \
    --exclude '.venv/' \
    --exclude 'out/' \
    --exclude '**/__pycache__/' \
    --exclude '.DS_Store' \
    "${ROOT_DIR}/benchmarks/nextstat-public-benchmarks/" \
    "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/benchmarks/nextstat-public-benchmarks/"
  rsync_rc=$?
fi
set -e
if [[ "${rsync_rc}" -ne 0 && "${rsync_rc}" -ne 24 ]]; then
  exit "${rsync_rc}"
fi
if [[ "${rsync_rc}" -eq 24 ]]; then
  echo "[mams-multiseed-remote] rsync reported vanished source files from unrelated concurrent work; continuing with the synchronized snapshot"
fi

echo "[mams-multiseed-remote] create remote venv..."
ssh_run bash -s -- "${BENCH_REMOTE_VENV}" <<'EOS'
set -euo pipefail
VENV="$1"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools maturin jsonschema >/dev/null
EOS

echo "[mams-multiseed-remote] build local nextstat into remote venv..."
ssh_run bash -s -- "${BENCH_REMOTE_REPO}" "${BENCH_REMOTE_VENV}" "${BENCH_REMOTE_TARGET}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
cd "$REPO/bindings/ns-py"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
bash "$REPO/benchmarks/nextstat-public-benchmarks/scripts/install_local_nextstat_python.sh" "$REPO" "$VENV" "$TARGET"
EOS

echo "[mams-multiseed-remote] run canonical multiseed suite..."
ssh_run bash -s -- \
  "${BENCH_REMOTE_REPO}" \
  "${BENCH_REMOTE_VENV}" \
  "${BENCH_REMOTE_TARGET}" \
  "${BENCH_REMOTE_OUT}" \
  "${BENCH_BACKENDS}" \
  "${BENCH_CHAINS}" \
  "${BENCH_WARMUP}" \
  "${BENCH_SAMPLES}" \
  "${BENCH_SEEDS}" \
  "${BENCH_DATASET_SEED}" \
  "${BENCH_TARGET_ACCEPT}" \
  "${BENCH_RUN_TIMEOUT_S}" \
  "${BENCH_PARITY_WARN_Z}" \
  "${BENCH_PARITY_FAIL_Z}" \
  "${BENCH_DETERMINISTIC}" \
  "${LOCAL_GIT_COMMIT}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
OUT="$4"
BACKENDS="$5"
CHAINS="$6"
WARMUP="$7"
SAMPLES="$8"
SEEDS="$9"
DATASET_SEED="${10}"
TARGET_ACCEPT="${11}"
RUN_TIMEOUT_S="${12}"
PARITY_WARN_Z="${13}"
PARITY_FAIL_Z="${14}"
DETERMINISTIC="${15}"
GIT_COMMIT="${16}"

cd "$REPO/benchmarks/nextstat-public-benchmarks"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NEXTSTAT_BENCH_GIT_COMMIT="$GIT_COMMIT"

CMD=(
  "$VENV/bin/python"
  "suites/mams/multiseed.py"
  "--out-dir" "$OUT"
  "--backends" "$BACKENDS"
  "--n-chains" "$CHAINS"
  "--warmup" "$WARMUP"
  "--samples" "$SAMPLES"
  "--seeds" "$SEEDS"
  "--dataset-seed" "$DATASET_SEED"
  "--target-accept" "$TARGET_ACCEPT"
  "--run-timeout-s" "$RUN_TIMEOUT_S"
  "--parity-warn-z" "$PARITY_WARN_Z"
  "--parity-fail-z" "$PARITY_FAIL_Z"
)
if [[ "$DETERMINISTIC" == "1" ]]; then
  CMD+=(--deterministic)
fi
"${CMD[@]}"

"$VENV/bin/python" "suites/mams/assess_multiseed.py" "$OUT"
"$VENV/bin/python" "scripts/validate_artifacts.py" --strict "$OUT"
EOS

echo "[mams-multiseed-remote] sync artifacts..."
rsync_run -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_OUT}/" "${BENCH_LOCAL_OUT}/"

echo "[mams-multiseed-remote] done: ${BENCH_LOCAL_OUT}"
