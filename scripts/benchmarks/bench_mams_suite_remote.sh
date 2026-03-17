#!/usr/bin/env bash
set -euo pipefail

# Canonical remote runner for the tracked public MAMS suite on nextstat-bench.
#
# It rsyncs the current working tree snapshot to the remote host, builds the
# local ns-py bindings into an isolated venv, runs the canonical tracked MAMS
# suite/report/validation flow, and syncs the resulting artifacts back locally.

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

BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/nextstat_mams_suite_repo_${STAMP}}"
BENCH_REMOTE_VENV="${BENCH_REMOTE_VENV:-/tmp/nextstat_mams_suite_venv_${STAMP}}"
BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/nextstat_mams_suite_target_${STAMP}}"
BENCH_REMOTE_OUT="${BENCH_REMOTE_OUT:-/tmp/nextstat_mams_suite_${STAMP}_${BENCH_HOST}}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-$ROOT_DIR/benchmarks/artifacts/mams_stable_surface_validation_${STAMP}/${BENCH_HOST}}"

BENCH_SEEDS="${BENCH_SEEDS:-42}"
BENCH_BACKENDS="${BENCH_BACKENDS:-nextstat_mams,nextstat_nuts}"
BENCH_CHAINS="${BENCH_CHAINS:-4}"
BENCH_WARMUP="${BENCH_WARMUP:-3500}"
BENCH_SAMPLES="${BENCH_SAMPLES:-2000}"
BENCH_TARGET_ACCEPT="${BENCH_TARGET_ACCEPT:-0.985}"
BENCH_RUN_TIMEOUT_S="${BENCH_RUN_TIMEOUT_S:-300}"
BENCH_PARITY_WARN_Z="${BENCH_PARITY_WARN_Z:-8}"
BENCH_PARITY_FAIL_Z="${BENCH_PARITY_FAIL_Z:-12}"
BENCH_SMOKE="${BENCH_SMOKE:-0}"
BENCH_DETERMINISTIC="${BENCH_DETERMINISTIC:-1}"

mkdir -p "${BENCH_LOCAL_OUT}"

echo "[mams-suite-remote] host=${REMOTE_SPEC}"
echo "[mams-suite-remote] remote_repo=${BENCH_REMOTE_REPO}"
echo "[mams-suite-remote] remote_venv=${BENCH_REMOTE_VENV}"
echo "[mams-suite-remote] remote_target=${BENCH_REMOTE_TARGET}"
echo "[mams-suite-remote] remote_out=${BENCH_REMOTE_OUT}"
echo "[mams-suite-remote] local_out=${BENCH_LOCAL_OUT}"
echo "[mams-suite-remote] seeds=${BENCH_SEEDS}"
echo "[mams-suite-remote] backends=${BENCH_BACKENDS}"

echo "[mams-suite-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version; cargo --version"

echo "[mams-suite-remote] create remote directories..."
"${SSH_BASE[@]}" "mkdir -p '${BENCH_REMOTE_REPO}' '${BENCH_REMOTE_REPO}/benchmarks' '${BENCH_REMOTE_REPO}/bindings' '${BENCH_REMOTE_REPO}/crates' '${BENCH_REMOTE_OUT}'"

echo "[mams-suite-remote] rsync snapshot..."
RSYNC_RSH_CMD="${RSYNC_RSH[*]}"
set +e
rsync -az \
  --rsh="${RSYNC_RSH_CMD}" \
  "${ROOT_DIR}/Cargo.toml" \
  "${ROOT_DIR}/Cargo.lock" \
  "${ROOT_DIR}/rust-toolchain.toml" \
  "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/"
rsync_rc=$?
if [[ "${rsync_rc}" -eq 0 ]]; then
  rsync -az \
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
  rsync -az \
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
  rsync -az \
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
  echo "[mams-suite-remote] rsync reported vanished source files from unrelated concurrent work; continuing with the synchronized snapshot"
fi

echo "[mams-suite-remote] create remote venv..."
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_VENV}" <<'EOS'
set -euo pipefail
VENV="$1"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools maturin jsonschema >/dev/null
EOS

echo "[mams-suite-remote] build local nextstat into remote venv..."
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_REPO}" "${BENCH_REMOTE_VENV}" "${BENCH_REMOTE_TARGET}" <<'EOS'
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

echo "[mams-suite-remote] run canonical suite..."
"${SSH_BASE[@]}" bash -s -- \
  "${BENCH_REMOTE_REPO}" \
  "${BENCH_REMOTE_VENV}" \
  "${BENCH_REMOTE_TARGET}" \
  "${BENCH_REMOTE_OUT}" \
  "${BENCH_SEEDS}" \
  "${BENCH_BACKENDS}" \
  "${BENCH_CHAINS}" \
  "${BENCH_WARMUP}" \
  "${BENCH_SAMPLES}" \
  "${BENCH_TARGET_ACCEPT}" \
  "${BENCH_RUN_TIMEOUT_S}" \
  "${BENCH_PARITY_WARN_Z}" \
  "${BENCH_PARITY_FAIL_Z}" \
  "${BENCH_SMOKE}" \
  "${BENCH_DETERMINISTIC}" \
  "${LOCAL_GIT_COMMIT}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
OUT="$4"
SEEDS="$5"
BACKENDS="$6"
CHAINS="$7"
WARMUP="$8"
SAMPLES="$9"
TARGET_ACCEPT="${10}"
RUN_TIMEOUT_S="${11}"
PARITY_WARN_Z="${12}"
PARITY_FAIL_Z="${13}"
SMOKE="${14}"
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

SUITE_CMD=(
  "$VENV/bin/python"
  "suites/mams/suite.py"
  "--out-dir" "$OUT"
  "--backends" "$BACKENDS"
  "--seeds" "$SEEDS"
  "--n-chains" "$CHAINS"
  "--warmup" "$WARMUP"
  "--samples" "$SAMPLES"
  "--target-accept" "$TARGET_ACCEPT"
  "--run-timeout-s" "$RUN_TIMEOUT_S"
  "--parity-warn-z" "$PARITY_WARN_Z"
  "--parity-fail-z" "$PARITY_FAIL_Z"
)
if [[ "$DETERMINISTIC" == "1" ]]; then
  SUITE_CMD+=(--deterministic)
fi
if [[ "$SMOKE" == "1" ]]; then
  SUITE_CMD+=(--smoke)
fi
"${SUITE_CMD[@]}"

"$VENV/bin/python" "suites/mams/report.py" "$OUT" >"$OUT/mams_benchmark_report.stdout.md"
"$VENV/bin/python" "suites/mams/assess.py" "$OUT"
"$VENV/bin/python" "scripts/validate_artifacts.py" --strict "$OUT"
EOS

echo "[mams-suite-remote] sync artifacts..."
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_OUT}/" "${BENCH_LOCAL_OUT}/"

echo "[mams-suite-remote] done: ${BENCH_LOCAL_OUT}"
