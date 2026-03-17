#!/usr/bin/env bash
set -euo pipefail

# Canonical remote runner for the MAMS publish_snapshot lane on nextstat-bench.
#
# It rsyncs the current working tree snapshot to the remote host, builds the
# local ns-py bindings into an isolated venv, runs publish_snapshot.py for the
# MAMS suite, validates the produced publish root, and syncs the publish root
# back locally so the returned bundle includes snapshot_registry.json.

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

BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/nextstat_mams_publisher_repo_${STAMP}}"
BENCH_REMOTE_VENV="${BENCH_REMOTE_VENV:-/tmp/nextstat_mams_publisher_venv_${STAMP}}"
BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/nextstat_mams_publisher_target_${STAMP}}"
BENCH_REMOTE_SNAPSHOT_ROOT="${BENCH_REMOTE_SNAPSHOT_ROOT:-/tmp/nextstat_mams_publisher_snapshots_${STAMP}}"
BENCH_SNAPSHOT_ID="${BENCH_SNAPSHOT_ID:-mams-publisher-${STAMP}}"
BENCH_REMOTE_OUT="${BENCH_REMOTE_SNAPSHOT_ROOT}/${BENCH_SNAPSHOT_ID}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-$ROOT_DIR/benchmarks/artifacts/mams_publisher_validation_${STAMP}/${BENCH_HOST}/${BENCH_SNAPSHOT_ID}}"
BENCH_LOCAL_ROOT="$(dirname "${BENCH_LOCAL_OUT}")"
BENCH_REMOTE_REGISTRY="${BENCH_REMOTE_SNAPSHOT_ROOT}/snapshot_registry.json"
BENCH_LOCAL_REGISTRY="${BENCH_LOCAL_ROOT}/snapshot_registry.json"
BENCH_SMOKE="${BENCH_SMOKE:-0}"
BENCH_DETERMINISTIC="${BENCH_DETERMINISTIC:-1}"

mkdir -p "${BENCH_LOCAL_OUT}"
mkdir -p "${BENCH_LOCAL_ROOT}"

echo "[mams-publisher-remote] host=${REMOTE_SPEC}"
echo "[mams-publisher-remote] remote_repo=${BENCH_REMOTE_REPO}"
echo "[mams-publisher-remote] remote_venv=${BENCH_REMOTE_VENV}"
echo "[mams-publisher-remote] remote_target=${BENCH_REMOTE_TARGET}"
echo "[mams-publisher-remote] remote_snapshot_root=${BENCH_REMOTE_SNAPSHOT_ROOT}"
echo "[mams-publisher-remote] remote_registry=${BENCH_REMOTE_REGISTRY}"
echo "[mams-publisher-remote] snapshot_id=${BENCH_SNAPSHOT_ID}"
echo "[mams-publisher-remote] local_out=${BENCH_LOCAL_OUT}"
echo "[mams-publisher-remote] local_registry=${BENCH_LOCAL_REGISTRY}"

echo "[mams-publisher-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version; cargo --version"

echo "[mams-publisher-remote] create remote directories..."
"${SSH_BASE[@]}" "mkdir -p '${BENCH_REMOTE_REPO}' '${BENCH_REMOTE_REPO}/benchmarks' '${BENCH_REMOTE_REPO}/bindings' '${BENCH_REMOTE_REPO}/crates' '${BENCH_REMOTE_SNAPSHOT_ROOT}'"

echo "[mams-publisher-remote] rsync snapshot..."
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
  echo "[mams-publisher-remote] rsync reported vanished source files from unrelated concurrent work; continuing with the synchronized snapshot"
fi

echo "[mams-publisher-remote] create remote venv..."
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_VENV}" <<'EOS'
set -euo pipefail
VENV="$1"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools maturin jsonschema >/dev/null
EOS

echo "[mams-publisher-remote] build local nextstat into remote venv..."
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

echo "[mams-publisher-remote] run publish_snapshot.py..."
"${SSH_BASE[@]}" bash -s -- \
  "${BENCH_REMOTE_REPO}" \
  "${BENCH_REMOTE_VENV}" \
  "${BENCH_REMOTE_TARGET}" \
  "${BENCH_REMOTE_SNAPSHOT_ROOT}" \
  "${BENCH_SNAPSHOT_ID}" \
  "${BENCH_SMOKE}" \
  "${BENCH_DETERMINISTIC}" \
  "${LOCAL_GIT_COMMIT}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
SNAPSHOT_ROOT="$4"
SNAPSHOT_ID="$5"
SMOKE="$6"
DETERMINISTIC="$7"
GIT_COMMIT="$8"

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

PUBLISH_CMD=(
  "$VENV/bin/python"
  "scripts/publish_snapshot.py"
  "--snapshot-id" "$SNAPSHOT_ID"
  "--out-root" "$SNAPSHOT_ROOT"
  "--mams"
)
if [[ "$DETERMINISTIC" == "1" ]]; then
  PUBLISH_CMD+=(--deterministic)
fi
if [[ "$SMOKE" == "1" ]]; then
  PUBLISH_CMD+=(--smoke)
fi
"${PUBLISH_CMD[@]}"

"$VENV/bin/python" "scripts/validate_artifacts.py" --strict "$SNAPSHOT_ROOT"
EOS

echo "[mams-publisher-remote] sync snapshot root..."
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_SNAPSHOT_ROOT}/" "${BENCH_LOCAL_ROOT}/"

echo "[mams-publisher-remote] done: ${BENCH_LOCAL_ROOT}"
