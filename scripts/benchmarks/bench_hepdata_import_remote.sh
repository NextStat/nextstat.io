#!/usr/bin/env bash
set -euo pipefail

# Canonical remote runner for the HEPData import runtime gate on nextstat-bench.
#
# It rsyncs the current working tree to a temporary remote checkout, runs the
# deterministic HEPData import benchmark gate there, and syncs the resulting
# artifacts back locally.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

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

BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/hepdata_import_bench_repo_${STAMP}}"
BENCH_REMOTE_OUT_DIR="${BENCH_REMOTE_OUT_DIR:-/tmp/hepdata_import_bench_${STAMP}_${BENCH_HOST}}"
BENCH_REMOTE_OUT_JSON="${BENCH_REMOTE_OUT_JSON:-${BENCH_REMOTE_OUT_DIR}/summary.json}"
BENCH_REMOTE_WORK_ROOT="${BENCH_REMOTE_WORK_ROOT:-${BENCH_REMOTE_OUT_DIR}/work}"
BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/hepdata_import_bench_target_${STAMP}}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-${ROOT_DIR}/tmp/hepdata_import_benchmark_${STAMP}/${BENCH_HOST}}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

mkdir -p "${BENCH_LOCAL_OUT}"

echo "[hepdata-import-remote] host=${REMOTE_SPEC}"
echo "[hepdata-import-remote] remote_repo=${BENCH_REMOTE_REPO}"
echo "[hepdata-import-remote] remote_out=${BENCH_REMOTE_OUT_JSON}"
echo "[hepdata-import-remote] remote_work_root=${BENCH_REMOTE_WORK_ROOT}"
echo "[hepdata-import-remote] remote_target=${BENCH_REMOTE_TARGET}"
echo "[hepdata-import-remote] local_out=${BENCH_LOCAL_OUT}"

echo "[hepdata-import-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version; cargo --version"

echo "[hepdata-import-remote] create remote directories..."
"${SSH_BASE[@]}" "mkdir -p '${BENCH_REMOTE_REPO}' '${BENCH_REMOTE_OUT_DIR}'"

echo "[hepdata-import-remote] rsync snapshot..."
RSYNC_RSH_CMD="${RSYNC_RSH[*]}"
rsync -az \
  --rsh="${RSYNC_RSH_CMD}" \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude '.nextstat-cargo-target/' \
  --exclude 'node_modules/' \
  --exclude 'benchmarks/artifacts/' \
  --exclude 'benchmarks/unbinned/artifacts/' \
  --exclude 'bench_results/' \
  --exclude '.venv*/' \
  --exclude 'tmp/' \
  --exclude 'tmp*/' \
  --exclude '**/__pycache__/' \
  --exclude '.DS_Store' \
  "${ROOT_DIR}/" \
  "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/"

echo "[hepdata-import-remote] run remote benchmark gate..."
"${SSH_BASE[@]}" bash -s -- \
  "${BENCH_REMOTE_REPO}" \
  "${BENCH_REMOTE_OUT_JSON}" \
  "${BENCH_REMOTE_WORK_ROOT}" \
  "${BENCH_REMOTE_TARGET}" \
  "${BENCH_EXTRA_ARGS}" <<'EOS'
set -euo pipefail
REPO="$1"
OUT_JSON="$2"
WORK_ROOT="$3"
TARGET="$4"
EXTRA_ARGS="${5-}"
cd "$REPO"
export NEXTSTAT_HEPDATA_BENCH_CARGO_TARGET_DIR="$TARGET"
python3 scripts/benchmarks/bench_hepdata_import.py \
  --deterministic \
  --out "$OUT_JSON" \
  --work-root "$WORK_ROOT" \
  $EXTRA_ARGS
EOS

echo "[hepdata-import-remote] sync artifacts..."
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_OUT_DIR}/" "${BENCH_LOCAL_OUT}/"

echo "[hepdata-import-remote] done: ${BENCH_LOCAL_OUT}"
