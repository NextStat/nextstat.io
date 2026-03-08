#!/usr/bin/env bash
set -euo pipefail

# Canonical remote runner for M15 reporting benchmarks on nextstat-bench.
#
# It rsyncs the current working tree to a temporary remote checkout, builds an
# isolated release ns-cli binary, runs the benchmark harness there, and syncs
# the JSON/Markdown artifacts back locally.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"

BENCH_HOST="${BENCH_HOST:-nextstat-bench}"
BENCH_SSH_USER="${BENCH_SSH_USER:-}"
BENCH_SSH_PORT="${BENCH_SSH_PORT:-}"
BENCH_SSH_KEY="${BENCH_SSH_KEY:-}"

BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/m15_reporting_bench_repo_${STAMP}}"
BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/m15_reporting_bench_target_${STAMP}}"
BENCH_REMOTE_OUT_DIR="${BENCH_REMOTE_OUT_DIR:-/tmp/m15_reporting_benchmark_${STAMP}_${BENCH_HOST}}"
BENCH_REMOTE_OUT_JSON="${BENCH_REMOTE_OUT_JSON:-${BENCH_REMOTE_OUT_DIR}/m15_reporting_benchmark.json}"
BENCH_REMOTE_OUT_MD="${BENCH_REMOTE_OUT_MD:-${BENCH_REMOTE_OUT_DIR}/m15_reporting_benchmark.md}"
BENCH_REMOTE_WORK_ROOT="${BENCH_REMOTE_WORK_ROOT:-${BENCH_REMOTE_OUT_DIR}/work}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-${ROOT_DIR}/tmp/m15_reporting_benchmark_${STAMP}/${BENCH_HOST}}"

BENCH_SMOKE="${BENCH_SMOKE:-0}"
BENCH_RUNS="${BENCH_RUNS:-5}"
BENCH_WARMUPS="${BENCH_WARMUPS:-1}"
BENCH_SKIP_BUILD="${BENCH_SKIP_BUILD:-0}"
BENCH_NEXTSTAT_BIN="${BENCH_NEXTSTAT_BIN:-}"

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

echo "[m15-bench-remote] host=${REMOTE_SPEC}"
echo "[m15-bench-remote] remote_repo=${BENCH_REMOTE_REPO}"
echo "[m15-bench-remote] remote_target=${BENCH_REMOTE_TARGET}"
echo "[m15-bench-remote] remote_out=${BENCH_REMOTE_OUT_JSON}"
echo "[m15-bench-remote] local_out=${BENCH_LOCAL_OUT}"
echo "[m15-bench-remote] skip_build=${BENCH_SKIP_BUILD}"

echo "[m15-bench-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version; cargo --version"

echo "[m15-bench-remote] create remote directories..."
"${SSH_BASE[@]}" "mkdir -p '${BENCH_REMOTE_REPO}' '${BENCH_REMOTE_OUT_DIR}'"

echo "[m15-bench-remote] rsync snapshot..."
RSYNC_RSH_CMD="${RSYNC_RSH[*]}"
rsync -az \
  --rsh="${RSYNC_RSH_CMD}" \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude '.nextstat-cargo-target/' \
  --exclude 'node_modules/' \
  --exclude 'benchmarks/artifacts/' \
  --exclude 'benchmarks/unbinned/artifacts/' \
  --exclude '.venv*/' \
  --exclude 'tmp/' \
  --exclude 'tmp*/' \
  --exclude '**/__pycache__/' \
  --exclude '.DS_Store' \
  "${ROOT_DIR}/" \
  "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/"

if [[ "${BENCH_SKIP_BUILD}" != "1" ]]; then
  echo "[m15-bench-remote] build release ns-cli..."
  "${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_REPO}" "${BENCH_REMOTE_TARGET}" <<'EOS'
set -euo pipefail
REPO="$1"
TARGET="$2"
cd "$REPO"
export CARGO_TARGET_DIR="$TARGET"
cargo build -p ns-cli --release
EOS
  BENCH_NEXTSTAT_BIN="${BENCH_REMOTE_TARGET}/release/nextstat"
fi

if [[ -z "${BENCH_NEXTSTAT_BIN}" ]]; then
  echo "BENCH_NEXTSTAT_BIN must be set when BENCH_SKIP_BUILD=1" >&2
  exit 2
fi

echo "[m15-bench-remote] run benchmark..."
"${SSH_BASE[@]}" bash -s -- \
  "${BENCH_REMOTE_REPO}" \
  "${BENCH_NEXTSTAT_BIN}" \
  "${BENCH_REMOTE_OUT_JSON}" \
  "${BENCH_REMOTE_OUT_MD}" \
  "${BENCH_REMOTE_WORK_ROOT}" \
  "${BENCH_SMOKE}" \
  "${BENCH_RUNS}" \
  "${BENCH_WARMUPS}" \
  "${LOCAL_GIT_COMMIT}" <<'EOS'
set -euo pipefail
REPO="$1"
NEXTSTAT_BIN="$2"
OUT_JSON="$3"
OUT_MD="$4"
WORK_ROOT="$5"
SMOKE="$6"
RUNS="$7"
WARMUPS="$8"
GIT_COMMIT="$9"
cd "$REPO"
export NEXTSTAT_BENCH_GIT_COMMIT="$GIT_COMMIT"
CMD=(python3 scripts/benchmarks/bench_m15_reporting.py --nextstat-bin "$NEXTSTAT_BIN" --out "$OUT_JSON" --markdown-out "$OUT_MD" --work-root "$WORK_ROOT" --deterministic)
if [[ "$SMOKE" == "1" ]]; then
  CMD+=(--smoke)
else
  CMD+=(--runs "$RUNS" --warmups "$WARMUPS")
fi
"${CMD[@]}"
EOS

echo "[m15-bench-remote] sync artifacts..."
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_OUT_DIR}/" "${BENCH_LOCAL_OUT}/"

echo "[m15-bench-remote] done: ${BENCH_LOCAL_OUT}"
