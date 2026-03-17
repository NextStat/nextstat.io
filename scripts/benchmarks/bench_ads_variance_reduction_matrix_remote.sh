#!/usr/bin/env bash
set -euo pipefail

# Canonical remote runner for the ads variance-reduction matrix benchmark on nextstat-bench.

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

REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/ads_variance_reduction_repo_${STAMP}}"
REMOTE_VENV="${BENCH_REMOTE_VENV:-/tmp/ads_variance_reduction_venv_${STAMP}}"
REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/ads_variance_reduction_target_${STAMP}}"
REMOTE_OUT_DIR="${BENCH_REMOTE_OUT_DIR:-/tmp/ads_variance_reduction_benchmark_${STAMP}_${BENCH_HOST}}"
REMOTE_OUT_JSON="${BENCH_REMOTE_OUT_JSON:-${REMOTE_OUT_DIR}/ads_variance_reduction_benchmark.json}"
REMOTE_OUT_MD="${BENCH_REMOTE_OUT_MD:-${REMOTE_OUT_DIR}/ads_variance_reduction_benchmark.md}"
REMOTE_WORK_ROOT="${BENCH_REMOTE_WORK_ROOT:-${REMOTE_OUT_DIR}/work}"
LOCAL_OUT_DIR="${BENCH_LOCAL_OUT_DIR:-${ROOT_DIR}/tmp/ads_variance_reduction_benchmark_${STAMP}/${BENCH_HOST}}"

BENCH_SMOKE="${BENCH_SMOKE:-0}"
BENCH_RUNS="${BENCH_RUNS:-5}"
BENCH_WARMUPS="${BENCH_WARMUPS:-1}"
BENCH_SKIP_BUILD="${BENCH_SKIP_BUILD:-0}"
BENCH_SCENARIO_MANIFEST="${BENCH_SCENARIO_MANIFEST:-tests/fixtures/variance_reduction_benchmark/scenario_matrix.json}"

mkdir -p "${LOCAL_OUT_DIR}"

echo "[ads-vr-remote] host=${REMOTE_SPEC}"
echo "[ads-vr-remote] remote_repo=${REMOTE_REPO}"
echo "[ads-vr-remote] remote_venv=${REMOTE_VENV}"
echo "[ads-vr-remote] remote_target=${REMOTE_TARGET}"
echo "[ads-vr-remote] remote_out=${REMOTE_OUT_JSON}"
echo "[ads-vr-remote] local_out=${LOCAL_OUT_DIR}"
echo "[ads-vr-remote] scenario_manifest=${BENCH_SCENARIO_MANIFEST}"
echo "[ads-vr-remote] skip_build=${BENCH_SKIP_BUILD}"

echo "[ads-vr-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version; cargo --version"

echo "[ads-vr-remote] create remote directories..."
"${SSH_BASE[@]}" "mkdir -p '${REMOTE_REPO}' '${REMOTE_OUT_DIR}'"

echo "[ads-vr-remote] rsync snapshot..."
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
  "${REMOTE_SPEC}:${REMOTE_REPO}/"

if [[ "${BENCH_SKIP_BUILD}" != "1" ]]; then
  echo "[ads-vr-remote] create venv..."
  "${SSH_BASE[@]}" bash -s -- "${REMOTE_VENV}" <<'EOS'
set -euo pipefail
VENV="$1"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools maturin pytest jsonschema >/dev/null
EOS

  echo "[ads-vr-remote] build local nextstat into remote venv..."
  "${SSH_BASE[@]}" bash -s -- "${REMOTE_REPO}" "${REMOTE_VENV}" "${REMOTE_TARGET}" <<'EOS'
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

  echo "[ads-vr-remote] build release ns-cli..."
  "${SSH_BASE[@]}" bash -s -- "${REMOTE_REPO}" "${REMOTE_TARGET}" <<'EOS'
set -euo pipefail
REPO="$1"
TARGET="$2"
cd "$REPO"
export CARGO_TARGET_DIR="$TARGET"
cargo build -p ns-cli --release
EOS
else
  echo "[ads-vr-remote] skipping remote builds; reusing existing venv/target"
fi

echo "[ads-vr-remote] run benchmark..."
"${SSH_BASE[@]}" bash -s -- \
  "${REMOTE_REPO}" \
  "${REMOTE_VENV}" \
  "${REMOTE_TARGET}" \
  "${REMOTE_OUT_JSON}" \
  "${REMOTE_OUT_MD}" \
  "${REMOTE_WORK_ROOT}" \
  "${BENCH_SMOKE}" \
  "${BENCH_RUNS}" \
  "${BENCH_WARMUPS}" \
  "${LOCAL_GIT_COMMIT}" \
  "${BENCH_SCENARIO_MANIFEST}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
OUT_JSON="$4"
OUT_MD="$5"
WORK_ROOT="$6"
SMOKE="$7"
RUNS="$8"
WARMUPS="$9"
GIT_COMMIT="${10}"
SCENARIO_MANIFEST="${11}"
cd "$REPO"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
export NEXTSTAT_BENCH_GIT_COMMIT="$GIT_COMMIT"
CMD=(
  "$VENV/bin/python"
  "scripts/benchmarks/bench_ads_variance_reduction_matrix.py"
  "--nextstat-bin"
  "${TARGET}/release/nextstat"
  "--scenario-manifest"
  "$SCENARIO_MANIFEST"
  "--out"
  "$OUT_JSON"
  "--markdown-out"
  "$OUT_MD"
  "--work-root"
  "$WORK_ROOT"
  "--deterministic"
)
if [[ "$SMOKE" == "1" ]]; then
  CMD+=(--smoke)
else
  CMD+=(--runs "$RUNS" --warmups "$WARMUPS")
fi
"${CMD[@]}"
EOS

echo "[ads-vr-remote] sync artifacts..."
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${REMOTE_OUT_DIR}/" "${LOCAL_OUT_DIR}/"

echo "[ads-vr-remote] done: ${LOCAL_OUT_DIR}"
