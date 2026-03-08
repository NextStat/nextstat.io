#!/usr/bin/env bash
set -euo pipefail

# Canonical remote runner for the Apex2 simplified-likelihood report on nextstat-bench.
#
# It rsyncs the current working tree to the remote bench host, builds the local
# NextStat bindings into an isolated venv, runs `tests/apex2_simplified_likelihood_report.py`,
# and syncs the resulting JSON artifact back locally.

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

REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/apex2_simplified_likelihood_repo_${STAMP}}"
REMOTE_VENV="${BENCH_REMOTE_VENV:-/tmp/apex2_simplified_likelihood_venv_${STAMP}}"
REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/apex2_simplified_likelihood_target_${STAMP}}"
REMOTE_OUT_DIR="${BENCH_REMOTE_OUT_DIR:-/tmp/apex2_simplified_likelihood_${STAMP}_${BENCH_HOST}}"
REMOTE_OUT_JSON="${BENCH_REMOTE_OUT_JSON:-${REMOTE_OUT_DIR}/apex2_simplified_likelihood_report.json}"
LOCAL_OUT_DIR="${BENCH_LOCAL_OUT_DIR:-${ROOT_DIR}/tmp/apex2_simplified_likelihood_${STAMP}/${BENCH_HOST}}"

BENCH_SUITE="${BENCH_SUITE:-bench}"
BENCH_FIT_REPEAT="${BENCH_FIT_REPEAT:-3}"
BENCH_UPPER_LIMIT_REPEAT="${BENCH_UPPER_LIMIT_REPEAT:-3}"
BENCH_EXPORT_REPEAT="${BENCH_EXPORT_REPEAT:-1}"
BENCH_SCAN="${BENCH_SCAN:-0.0,0.5,1.0,1.5,2.0,2.5,3.0}"
BENCH_LIMIT_HI="${BENCH_LIMIT_HI:-5.0}"
BENCH_LIMIT_RTOL="${BENCH_LIMIT_RTOL:-1e-4}"
BENCH_LIMIT_MAX_ITER="${BENCH_LIMIT_MAX_ITER:-80}"
BENCH_INCLUDE_PUBLIC_FIXTURES="${BENCH_INCLUDE_PUBLIC_FIXTURES:-1}"
BENCH_INCLUDE_EXPORT_MATRIX="${BENCH_INCLUDE_EXPORT_MATRIX:-1}"
BENCH_INCLUDE_EXPORT_PUBLIC_CASES="${BENCH_INCLUDE_EXPORT_PUBLIC_CASES:-1}"
BENCH_PUBLIC_FIXTURE_CATALOG="${BENCH_PUBLIC_FIXTURE_CATALOG:-docs/specs/apex2_simplified_likelihood_public_fixture_catalog_v0.example.json}"
BENCH_EXPORT_PUBLIC_CASE_CATALOG="${BENCH_EXPORT_PUBLIC_CASE_CATALOG:-docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json}"
BENCH_NEXTSTAT_CLI_BINARY="${BENCH_NEXTSTAT_CLI_BINARY:-${REMOTE_TARGET}/release/nextstat}"

mkdir -p "${LOCAL_OUT_DIR}"

echo "[sl-apex2-remote] host=${REMOTE_SPEC}"
echo "[sl-apex2-remote] remote_repo=${REMOTE_REPO}"
echo "[sl-apex2-remote] remote_venv=${REMOTE_VENV}"
echo "[sl-apex2-remote] remote_target=${REMOTE_TARGET}"
echo "[sl-apex2-remote] remote_out=${REMOTE_OUT_JSON}"
echo "[sl-apex2-remote] local_out=${LOCAL_OUT_DIR}"
echo "[sl-apex2-remote] include_public_fixtures=${BENCH_INCLUDE_PUBLIC_FIXTURES}"
echo "[sl-apex2-remote] include_export_matrix=${BENCH_INCLUDE_EXPORT_MATRIX}"
echo "[sl-apex2-remote] include_export_public_cases=${BENCH_INCLUDE_EXPORT_PUBLIC_CASES}"
echo "[sl-apex2-remote] public_fixture_catalog=${BENCH_PUBLIC_FIXTURE_CATALOG}"
echo "[sl-apex2-remote] export_public_case_catalog=${BENCH_EXPORT_PUBLIC_CASE_CATALOG}"
echo "[sl-apex2-remote] nextstat_cli_binary=${BENCH_NEXTSTAT_CLI_BINARY}"

echo "[sl-apex2-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version"

echo "[sl-apex2-remote] create remote directories..."
"${SSH_BASE[@]}" "mkdir -p '${REMOTE_REPO}' '${REMOTE_OUT_DIR}'"

echo "[sl-apex2-remote] rsync snapshot..."
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
  --exclude '**/__pycache__/' \
  --exclude '.DS_Store' \
  "${ROOT_DIR}/" \
  "${REMOTE_SPEC}:${REMOTE_REPO}/"

echo "[sl-apex2-remote] create venv..."
"${SSH_BASE[@]}" bash -s -- "${REMOTE_VENV}" <<'EOS'
set -euo pipefail
VENV="$1"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools maturin jsonschema >/dev/null
EOS

echo "[sl-apex2-remote] build local nextstat into remote venv..."
"${SSH_BASE[@]}" bash -s -- "${REMOTE_REPO}" "${REMOTE_VENV}" "${REMOTE_TARGET}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
cd "$REPO/bindings/ns-py"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
"$VENV/bin/python" -m maturin develop --release --skip-install
EOS

echo "[sl-apex2-remote] build ns-cli binary..."
"${SSH_BASE[@]}" bash -s -- "${REMOTE_REPO}" "${REMOTE_TARGET}" <<'EOS'
set -euo pipefail
REPO="$1"
TARGET="$2"
cd "$REPO"
export CARGO_TARGET_DIR="$TARGET"
cargo build -q -p ns-cli --release
EOS

echo "[sl-apex2-remote] run Apex2 report..."
"${SSH_BASE[@]}" bash -s -- \
  "${REMOTE_REPO}" \
  "${REMOTE_VENV}" \
  "${REMOTE_TARGET}" \
  "${REMOTE_OUT_JSON}" \
  "${BENCH_SUITE}" \
  "${BENCH_FIT_REPEAT}" \
  "${BENCH_UPPER_LIMIT_REPEAT}" \
  "${BENCH_EXPORT_REPEAT}" \
  "${BENCH_SCAN}" \
  "${BENCH_LIMIT_HI}" \
  "${BENCH_LIMIT_RTOL}" \
  "${BENCH_LIMIT_MAX_ITER}" \
  "${BENCH_INCLUDE_PUBLIC_FIXTURES}" \
  "${BENCH_PUBLIC_FIXTURE_CATALOG}" \
  "${BENCH_INCLUDE_EXPORT_MATRIX}" \
  "${BENCH_INCLUDE_EXPORT_PUBLIC_CASES}" \
  "${BENCH_EXPORT_PUBLIC_CASE_CATALOG}" \
  "${BENCH_NEXTSTAT_CLI_BINARY}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
OUT_JSON="$4"
SUITE="$5"
FIT_REPEAT="$6"
UL_REPEAT="$7"
EXPORT_REPEAT="$8"
SCAN="$9"
LIMIT_HI="${10}"
LIMIT_RTOL="${11}"
LIMIT_MAX_ITER="${12}"
INCLUDE_PUBLIC_FIXTURES="${13}"
PUBLIC_FIXTURE_CATALOG="${14}"
INCLUDE_EXPORT_MATRIX="${15}"
INCLUDE_EXPORT_PUBLIC_CASES="${16}"
EXPORT_PUBLIC_CASE_CATALOG="${17}"
NEXTSTAT_CLI_BINARY="${18}"
cd "$REPO"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET"
export PYTHONPATH="$REPO/bindings/ns-py/python${PYTHONPATH:+:$PYTHONPATH}"
REPORT_ARGS=(
  --suite "$SUITE"
  --fit-repeat "$FIT_REPEAT"
  --upper-limit-repeat "$UL_REPEAT"
  --export-repeat "$EXPORT_REPEAT"
  --scan "$SCAN"
  --limit-hi "$LIMIT_HI"
  --limit-rtol "$LIMIT_RTOL"
  --limit-max-iter "$LIMIT_MAX_ITER"
  --out "$OUT_JSON"
)
if [[ "$INCLUDE_PUBLIC_FIXTURES" == "1" ]]; then
  REPORT_ARGS+=(
    --include-public-fixtures
    --public-fixture-catalog "$PUBLIC_FIXTURE_CATALOG"
  )
fi
if [[ "$INCLUDE_EXPORT_MATRIX" == "1" ]]; then
  REPORT_ARGS+=(
    --include-export-matrix
    --nextstat-cli "$NEXTSTAT_CLI_BINARY"
  )
  if [[ "$INCLUDE_EXPORT_PUBLIC_CASES" == "1" ]]; then
    REPORT_ARGS+=(
      --include-export-public-cases
      --export-public-case-catalog "$EXPORT_PUBLIC_CASE_CATALOG"
    )
  fi
fi
"$VENV/bin/python" tests/apex2_simplified_likelihood_report.py "${REPORT_ARGS[@]}"
EOS

echo "[sl-apex2-remote] sync artifacts..."
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${REMOTE_OUT_DIR}/" "${LOCAL_OUT_DIR}/"

echo "[sl-apex2-remote] done: ${LOCAL_OUT_DIR}"
