#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the generic sampler-matrix remote runner.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

export BENCH_SCRIPT="${BENCH_SCRIPT:-scripts/benchmarks/bench_walnuts_vs_nuts.py}"
export BENCH_METHODS="${BENCH_METHODS:-nuts,walnuts}"
export BENCH_MODELS="${BENCH_MODELS:-std_normal_10d,eight_schools,glm_logistic,funnel_ncp_10d,glm_negbin}"
export BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/nextstat_walnuts_bench_repo_${STAMP}}"
export BENCH_REMOTE_VENV="${BENCH_REMOTE_VENV:-/tmp/nextstat_walnuts_bench_venv_${STAMP}}"
export BENCH_REMOTE_OUT="${BENCH_REMOTE_OUT:-/tmp/bench_walnuts_vs_nuts_${STAMP}_${BENCH_HOST:-nextstat-bench}}"
export BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/nextstat_walnuts_bench_target_${STAMP}}"
export BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-$ROOT_DIR/tmp/bench_walnuts_vs_nuts_${STAMP}/${BENCH_HOST:-nextstat-bench}}"

exec bash "${ROOT_DIR}/scripts/benchmarks/bench_sampler_matrix_remote.sh"
