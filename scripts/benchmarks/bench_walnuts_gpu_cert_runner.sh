#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BENCH_PROFILE=cert
exec bash "${ROOT_DIR}/scripts/benchmarks/bench_walnuts_gpu_transfer_smoke.sh" "$@"
