#!/usr/bin/env bash
set -euo pipefail

# Apex2 Phase 0 benchmark runner (Quick Wins).
#
# Intended usage:
#   scripts/benchmarks/apex2_phase0.sh /data/apex2_phase0_20260223 "42,123,777"
#
# Notes:
# - Suites record environment snapshots per artifact via scripts/bench_env.py.
# - This script focuses on Phase 0 coverage (timeseries/glm/econometrics + LAPS GLM on GPU).

OUT_ROOT="${1:-}"
SEEDS_CSV="${2:-42,123,777}"
GPU_ONLY="${APEX2_GPU_ONLY:-0}"
if [[ "${GPU_ONLY}" == "true" ]]; then
  GPU_ONLY="1"
fi

if [[ -z "${OUT_ROOT}" ]]; then
  echo "usage: $0 OUT_ROOT [SEEDS_CSV]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/timeseries/suite.py"
GLM_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/glm/suite.py"
EC_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/econometrics/suite.py"
EC_REPORT="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/econometrics/report.py"
LAPS_RUN="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/laps_h100/run.py"
LAPS_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/laps_h100/suite.py"

# Prefer a pinned benchmark venv if present.
PY_BIN="${APEX2_PYTHON_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/.venv/bin/python" ]]; then
    PY_BIN="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/.venv/bin/python"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PY_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PY_BIN="python3"
  fi
fi

mkdir -p "${OUT_ROOT}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

echo "[Apex2 Phase 0] OUT_ROOT=${OUT_ROOT}"
echo "[Apex2 Phase 0] SEEDS=${SEEDS_CSV}"
echo "[Apex2 Phase 0] GPU_ONLY=${GPU_ONLY}"

run_suite() {
  local name="$1"
  local seed="$2"
  shift 2
  echo
  echo "== ${name} (seed=${seed}) =="
  "$@"
}

if [[ "${GPU_ONLY}" != "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    # Timeseries (competitor baselines are optional; ensure arch/statsmodels/pykalman are installed on the host)
    run_suite "timeseries" "${seed}" "${PY_BIN}" "${TS_SUITE}" \
      --out-dir "${OUT_ROOT}/timeseries_seed${seed}" \
      --seed "${seed}" \
      --repeat 5 \
      --baseline-repeat 5

    # GLM (canonical + targeted high-dim cases).
    #
    # NOTE: The GLM suite already supports a safe high-dim mode (n=10k, p in {100,500})
    # without attempting to materialize enormous Python nested lists like n=100k,p=500.
    run_suite "glm(p=10+high-dim)" "${seed}" "${PY_BIN}" "${GLM_SUITE}" \
      --out-dir "${OUT_ROOT}/glm_seed${seed}" \
      --seed "${seed}" \
      --p 10 \
      --high-dim

    # Econometrics (includes baselines if installed)
    run_suite "econometrics" "${seed}" "${PY_BIN}" "${EC_SUITE}" \
      --out-dir "${OUT_ROOT}/econometrics_seed${seed}" \
      --seed "${seed}" \
      --repeat 5 \
      --n-entities 500 \
      --n-times 8 \
      --n-obs 5000

    "${PY_BIN}" "${EC_REPORT}" \
      --suite "${OUT_ROOT}/econometrics_seed${seed}/econometrics_suite.json" \
      --out "${OUT_ROOT}/econometrics_seed${seed}/econometrics_report.md" || true
  done
else
  echo "[skip] CPU suites: APEX2_GPU_ONLY=1"
fi

# LAPS (GPU-only). On 1-GPU hosts (e.g. V100), run a small single-device matrix via run.py.
# On multi-GPU hosts (e.g. H100 stands), we also support the full suite group.
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || true)"
  gpu_count="${gpu_count:-0}"

  echo
  echo "== laps (GPU, gpus=${gpu_count}, seeds=${SEEDS_CSV}) =="

  APEX2_LAPS_CHAINS="${APEX2_LAPS_CHAINS:-4096}"
  APEX2_LAPS_WARMUP="${APEX2_LAPS_WARMUP:-500}"
  APEX2_LAPS_SAMPLES="${APEX2_LAPS_SAMPLES:-2000}"

  if [[ "${gpu_count}" -ge 4 ]]; then
    "${PY_BIN}" "${LAPS_SUITE}" \
      --out-dir "${OUT_ROOT}/laps_h100" \
      --seeds "${SEEDS_CSV}" \
      --groups "model_comparison" \
      --n-warmup "${APEX2_LAPS_WARMUP}" \
      --n-samples "${APEX2_LAPS_SAMPLES}"
  else
    mkdir -p "${OUT_ROOT}/laps_cuda"
    for seed in "${SEEDS[@]}"; do
      "${PY_BIN}" "${LAPS_RUN}" --label "std_normal_10d" --model std_normal --dim 10 \
        --n-chains "${APEX2_LAPS_CHAINS}" --n-warmup "${APEX2_LAPS_WARMUP}" --n-samples "${APEX2_LAPS_SAMPLES}" \
        --seed "${seed}" --devices "0" --out "${OUT_ROOT}/laps_cuda"
      "${PY_BIN}" "${LAPS_RUN}" --label "eight_schools" --model eight_schools --dim 10 \
        --n-chains "${APEX2_LAPS_CHAINS}" --n-warmup "${APEX2_LAPS_WARMUP}" --n-samples "${APEX2_LAPS_SAMPLES}" \
        --seed "${seed}" --devices "0" --out "${OUT_ROOT}/laps_cuda"
      "${PY_BIN}" "${LAPS_RUN}" --label "neal_funnel_10d" --model neal_funnel --dim 10 \
        --n-chains "${APEX2_LAPS_CHAINS}" --n-warmup "${APEX2_LAPS_WARMUP}" --n-samples "${APEX2_LAPS_SAMPLES}" \
        --seed "${seed}" --devices "0" --out "${OUT_ROOT}/laps_cuda"
      "${PY_BIN}" "${LAPS_RUN}" --label "glm_logistic_n5000_p20" --model glm_logistic --dim 20 --glm-n 5000 --glm-p 20 \
        --n-chains "${APEX2_LAPS_CHAINS}" --n-warmup "${APEX2_LAPS_WARMUP}" --n-samples "${APEX2_LAPS_SAMPLES}" \
        --seed "${seed}" --devices "0" --out "${OUT_ROOT}/laps_cuda"
    done
  fi
else
  echo
  echo "[skip] laps_h100: nvidia-smi not found"
fi

echo
echo "[done] Apex2 Phase 0 results in: ${OUT_ROOT}"
