#!/usr/bin/env bash
set -euo pipefail

# Apex2 Phase 1 benchmark runner (New Suites).
#
# Intended usage:
#   scripts/benchmarks/apex2_phase1.sh /data/apex2_phase1_20260223 "42,123,777"
#
# Notes:
# - Suites record environment snapshots per artifact via scripts/bench_env.py.
# - Phase 1 targets EPYC (canonical x86) and expands coverage across suites
#   that already exist in benchmarks/nextstat-public-benchmarks/suites/.

OUT_ROOT="${1:-}"
SEEDS_CSV="${2:-42,123,777}"

if [[ -z "${OUT_ROOT}" ]]; then
  echo "usage: $0 OUT_ROOT [SEEDS_CSV]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

IC_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/ic_survival/suite.py"
CR_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/competing_risks/suite.py"
GF_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/garch_family/suite.py"
OR_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/ordinal/suite.py"
GT_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/gamma_tweedie/suite.py"
LMM_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/lmm/suite.py"
PK_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/pk_3cpt/suite.py"
ODE_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/ode_pk/suite.py"
PD_SUITE="${REPO_ROOT}/benchmarks/nextstat-public-benchmarks/suites/pd_models/suite.py"

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

echo "[Apex2 Phase 1] OUT_ROOT=${OUT_ROOT}"
echo "[Apex2 Phase 1] SEEDS=${SEEDS_CSV}"

FAIL=0

FILTER_CSV="${APEX2_PHASE1_SUITES:-}"
declare -A FILTER=()
if [[ -n "${FILTER_CSV}" ]]; then
  IFS=',' read -r -a FILTER_ITEMS <<< "${FILTER_CSV}"
  for x in "${FILTER_ITEMS[@]}"; do
    x="$(echo "$x" | tr -d ' ')"
    if [[ -n "$x" ]]; then
      FILTER["$x"]=1
    fi
  done
  echo "[Apex2 Phase 1] SUITES_FILTER=${FILTER_CSV}"
fi

should_run() {
  local suite="$1"
  if [[ -z "${FILTER_CSV}" ]]; then
    return 0
  fi
  [[ "${FILTER[$suite]:-0}" == "1" ]]
}

run_suite() {
  local name="$1"
  local seed="$2"
  shift 2
  echo
  echo "== ${name} (seed=${seed}) =="
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[warn] suite failed (rc=${rc}): ${name} (seed=${seed})" >&2
    FAIL=1
  fi
}

for seed in "${SEEDS[@]}"; do
  # Interval-censored survival (scipy baseline parity; no standard Python competitor).
  if should_run "ic_survival"; then
    run_suite "ic_survival" "${seed}" "${PY_BIN}" "${IC_SUITE}" \
      --out-dir "${OUT_ROOT}/ic_survival_seed${seed}" \
      --seed "${seed}" \
      --repeat 20
  fi

  # Competing risks (lifelines + optional R cmprsk).
  if should_run "competing_risks"; then
    run_suite "competing_risks" "${seed}" "${PY_BIN}" "${CR_SUITE}" \
      --out-dir "${OUT_ROOT}/competing_risks_seed${seed}" \
      --seed "${seed}" \
      --baseline-repeat 1
  fi

  # GARCH family (arch baseline where applicable).
  if should_run "garch_family"; then
    run_suite "garch_family" "${seed}" "${PY_BIN}" "${GF_SUITE}" \
      --out-dir "${OUT_ROOT}/garch_family_seed${seed}" \
      --seed "${seed}" \
      --baseline-repeat 5 \
      --run-all-baselines
  fi

  # Ordinal regression (statsmodels baseline).
  if should_run "ordinal"; then
    run_suite "ordinal" "${seed}" "${PY_BIN}" "${OR_SUITE}" \
      --out-dir "${OUT_ROOT}/ordinal_seed${seed}" \
      --seed "${seed}"
  fi

  # Gamma/Tweedie GLM (statsmodels/glum baselines where available).
  if should_run "gamma_tweedie"; then
    run_suite "gamma_tweedie" "${seed}" "${PY_BIN}" "${GT_SUITE}" \
      --out-dir "${OUT_ROOT}/gamma_tweedie_seed${seed}" \
      --seed "${seed}"
  fi

  # Linear mixed models (statsmodels MixedLM baseline).
  if should_run "lmm"; then
    run_suite "lmm" "${seed}" "${PY_BIN}" "${LMM_SUITE}" \
      --out-dir "${OUT_ROOT}/lmm_seed${seed}" \
      --seed "${seed}"
  fi

  # 3-cpt PK (NS-only; parity vs ground truth synthetic).
  if should_run "pk_3cpt"; then
    run_suite "pk_3cpt" "${seed}" "${PY_BIN}" "${PK_SUITE}" \
      --out-dir "${OUT_ROOT}/pk_3cpt_seed${seed}" \
      --seed "${seed}"
  fi

  # ODE PK (scipy solve_ivp baseline).
  if should_run "ode_pk"; then
    run_suite "ode_pk" "${seed}" "${PY_BIN}" "${ODE_SUITE}" \
      --out-dir "${OUT_ROOT}/ode_pk_seed${seed}" \
      --seed "${seed}"
  fi

  # PD models (baseline = pure Python NLL + scipy optimize for Emax fits).
  if should_run "pd_models"; then
    run_suite "pd_models" "${seed}" "${PY_BIN}" "${PD_SUITE}" \
      --out-dir "${OUT_ROOT}/pd_models_seed${seed}" \
      --seed "${seed}"
  fi
done

echo
echo "[done] Apex2 Phase 1 results in: ${OUT_ROOT}"
exit $FAIL
