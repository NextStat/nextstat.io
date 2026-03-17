#!/usr/bin/env bash
set -euo pipefail

REPO="${1:?repo root is required}"
VENV="${2:?venv path is required}"
TARGET="${3:?cargo target dir is required}"
shift 3
BUILD_ARGS=("$@")

WHEEL_DIR="${REPO}/.nextstat_remote_wheels"

cd "${REPO}/bindings/ns-py"
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${PATH}"
export CARGO_TARGET_DIR="${TARGET}"

mkdir -p "${WHEEL_DIR}"

"${VENV}/bin/python" -m maturin build \
  --release \
  --interpreter "${VENV}/bin/python" \
  "${BUILD_ARGS[@]}" \
  --out "${WHEEL_DIR}"

wheel_path="$(ls -1t "${WHEEL_DIR}"/nextstat-*.whl | head -n 1)"
if [[ -z "${wheel_path}" ]]; then
  echo "missing built nextstat wheel in ${WHEEL_DIR}" >&2
  exit 1
fi

"${VENV}/bin/python" -m pip install --force-reinstall --no-deps "${wheel_path}"
