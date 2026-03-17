#!/usr/bin/env bash
set -euo pipefail

# Canonical remote runner for the Bayesian publish_snapshot lane on nextstat-bench.
#
# It rsyncs the current working tree snapshot to the remote host, builds the
# local ns-py bindings into an isolated venv, runs publish_snapshot.py for the
# Bayesian suite, validates the produced snapshot, and syncs the snapshot back
# locally for inspection.

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

BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/nextstat_bayesian_publisher_repo_${STAMP}}"
BENCH_REMOTE_VENV="${BENCH_REMOTE_VENV:-/tmp/nextstat_bayesian_publisher_venv_${STAMP}}"
BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/nextstat_bayesian_publisher_target_${STAMP}}"
BENCH_REMOTE_SNAPSHOT_ROOT="${BENCH_REMOTE_SNAPSHOT_ROOT:-/tmp/nextstat_bayesian_publisher_snapshots_${STAMP}}"
BENCH_SNAPSHOT_ID="${BENCH_SNAPSHOT_ID:-bayesian-publisher-${STAMP}}"
BENCH_REMOTE_OUT="${BENCH_REMOTE_SNAPSHOT_ROOT}/${BENCH_SNAPSHOT_ID}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-$ROOT_DIR/benchmarks/artifacts/bayesian_publisher_validation_${STAMP}/${BENCH_HOST}/${BENCH_SNAPSHOT_ID}}"
BENCH_LOCAL_ROOT="$(dirname "${BENCH_LOCAL_OUT}")"
BENCH_REMOTE_REGISTRY="${BENCH_REMOTE_SNAPSHOT_ROOT}/snapshot_registry.json"
BENCH_LOCAL_REGISTRY="${BENCH_LOCAL_ROOT}/snapshot_registry.json"

BENCH_BACKENDS="${BENCH_BACKENDS:-nextstat,nextstat_dense}"
BENCH_CHAINS="${BENCH_CHAINS:-4}"
BENCH_WARMUP="${BENCH_WARMUP:-500}"
BENCH_SAMPLES="${BENCH_SAMPLES:-1000}"
BENCH_SEED="${BENCH_SEED:-0}"
BENCH_MAX_TREEDEPTH="${BENCH_MAX_TREEDEPTH:-10}"
BENCH_TARGET_ACCEPT="${BENCH_TARGET_ACCEPT:-0.8}"
BENCH_INIT_JITTER_REL="${BENCH_INIT_JITTER_REL:-0.10}"
BENCH_DETERMINISTIC="${BENCH_DETERMINISTIC:-1}"
BENCH_EXTRA_PIP_PACKAGES="${BENCH_EXTRA_PIP_PACKAGES:-}"
BENCH_CMDSTAN_VERSION="${BENCH_CMDSTAN_VERSION:-2.38.0}"
BENCH_CMDSTAN_CORES="${BENCH_CMDSTAN_CORES:-4}"

mkdir -p "${BENCH_LOCAL_OUT}"
mkdir -p "${BENCH_LOCAL_ROOT}"

echo "[bayesian-publisher-remote] host=${REMOTE_SPEC}"
echo "[bayesian-publisher-remote] remote_repo=${BENCH_REMOTE_REPO}"
echo "[bayesian-publisher-remote] remote_venv=${BENCH_REMOTE_VENV}"
echo "[bayesian-publisher-remote] remote_target=${BENCH_REMOTE_TARGET}"
echo "[bayesian-publisher-remote] remote_snapshot_root=${BENCH_REMOTE_SNAPSHOT_ROOT}"
echo "[bayesian-publisher-remote] remote_registry=${BENCH_REMOTE_REGISTRY}"
echo "[bayesian-publisher-remote] snapshot_id=${BENCH_SNAPSHOT_ID}"
echo "[bayesian-publisher-remote] local_out=${BENCH_LOCAL_OUT}"
echo "[bayesian-publisher-remote] local_registry=${BENCH_LOCAL_REGISTRY}"
echo "[bayesian-publisher-remote] backends=${BENCH_BACKENDS}"

echo "[bayesian-publisher-remote] probe remote..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; uname -a; python3 --version; cargo --version"

echo "[bayesian-publisher-remote] create remote directories..."
"${SSH_BASE[@]}" "mkdir -p '${BENCH_REMOTE_REPO}' '${BENCH_REMOTE_REPO}/benchmarks' '${BENCH_REMOTE_REPO}/bindings' '${BENCH_REMOTE_REPO}/crates' '${BENCH_REMOTE_SNAPSHOT_ROOT}'"

echo "[bayesian-publisher-remote] rsync snapshot..."
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
  echo "[bayesian-publisher-remote] rsync reported vanished source files from unrelated concurrent work; continuing with the synchronized snapshot"
fi

echo "[bayesian-publisher-remote] create remote venv..."
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_VENV}" "${BENCH_BACKENDS}" "${BENCH_EXTRA_PIP_PACKAGES}" <<'EOS'
set -euo pipefail
VENV="$1"
BACKENDS="$2"
EXTRA_PIP_PACKAGES="${3-}"
REQ_PKGS=(maturin jsonschema)
if [[ ",$BACKENDS," == *",cmdstanpy,"* ]]; then
  REQ_PKGS+=(cmdstanpy)
fi
if [[ ",$BACKENDS," == *",pymc,"* ]]; then
  REQ_PKGS+=(pymc arviz)
fi
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools "${REQ_PKGS[@]}" >/dev/null
if [[ -n "$EXTRA_PIP_PACKAGES" ]]; then
  "$VENV/bin/python" -m pip install $EXTRA_PIP_PACKAGES >/dev/null
fi
EOS

echo "[bayesian-publisher-remote] build local nextstat into remote venv..."
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

echo "[bayesian-publisher-remote] provision optional backends..."
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_REPO}" "${BENCH_REMOTE_VENV}" "${BENCH_BACKENDS}" "${BENCH_CMDSTAN_VERSION}" "${BENCH_CMDSTAN_CORES}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
BACKENDS="$3"
CMDSTAN_VERSION="$4"
CMDSTAN_CORES="$5"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

if [[ ",$BACKENDS," == *",cmdstanpy,"* ]]; then
  VENDOR_ROOT="$REPO/benchmarks/nextstat-public-benchmarks/vendor/cmdstan"
  mkdir -p "$VENDOR_ROOT"
  "$VENV/bin/python" - <<'PY' "$VENDOR_ROOT" "$CMDSTAN_VERSION" "$CMDSTAN_CORES"
from pathlib import Path
import re
import sys
from cmdstanpy import cmdstan_path, set_cmdstan_path

try:
    from cmdstanpy import install_cmdstan as install_cmdstan_compat
except Exception:
    from cmdstanpy.install_cmdstan import InstallationSettings, run_install

    def install_cmdstan_compat(**kwargs):
        settings = InstallationSettings(
            version=kwargs.get("version"),
            dir=kwargs.get("dir"),
            overwrite=kwargs.get("overwrite", False),
            verbose=kwargs.get("verbose", False),
            progress=kwargs.get("progress", False),
            cores=kwargs.get("cores", 1),
            compiler=kwargs.get("compiler", False),
        )
        run_install(settings)
        return True

root = Path(sys.argv[1])
version = sys.argv[2].strip()
cores = int(sys.argv[3])

def version_key(path: Path) -> tuple[int, ...]:
    raw = path.name.removeprefix("cmdstan-")
    parts = []
    for token in raw.split("."):
        m = re.match(r"^(\d+)", token)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)

candidates = sorted((p for p in root.glob("cmdstan-*") if p.is_dir()), key=version_key)
if candidates:
    chosen = candidates[-1].resolve()
    set_cmdstan_path(str(chosen))
else:
    kwargs = {"dir": str(root), "overwrite": False, "verbose": False, "cores": cores}
    if version:
        kwargs["version"] = version
    ok = install_cmdstan_compat(**kwargs)
    if not ok:
        raise SystemExit("install_cmdstan failed")
    candidates = sorted((p for p in root.glob("cmdstan-*") if p.is_dir()), key=version_key)
    if candidates:
        chosen = candidates[-1].resolve()
        set_cmdstan_path(str(chosen))
print(f"cmdstan_path={cmdstan_path()}")
PY
fi

if [[ ",$BACKENDS," == *",pymc,"* ]]; then
  "$VENV/bin/python" - <<'PY'
import arviz
import pymc
print(f"pymc={pymc.__version__}")
print(f"arviz={arviz.__version__}")
PY
fi
EOS

echo "[bayesian-publisher-remote] run publish_snapshot.py..."
"${SSH_BASE[@]}" bash -s -- \
  "${BENCH_REMOTE_REPO}" \
  "${BENCH_REMOTE_VENV}" \
  "${BENCH_REMOTE_TARGET}" \
  "${BENCH_REMOTE_SNAPSHOT_ROOT}" \
  "${BENCH_SNAPSHOT_ID}" \
  "${BENCH_BACKENDS}" \
  "${BENCH_CHAINS}" \
  "${BENCH_WARMUP}" \
  "${BENCH_SAMPLES}" \
  "${BENCH_SEED}" \
  "${BENCH_MAX_TREEDEPTH}" \
  "${BENCH_TARGET_ACCEPT}" \
  "${BENCH_INIT_JITTER_REL}" \
  "${BENCH_DETERMINISTIC}" \
  "${LOCAL_GIT_COMMIT}" <<'EOS'
set -euo pipefail
REPO="$1"
VENV="$2"
TARGET="$3"
SNAPSHOT_ROOT="$4"
SNAPSHOT_ID="$5"
BACKENDS="$6"
CHAINS="$7"
WARMUP="$8"
SAMPLES="$9"
SEED="${10}"
MAX_TREEDEPTH="${11}"
TARGET_ACCEPT="${12}"
INIT_JITTER_REL="${13}"
DETERMINISTIC="${14}"
GIT_COMMIT="${15}"

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

if [[ ",$BACKENDS," == *",cmdstanpy,"* ]]; then
  VENDOR_ROOT="$REPO/benchmarks/nextstat-public-benchmarks/vendor/cmdstan"
  if [[ -d "$VENDOR_ROOT" ]]; then
    CMDSTAN_DIR="$(find "$VENDOR_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'cmdstan-*' | sort -V | tail -n 1)"
    if [[ -n "$CMDSTAN_DIR" ]]; then
      export CMDSTAN="$CMDSTAN_DIR"
      echo "[bayesian-publisher-remote] cmdstan_vendor=${CMDSTAN_DIR}"
    fi
  fi
fi

if [[ ",$BACKENDS," == *",pymc,"* ]]; then
  export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:-blas__ldflags=-lblas}"
  echo "[bayesian-publisher-remote] pytensor_flags=${PYTENSOR_FLAGS}"
fi

PUBLISH_CMD=(
  "$VENV/bin/python"
  "scripts/publish_snapshot.py"
  "--snapshot-id" "$SNAPSHOT_ID"
  "--out-root" "$SNAPSHOT_ROOT"
  "--bayesian"
  "--bayesian-backends" "$BACKENDS"
  "--bayesian-n-chains" "$CHAINS"
  "--bayesian-warmup" "$WARMUP"
  "--bayesian-samples" "$SAMPLES"
  "--bayesian-seed" "$SEED"
  "--bayesian-max-treedepth" "$MAX_TREEDEPTH"
  "--bayesian-target-accept" "$TARGET_ACCEPT"
  "--bayesian-init-jitter-rel" "$INIT_JITTER_REL"
)
if [[ "$DETERMINISTIC" == "1" ]]; then
  PUBLISH_CMD+=(--deterministic)
fi
"${PUBLISH_CMD[@]}"

"$VENV/bin/python" "scripts/validate_artifacts.py" --strict "$SNAPSHOT_ROOT"
EOS

echo "[bayesian-publisher-remote] sync snapshot root..."
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_SNAPSHOT_ROOT}/" "${BENCH_LOCAL_ROOT}/"

echo "[bayesian-publisher-remote] done: ${BENCH_LOCAL_ROOT}"
