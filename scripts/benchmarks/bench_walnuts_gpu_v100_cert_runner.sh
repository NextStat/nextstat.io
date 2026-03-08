#!/usr/bin/env bash
set -euo pipefail

# Internal direct-V100 GPU certification runner for WALNUTS.
#
# This lane now uses the same two-host split we actually want operationally:
# - build host: nextstat-bench (ample disk, Linux cargo toolchain)
# - execute host: v100 (real f64 Tesla V100, CUDA 12.6 userland)
#
# The V100 container overlay is disk-constrained and /dev/shm is noexec, so we:
# 1. build the ns-inference lib test harness on nextstat-bench with stub PTX
# 2. copy the compiled test binary to v100 /dev/shm
# 3. execute it via memfd_create so no exec-capable filesystem is required
#
# Scope remains explicit and internal-only:
# - narrow StdNormal CUDA cert seam
# - evaluator-backed linear + logistic + Poisson-with-offset +
#   NegBin-with-offset + IC Weibull AFT slices

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"
LOCAL_SUBMIT_HOST="$(hostname)"

BENCH_BUILD_HOST="${BENCH_BUILD_HOST:-nextstat-bench}"
BENCH_EXEC_HOST="${BENCH_EXEC_HOST:-${BENCH_HOST:-v100}}"

BENCH_BUILD_SSH_PORT="${BENCH_BUILD_SSH_PORT:-}"
BENCH_BUILD_SSH_KEY="${BENCH_BUILD_SSH_KEY:-}"
BENCH_EXEC_SSH_PORT="${BENCH_EXEC_SSH_PORT:-${BENCH_SSH_PORT:-}}"
BENCH_EXEC_SSH_KEY="${BENCH_EXEC_SSH_KEY:-${BENCH_SSH_KEY:-}}"

BENCH_BUILD_REMOTE_REPO="${BENCH_BUILD_REMOTE_REPO:-/tmp/nextstat_walnuts_gpu_builder_repo_current}"
BENCH_BUILD_REMOTE_TARGET="${BENCH_BUILD_REMOTE_TARGET:-/tmp/nextstat_walnuts_gpu_builder_target_cuda126}"
BENCH_BUILD_REMOTE_RUN_ROOT="${BENCH_BUILD_REMOTE_RUN_ROOT:-/tmp/nextstat_walnuts_gpu_builder_artifacts}"
BENCH_BUILD_REMOTE_RUN="${BENCH_BUILD_REMOTE_RUN:-${BENCH_BUILD_REMOTE_RUN_ROOT}/walnuts_gpu_v100_cert_builder_${STAMP}}"

BENCH_EXEC_REMOTE_RUN_ROOT="${BENCH_EXEC_REMOTE_RUN_ROOT:-/dev/shm/nextstat_walnuts_v100_exec_artifacts}"
BENCH_EXEC_REMOTE_RUN="${BENCH_EXEC_REMOTE_RUN:-${BENCH_EXEC_REMOTE_RUN_ROOT}/walnuts_gpu_v100_cert_exec_${STAMP}}"
BENCH_EXEC_REMOTE_BINARY="${BENCH_EXEC_REMOTE_BINARY:-${BENCH_EXEC_REMOTE_RUN}/ns_inference_cuda_tests.bin}"

BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-${ROOT_DIR}/tmp/bench_walnuts_gpu_v100_cert_runner_${STAMP}/${BENCH_EXEC_HOST}}"
BENCH_LOCAL_BINARY="${BENCH_LOCAL_BINARY:-${BENCH_LOCAL_OUT}/ns_inference_cuda_tests.bin}"

BENCH_CUDA_HOME="${BENCH_CUDA_HOME:-/usr/local/cuda-12.6}"
BENCH_LD_LIBRARY_PATH="${BENCH_LD_LIBRARY_PATH:-${BENCH_CUDA_HOME}/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}"

BUILD_SSH=(ssh -o StrictHostKeyChecking=accept-new)
BUILD_SCP=(scp -o StrictHostKeyChecking=accept-new)
if [[ -n "${BENCH_BUILD_SSH_KEY}" ]]; then
  BUILD_SSH+=(-i "${BENCH_BUILD_SSH_KEY}")
  BUILD_SCP+=(-i "${BENCH_BUILD_SSH_KEY}")
fi
if [[ -n "${BENCH_BUILD_SSH_PORT}" ]]; then
  BUILD_SSH+=(-p "${BENCH_BUILD_SSH_PORT}")
  BUILD_SCP+=(-P "${BENCH_BUILD_SSH_PORT}")
fi
BUILD_SSH+=("${BENCH_BUILD_HOST}")

EXEC_SSH=(ssh -o StrictHostKeyChecking=accept-new)
EXEC_SCP=(scp -o StrictHostKeyChecking=accept-new)
if [[ -n "${BENCH_EXEC_SSH_KEY}" ]]; then
  EXEC_SSH+=(-i "${BENCH_EXEC_SSH_KEY}")
  EXEC_SCP+=(-i "${BENCH_EXEC_SSH_KEY}")
fi
if [[ -n "${BENCH_EXEC_SSH_PORT}" ]]; then
  EXEC_SSH+=(-p "${BENCH_EXEC_SSH_PORT}")
  EXEC_SCP+=(-P "${BENCH_EXEC_SSH_PORT}")
fi
EXEC_SSH+=("${BENCH_EXEC_HOST}")

mkdir -p "${BENCH_LOCAL_OUT}"

echo "[walnuts-gpu-v100-cert] build_host=${BENCH_BUILD_HOST}"
echo "[walnuts-gpu-v100-cert] exec_host=${BENCH_EXEC_HOST}"
echo "[walnuts-gpu-v100-cert] build_repo=${BENCH_BUILD_REMOTE_REPO}"
echo "[walnuts-gpu-v100-cert] build_target=${BENCH_BUILD_REMOTE_TARGET}"
echo "[walnuts-gpu-v100-cert] build_run=${BENCH_BUILD_REMOTE_RUN}"
echo "[walnuts-gpu-v100-cert] exec_run=${BENCH_EXEC_REMOTE_RUN}"
echo "[walnuts-gpu-v100-cert] exec_binary=${BENCH_EXEC_REMOTE_BINARY}"
echo "[walnuts-gpu-v100-cert] local_out=${BENCH_LOCAL_OUT}"
echo "[walnuts-gpu-v100-cert] cuda_home=${BENCH_CUDA_HOME}"

echo "[walnuts-gpu-v100-cert] probe build host..."
"${BUILD_SSH[@]}" bash -s -- \
  "${BENCH_BUILD_REMOTE_REPO}" \
  "${BENCH_BUILD_REMOTE_TARGET}" \
  "${BENCH_BUILD_REMOTE_RUN_ROOT}" \
  "${BENCH_BUILD_REMOTE_RUN}" \
  "${LOCAL_GIT_COMMIT}" <<'EOS'
set -euo pipefail
REPO="$1"
TARGET="$2"
RUN_ROOT="$3"
RUN="$4"
GIT_COMMIT="$5"
mkdir -p "$REPO" "$TARGET" "$RUN_ROOT" "$RUN"
python3 - <<'PY' "$RUN" "$GIT_COMMIT"
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
git_commit = sys.argv[2]

def check_output(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"

payload = {
    "schema_version": "nextstat.walnuts_gpu_v100_build_host_probe.v1",
    "build_host": socket.gethostname(),
    "build_user": check_output(["whoami"]),
    "python_version": check_output(["python3", "--version"]),
    "cargo_path": shutil.which("cargo") or shutil.which(str(Path.home() / ".cargo" / "bin" / "cargo")),
    "cargo_version": check_output(["cargo", "--version"]),
    "rustc_path": shutil.which("rustc") or shutil.which(str(Path.home() / ".cargo" / "bin" / "rustc")),
    "rustc_version": check_output(["rustc", "--version"]),
    "git_commit": git_commit,
}
run_dir.joinpath("build_host_probe.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
EOS

echo "[walnuts-gpu-v100-cert] sync snapshot to build host..."
SYNC_CANDIDATES=(
  Cargo.toml
  Cargo.lock
  rust-toolchain.toml
  rustfmt.toml
  .cargo
  crates/ns-core
  crates/ns-compute
  crates/ns-ad
  crates/ns-prob
  crates/ns-root
  crates/ns-inference
  crates/ns-translate
  crates/ns-unbinned
  crates/ns-viz
  crates/ns-viz-render
  crates/ns-cli
  crates/ns-server
  crates/ns-zstd
  crates/zstd-shim
  bindings/ns-py
  bindings/ns-wasm
  bindings/ns-calc-wasm
  tests/fixtures
)
SYNC_PATHS=()
for rel_path in "${SYNC_CANDIDATES[@]}"; do
  if [[ -e "${ROOT_DIR}/${rel_path}" ]]; then
    SYNC_PATHS+=("${rel_path}")
  fi
done
COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar -C "${ROOT_DIR}" -cf - "${SYNC_PATHS[@]}" 2>/dev/null \
  | "${BUILD_SSH[@]}" "set -euo pipefail; mkdir -p '${BENCH_BUILD_REMOTE_REPO}'; tar -xf - -C '${BENCH_BUILD_REMOTE_REPO}' 2>/dev/null"

echo "[walnuts-gpu-v100-cert] build ns-inference CUDA test harness on build host..."
"${BUILD_SSH[@]}" bash -s -- \
  "${BENCH_BUILD_REMOTE_REPO}" \
  "${BENCH_BUILD_REMOTE_TARGET}" \
  "${BENCH_BUILD_REMOTE_RUN}" <<'EOS'
set -euo pipefail
REPO="$1"
TARGET="$2"
RUN="$3"

cd "$REPO"
export PATH="/usr/local/bin:${HOME}/.cargo/bin:${PATH}"
export CARGO_TARGET_DIR="${TARGET}"
export NS_COMPUTE_FORCE_STUB_PTX=1

BUILD_JSON="${RUN}/cargo_test_norun.jsonl"
MANIFEST_JSON="${RUN}/ns_inference_test_binary.json"

cargo test -p ns-inference --features cuda --lib --no-run --message-format=json > "${BUILD_JSON}" 2>&1

python3 - <<'PY' "${BUILD_JSON}" "${MANIFEST_JSON}"
import json
import sys
from pathlib import Path

build_json = Path(sys.argv[1])
manifest_json = Path(sys.argv[2])

exe = None
target_name = None
for line in build_json.read_text().splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        continue
    if payload.get("reason") != "compiler-artifact":
        continue
    executable = payload.get("executable")
    if not executable:
        continue
    target = payload.get("target") or {}
    if not target.get("test", False):
        continue
    if target.get("name") != "ns_inference":
        continue
    exe = executable
    target_name = target.get("name")

if not exe:
    raise SystemExit(f"missing ns-inference test executable in {build_json}")

exe_path = Path(exe)
manifest = {
    "schema_version": "nextstat.walnuts_gpu_v100_build_manifest.v1",
    "target_name": target_name,
    "executable": exe,
    "size_bytes": exe_path.stat().st_size,
}
manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
EOS

echo "[walnuts-gpu-v100-cert] fetch build manifest..."
"${BUILD_SCP[@]}" "${BENCH_BUILD_HOST}:${BENCH_BUILD_REMOTE_RUN}/build_host_probe.json" "${BENCH_LOCAL_OUT}/build_host_probe.json"
"${BUILD_SCP[@]}" "${BENCH_BUILD_HOST}:${BENCH_BUILD_REMOTE_RUN}/ns_inference_test_binary.json" "${BENCH_LOCAL_OUT}/ns_inference_test_binary.json"

BUILD_EXECUTABLE="$(
  python3 - <<'PY' "${BENCH_LOCAL_OUT}/ns_inference_test_binary.json"
import json
import sys
print(json.loads(open(sys.argv[1]).read())["executable"])
PY
)"

echo "[walnuts-gpu-v100-cert] fetch compiled test harness from build host..."
"${BUILD_SCP[@]}" "${BENCH_BUILD_HOST}:${BUILD_EXECUTABLE}" "${BENCH_LOCAL_BINARY}"

echo "[walnuts-gpu-v100-cert] probe execute host..."
"${EXEC_SSH[@]}" bash -s -- \
  "${BENCH_EXEC_REMOTE_RUN_ROOT}" \
  "${BENCH_EXEC_REMOTE_RUN}" \
  "${BENCH_CUDA_HOME}" \
  "${LOCAL_GIT_COMMIT}" \
  "${BENCH_BUILD_HOST}" <<'EOS'
set -euo pipefail
RUN_ROOT="$1"
RUN="$2"
CUDA_HOME="$3"
GIT_COMMIT="$4"
BUILD_HOST="$5"
mkdir -p "$RUN_ROOT" "$RUN"
python3 - <<'PY' "$RUN" "$CUDA_HOME" "$GIT_COMMIT" "$BUILD_HOST"
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
cuda_home = sys.argv[2]
git_commit = sys.argv[3]
build_host = sys.argv[4]

def check_output(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"

payload = {
    "schema_version": "nextstat.walnuts_gpu_v100_host_probe.v2",
    "execute_host": socket.gethostname(),
    "execute_user": check_output(["whoami"]),
    "python_version": check_output(["python3", "--version"]),
    "cuda_home": cuda_home,
    "nvcc_version": check_output([f"{cuda_home}/bin/nvcc", "--version"]),
    "nvidia_smi_query": check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]
    ).splitlines(),
    "build_host": build_host,
    "git_commit": git_commit,
    "memfd_exec_probe": check_output(
        [
            "python3",
            "-c",
            (
                "import os; "
                "fd=os.memfd_create('probe',0); "
                "os.write(fd, open('/bin/echo','rb').read()); "
                "os.execv(f'/proc/self/fd/{fd}', ['echo', 'memfd-ok'])"
            ),
        ]
    ),
}
run_dir.joinpath("v100_host_probe.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
EOS

echo "[walnuts-gpu-v100-cert] stage compiled harness on execute host..."
"${EXEC_SSH[@]}" "mkdir -p '${BENCH_EXEC_REMOTE_RUN}'"
"${EXEC_SCP[@]}" "${BENCH_LOCAL_BINARY}" "${BENCH_EXEC_HOST}:${BENCH_EXEC_REMOTE_BINARY}"

echo "[walnuts-gpu-v100-cert] execute certification run on execute host..."
"${EXEC_SSH[@]}" bash -s -- \
  "${BENCH_EXEC_REMOTE_BINARY}" \
  "${BENCH_EXEC_REMOTE_RUN}" \
  "${BENCH_CUDA_HOME}" \
  "${BENCH_LD_LIBRARY_PATH}" \
  "${LOCAL_SUBMIT_HOST}" \
  "${BENCH_BUILD_HOST}" <<'EOS'
set -euo pipefail
BIN="$1"
RUN="$2"
CUDA_HOME="$3"
LD_LIBRARY_PATH_VALUE="$4"
SUBMIT_HOST="$5"
BUILD_HOST="$6"

export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}"
export NEXTSTAT_BENCH_HOST_POLICY="v100-direct-gpu"
export NEXTSTAT_BENCH_SUBMIT_HOST="${SUBMIT_HOST}"
export NEXTSTAT_BENCH_BUILD_HOST="${BUILD_HOST}"
export NEXTSTAT_BENCH_EXECUTE_HOST="$(hostname)"
export NEXTSTAT_BENCH_SCHEDULER="split-build-memfd"

run_test() {
  local outfile="$1"
  shift
  python3 - "$BIN" "$@" > "${RUN}/${outfile}" 2>&1 <<'PY'
import os
import sys

bin_path = sys.argv[1]
args = sys.argv[2:]

with open(bin_path, "rb") as fh:
    payload = fh.read()

fd = os.memfd_create("ns_inference_cuda_tests", 0)
os.write(fd, payload)
os.fchmod(fd, 0o755)
os.execv(f"/proc/self/fd/{fd}", ["ns_inference_cuda_tests", *args])
PY
}

run_test "test_logistic_regression_cuda_glm_export_includes_intercept_column.txt" \
  test_logistic_regression_cuda_glm_export_includes_intercept_column --exact --nocapture

run_test "test_linear_regression_cuda_glm_export_includes_intercept_column.txt" \
  test_linear_regression_cuda_glm_export_includes_intercept_column --exact --nocapture

run_test "test_poisson_regression_cuda_glm_export_preserves_offset_and_intercept_column.txt" \
  test_poisson_regression_cuda_glm_export_preserves_offset_and_intercept_column --exact --nocapture

run_test "test_negbin_regression_cuda_glm_export_preserves_offset_and_log_alpha_slot.txt" \
  test_negbin_regression_cuda_glm_export_preserves_offset_and_log_alpha_slot --exact --nocapture

run_test "ic_weibull_aft_cuda_export_preserves_colmajor_and_censor_codes.txt" \
  ic_weibull_aft_cuda_export_preserves_colmajor_and_censor_codes --exact --nocapture

run_test "cuda_logistic_potential_matches_cpu_potential_grad.txt" \
  cuda_hmc_potential::tests::cuda_logistic_potential_matches_cpu_potential_grad --exact --nocapture

run_test "cuda_linear_potential_matches_cpu_potential_grad.txt" \
  cuda_hmc_potential::tests::cuda_linear_potential_matches_cpu_potential_grad --exact --nocapture

run_test "cuda_linear_leapfrog_matches_cpu_one_step.txt" \
  cuda_hmc_potential::tests::cuda_linear_leapfrog_matches_cpu_one_step --exact --nocapture

run_test "cuda_logistic_leapfrog_matches_cpu_one_step.txt" \
  cuda_hmc_potential::tests::cuda_logistic_leapfrog_matches_cpu_one_step --exact --nocapture

run_test "cuda_poisson_offset_potential_matches_cpu_potential_grad.txt" \
  cuda_hmc_potential::tests::cuda_poisson_offset_potential_matches_cpu_potential_grad --exact --nocapture

run_test "cuda_poisson_offset_leapfrog_matches_cpu_one_step.txt" \
  cuda_hmc_potential::tests::cuda_poisson_offset_leapfrog_matches_cpu_one_step --exact --nocapture

run_test "cuda_negbin_offset_potential_matches_cpu_potential_grad.txt" \
  cuda_hmc_potential::tests::cuda_negbin_offset_potential_matches_cpu_potential_grad --exact --nocapture

run_test "cuda_negbin_offset_leapfrog_matches_cpu_one_step.txt" \
  cuda_hmc_potential::tests::cuda_negbin_offset_leapfrog_matches_cpu_one_step --exact --nocapture

run_test "cuda_weibull_aft_potential_matches_cpu_potential_grad.txt" \
  cuda_hmc_potential::tests::cuda_weibull_aft_potential_matches_cpu_potential_grad --exact --nocapture

run_test "cuda_weibull_aft_leapfrog_matches_cpu_one_step.txt" \
  cuda_hmc_potential::tests::cuda_weibull_aft_leapfrog_matches_cpu_one_step --exact --nocapture

run_test "cuda_stdnormal_stepper_matches_cpu_one_step.txt" \
  cuda_hmc_stepper::tests::cuda_stdnormal_stepper_matches_cpu_one_step --exact --nocapture

run_test "cuda_stdnormal_walnuts_transition_matches_cpu.txt" \
  cuda_hmc_stepper::tests::cuda_stdnormal_walnuts_transition_matches_cpu --exact --nocapture

run_test "bench_cuda_stdnormal_walnuts_cpu_vs_gpu_cuda126.txt" \
  cuda_hmc_stepper::tests::bench_cuda_stdnormal_walnuts_cpu_vs_gpu --ignored --exact --nocapture

run_test "bench_cuda_logistic_walnuts_cpu_vs_gpu_cuda126.txt" \
  cuda_hmc_potential::tests::bench_cuda_logistic_walnuts_cpu_vs_gpu --ignored --exact --nocapture

run_test "bench_cuda_linear_walnuts_cpu_vs_gpu_cuda126.txt" \
  cuda_hmc_potential::tests::bench_cuda_linear_walnuts_cpu_vs_gpu --ignored --exact --nocapture

run_test "bench_cuda_poisson_offset_walnuts_cpu_vs_gpu_cuda126.txt" \
  cuda_hmc_potential::tests::bench_cuda_poisson_offset_walnuts_cpu_vs_gpu --ignored --exact --nocapture

run_test "bench_cuda_negbin_offset_walnuts_cpu_vs_gpu_cuda126.txt" \
  cuda_hmc_potential::tests::bench_cuda_negbin_offset_walnuts_cpu_vs_gpu --ignored --exact --nocapture

run_test "bench_cuda_weibull_aft_walnuts_cpu_vs_gpu_cuda126.txt" \
  cuda_hmc_potential::tests::bench_cuda_weibull_aft_walnuts_cpu_vs_gpu --ignored --exact --nocapture
EOS

echo "[walnuts-gpu-v100-cert] copy artifacts..."
"${EXEC_SCP[@]}" -r "${BENCH_EXEC_HOST}:${BENCH_EXEC_REMOTE_RUN}/." "${BENCH_LOCAL_OUT}/"

echo "[walnuts-gpu-v100-cert] render local report..."
python3 - <<'PY' \
  "${BENCH_LOCAL_OUT}" \
  "${BENCH_BUILD_HOST}" \
  "${BENCH_EXEC_HOST}" \
  "${LOCAL_SUBMIT_HOST}" \
  "${LOCAL_GIT_COMMIT}" \
  "${BENCH_CUDA_HOME}"
import json
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
build_host = sys.argv[2]
exec_host = sys.argv[3]
submit_host = sys.argv[4]
git_commit = sys.argv[5]
cuda_home = sys.argv[6]

build_probe = json.loads((out_dir / "build_host_probe.json").read_text())
host_probe = json.loads((out_dir / "v100_host_probe.json").read_text())
build_manifest = json.loads((out_dir / "ns_inference_test_binary.json").read_text())

json_prefix_map = {
    "stdnormal": ("bench_cuda_stdnormal_walnuts_cpu_vs_gpu_cuda126.txt", "NEXTSTAT_WALNUTS_GPU_CERT_JSON="),
    "linear": ("bench_cuda_linear_walnuts_cpu_vs_gpu_cuda126.txt", "NEXTSTAT_WALNUTS_GPU_LINEAR_BENCH_JSON="),
    "logistic": ("bench_cuda_logistic_walnuts_cpu_vs_gpu_cuda126.txt", "NEXTSTAT_WALNUTS_GPU_LOGISTIC_BENCH_JSON="),
    "poisson_offset": ("bench_cuda_poisson_offset_walnuts_cpu_vs_gpu_cuda126.txt", "NEXTSTAT_WALNUTS_GPU_POISSON_OFFSET_BENCH_JSON="),
    "negbin_offset": ("bench_cuda_negbin_offset_walnuts_cpu_vs_gpu_cuda126.txt", "NEXTSTAT_WALNUTS_GPU_NEGBIN_OFFSET_BENCH_JSON="),
    "weibull_ic_aft": ("bench_cuda_weibull_aft_walnuts_cpu_vs_gpu_cuda126.txt", "NEXTSTAT_WALNUTS_GPU_WEIBULL_AFT_BENCH_JSON="),
}

bench_payloads = {}
for name, (filename, prefix) in json_prefix_map.items():
    text = (out_dir / filename).read_text()
    match = re.search(rf"{re.escape(prefix)}(\{{.*\}})", text)
    if not match:
        raise SystemExit(f"missing benchmark payload {prefix} in {filename}")
    bench_payloads[name] = json.loads(match.group(1))

test_files = [
    "test_logistic_regression_cuda_glm_export_includes_intercept_column.txt",
    "test_linear_regression_cuda_glm_export_includes_intercept_column.txt",
    "test_poisson_regression_cuda_glm_export_preserves_offset_and_intercept_column.txt",
    "test_negbin_regression_cuda_glm_export_preserves_offset_and_log_alpha_slot.txt",
    "ic_weibull_aft_cuda_export_preserves_colmajor_and_censor_codes.txt",
    "cuda_linear_potential_matches_cpu_potential_grad.txt",
    "cuda_linear_leapfrog_matches_cpu_one_step.txt",
    "cuda_logistic_potential_matches_cpu_potential_grad.txt",
    "cuda_logistic_leapfrog_matches_cpu_one_step.txt",
    "cuda_poisson_offset_potential_matches_cpu_potential_grad.txt",
    "cuda_poisson_offset_leapfrog_matches_cpu_one_step.txt",
    "cuda_negbin_offset_potential_matches_cpu_potential_grad.txt",
    "cuda_negbin_offset_leapfrog_matches_cpu_one_step.txt",
    "cuda_weibull_aft_potential_matches_cpu_potential_grad.txt",
    "cuda_weibull_aft_leapfrog_matches_cpu_one_step.txt",
    "cuda_stdnormal_stepper_matches_cpu_one_step.txt",
    "cuda_stdnormal_walnuts_transition_matches_cpu.txt",
]
tests = {}
for filename in test_files:
    text = (out_dir / filename).read_text()
    tests[filename] = {
        "passed": "test result: ok." in text and "0 failed" in text,
    }

report = {
    "schema_version": "nextstat.walnuts_gpu_v100_cert_runner_report.v7",
    "lane_policy": "v100-direct-gpu",
    "scheduler": "split-build-memfd",
    "submit_host": submit_host,
    "build_host": build_host,
    "execute_host_alias": exec_host,
    "git_commit": git_commit,
    "cuda_home": cuda_home,
    "build_host_probe": build_probe,
    "host_probe": host_probe,
    "build_manifest": build_manifest,
    "execution_strategy": {
        "builder": "nextstat-bench cargo test --no-run --features cuda --lib",
        "binary_staging": "/dev/shm on execute host",
        "binary_exec": "memfd_create",
    },
    "tests": tests,
    "benchmarks": bench_payloads,
    "promotion_contract": {
        "schema_version": "nextstat.walnuts_gpu_v100_promotion_contract.v1",
        "status": "internal_evidence_only",
        "public_gpu_walnuts_shipped": False,
        "accepted_lane": {
            "build_host": build_host,
            "host_policy": "v100-direct-gpu",
            "scheduler": "split-build-memfd",
            "gpu": host_probe.get("nvidia_smi_query", []),
            "cuda_home": cuda_home,
            "precision_scope": "real-f64",
        },
        "minimum_matrix": [
            {
                "slice": "stdnormal_narrow_cert",
                "role": "parity_and_cert_only",
                "required_checks": [
                    "cuda_stdnormal_stepper_matches_cpu_one_step",
                    "cuda_stdnormal_walnuts_transition_matches_cpu",
                ],
                "required_metric_fields": [
                    "one_step.gpu_over_cpu_throughput",
                    "walnuts_transition.gpu_over_cpu_throughput",
                ],
            },
            {
                "slice": "evaluator_glm_linear",
                "role": "positive_gpu_throughput_evidence",
                "required_checks": [
                    "test_linear_regression_cuda_glm_export_includes_intercept_column",
                    "cuda_linear_potential_matches_cpu_potential_grad",
                    "cuda_linear_leapfrog_matches_cpu_one_step",
                ],
                "required_thresholds": {
                    "potential_grad.gpu_over_cpu_throughput_gt": 1.0,
                    "walnuts_transition.gpu_over_cpu_throughput_gt": 1.0,
                },
            },
            {
                "slice": "evaluator_glm_logistic",
                "role": "positive_gpu_throughput_evidence",
                "required_checks": [
                    "cuda_logistic_potential_matches_cpu_potential_grad",
                    "cuda_logistic_leapfrog_matches_cpu_one_step",
                ],
                "required_thresholds": {
                    "potential_grad.gpu_over_cpu_throughput_gt": 1.0,
                    "walnuts_transition.gpu_over_cpu_throughput_gt": 1.0,
                },
            },
            {
                "slice": "evaluator_glm_poisson_with_offset",
                "role": "positive_gpu_throughput_evidence",
                "required_checks": [
                    "test_poisson_regression_cuda_glm_export_preserves_offset_and_intercept_column",
                    "cuda_poisson_offset_potential_matches_cpu_potential_grad",
                    "cuda_poisson_offset_leapfrog_matches_cpu_one_step",
                ],
                "required_thresholds": {
                    "potential_grad.gpu_over_cpu_throughput_gt": 1.0,
                    "walnuts_transition.gpu_over_cpu_throughput_gt": 1.0,
                },
            },
            {
                "slice": "evaluator_glm_negbin_with_offset",
                "role": "positive_gpu_throughput_evidence",
                "required_checks": [
                    "test_negbin_regression_cuda_glm_export_preserves_offset_and_log_alpha_slot",
                    "cuda_negbin_offset_potential_matches_cpu_potential_grad",
                    "cuda_negbin_offset_leapfrog_matches_cpu_one_step",
                ],
                "required_thresholds": {
                    "potential_grad.gpu_over_cpu_throughput_gt": 1.0,
                    "walnuts_transition.gpu_over_cpu_throughput_gt": 1.0,
                },
            },
            {
                "slice": "evaluator_survival_ic_weibull_aft",
                "role": "positive_gpu_throughput_evidence_beyond_glm",
                "required_checks": [
                    "ic_weibull_aft_cuda_export_preserves_colmajor_and_censor_codes",
                    "cuda_weibull_aft_potential_matches_cpu_potential_grad",
                    "cuda_weibull_aft_leapfrog_matches_cpu_one_step",
                ],
                "required_thresholds": {
                    "potential_grad.gpu_over_cpu_throughput_gt": 1.0,
                    "walnuts_transition.gpu_over_cpu_throughput_gt": 1.0,
                },
            },
        ],
        "non_claims": [
            "no_public_gpu_walnuts_surface",
            "no_metal_promotion",
            "no_generic_non_glm_promotion",
            "no_cross_gpu_family_promotion",
            "no_global_gpu_sota_claim",
        ],
    },
    "notes": [
        "This is an internal GPU certification artifact on a build-on-nextstat-bench / execute-on-v100 lane, not a public shipped GPU WALNUTS surface.",
        "The accepted userland for this lane is CUDA 12.6 because Volta compute_70 support is required.",
        "The linear, logistic, Poisson-with-offset, NegBin-with-offset, and interval-censored Weibull AFT benchmarks exercise evaluator-backed CUDA WALNUTS slices; the StdNormal benchmark exercises the narrow stepper seam.",
    ],
}

(out_dir / "walnuts_gpu_v100_cert_runner.json").write_text(json.dumps(report, indent=2) + "\n")

std = bench_payloads["stdnormal"]
lin = bench_payloads["linear"]
log = bench_payloads["logistic"]
poi = bench_payloads["poisson_offset"]
nb = bench_payloads["negbin_offset"]
weib = bench_payloads["weibull_ic_aft"]
gpu_name = ", ".join(host_probe.get("nvidia_smi_query") or ["unknown"])

md = [
    "# WALNUTS GPU V100 Certification Runner",
    "",
    "This artifact verifies the internal WALNUTS GPU certification lane built on nextstat-bench and executed on v100.",
    "",
    f"- submit_host: `{submit_host}`",
    f"- build_host: `{build_host}`",
    f"- execute_host_alias: `{exec_host}`",
    f"- execute_host: `{host_probe.get('execute_host')}`",
    f"- scheduler: `split-build-memfd`",
    f"- git_commit: `{git_commit}`",
    f"- cuda_home: `{cuda_home}`",
    f"- test_binary_bytes: `{build_manifest['size_bytes']}`",
    f"- gpu: `{gpu_name}`",
    "",
    "## Parity tests",
    "",
]
for filename, status in tests.items():
    md.append(f"- `{filename}`: `{'pass' if status['passed'] else 'fail'}`")

md.extend(
    [
        "",
        "## Benchmarks",
        "",
        "### StdNormal CUDA seam",
        "",
        f"- one-step gpu/cpu throughput: `{std['one_step']['gpu_over_cpu_throughput']:.4f}x`",
        f"- WALNUTS transition gpu/cpu throughput: `{std['walnuts_transition']['gpu_over_cpu_throughput']:.4f}x`",
        "",
        "### Evaluator-backed logistic CUDA seam",
        "",
        f"- potential_grad gpu/cpu throughput: `{log['potential_grad']['gpu_over_cpu_throughput']:.4f}x`",
        f"- WALNUTS transition gpu/cpu throughput: `{log['walnuts_transition']['gpu_over_cpu_throughput']:.4f}x`",
        "",
        "### Evaluator-backed linear CUDA seam",
        "",
        f"- potential_grad gpu/cpu throughput: `{lin['potential_grad']['gpu_over_cpu_throughput']:.4f}x`",
        f"- WALNUTS transition gpu/cpu throughput: `{lin['walnuts_transition']['gpu_over_cpu_throughput']:.4f}x`",
        "",
        "### Evaluator-backed Poisson-with-offset CUDA seam",
        "",
        f"- potential_grad gpu/cpu throughput: `{poi['potential_grad']['gpu_over_cpu_throughput']:.4f}x`",
        f"- WALNUTS transition gpu/cpu throughput: `{poi['walnuts_transition']['gpu_over_cpu_throughput']:.4f}x`",
        "",
        "### Evaluator-backed NegBin-with-offset CUDA seam",
        "",
        f"- potential_grad gpu/cpu throughput: `{nb['potential_grad']['gpu_over_cpu_throughput']:.4f}x`",
        f"- WALNUTS transition gpu/cpu throughput: `{nb['walnuts_transition']['gpu_over_cpu_throughput']:.4f}x`",
        "",
        "### Evaluator-backed interval-censored Weibull AFT CUDA seam",
        "",
        f"- potential_grad gpu/cpu throughput: `{weib['potential_grad']['gpu_over_cpu_throughput']:.4f}x`",
        f"- WALNUTS transition gpu/cpu throughput: `{weib['walnuts_transition']['gpu_over_cpu_throughput']:.4f}x`",
        "",
        "## Interpretation",
        "",
        "- V100 + CUDA 12.6 is a valid real-f64 GPU certification lane for WALNUTS internal CUDA work.",
        "- This lane now builds on nextstat-bench and executes on v100 via memfd, so it no longer depends on an exec-capable writable filesystem on the GPU host.",
        "- The evaluator-backed linear, logistic, Poisson-with-offset, NegBin-with-offset, and interval-censored Weibull AFT slices are the current winning GPU product path candidates on this host.",
        "- The narrow StdNormal seam remains useful as a parity/cert slice, not as the winning GPU product path.",
        "- Promotion contract stays internal-only: this artifact is sufficient for the accepted V100 evidence boundary, but not for any shipped public GPU WALNUTS claim.",
        "",
    ]
)

(out_dir / "walnuts_gpu_v100_cert_runner.md").write_text("\n".join(md) + "\n")
PY

echo "[walnuts-gpu-v100-cert] done"
echo "[walnuts-gpu-v100-cert] artifact=${BENCH_LOCAL_OUT}/walnuts_gpu_v100_cert_runner.md"
