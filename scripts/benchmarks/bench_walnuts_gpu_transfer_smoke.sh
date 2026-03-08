#!/usr/bin/env bash
set -euo pipefail

# Internal HTCondor-backed transfer-built GPU runner for WALNUTS.
#
# Default profile (`smoke`) proves a real CUDA-backed WALNUTS seam can run on an
# HTCondor GPU execute node even when the submit host lacks nvcc.
# Alternate profile (`cert`) promotes the same transfer-built path into a narrow
# internal GPU certification runner for the StdNormal CUDA seam.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"

BENCH_PROFILE="${BENCH_PROFILE:-smoke}"
case "${BENCH_PROFILE}" in
  smoke)
    LOG_PREFIX="walnuts-gpu-transfer-smoke"
    REPORT_STEM="walnuts_gpu_transfer_smoke"
    REPORT_TITLE="WALNUTS GPU Transfer Smoke"
    REPORT_INTRO="This artifact verifies a transfer-built internal CUDA seam slice. It is not a sampler-quality benchmark."
    EXECUTE_PROBE_NAME="gpu_transfer_smoke_probe.json"
    EXECUTE_PROBE_SCHEMA="nextstat.walnuts_gpu_transfer_smoke_probe.v1"
    REPORT_SCHEMA="nextstat.walnuts_gpu_transfer_smoke_report.v1"
    TEST_MATRIX_JSON='[
      {
        "name": "cuda_hmc_stepper::tests::cuda_stdnormal_stepper_matches_cpu_one_step",
        "output": "cuda_stdnormal_stepper_matches_cpu_one_step.txt",
        "ignored": false
      },
      {
        "name": "cuda_hmc_stepper::tests::cuda_stdnormal_walnuts_transition_matches_cpu",
        "output": "cuda_stdnormal_walnuts_transition_matches_cpu.txt",
        "ignored": false
      }
    ]'
    REPORT_NOTES_JSON='[
      "This is an internal GPU execution smoke, not a product benchmark artifact.",
      "The submit host build intentionally uses NS_COMPUTE_FORCE_STUB_PTX=1 because nextstat-bench lacks nvcc.",
      "A green result means the execute node successfully JIT-compiled the narrow StdNormal kernel via NVRTC and passed the CUDA seam parity smoke."
    ]'
    REPORT_VERDICT_JSON='[
      "Transfer-built WALNUTS CUDA smoke is viable from `nextstat-bench` via HTCondor.",
      "Submit host still lacks `nvcc`; the narrow StdNormal kernel is compiled on the execute node via NVRTC.",
      "This is internal seam evidence only; it is not a GPU benchmark certification artifact."
    ]'
    DEFAULT_CARGO_FILTER="cuda_stdnormal"
    ;;
  cert)
    LOG_PREFIX="walnuts-gpu-cert-runner"
    REPORT_STEM="walnuts_gpu_cert_runner"
    REPORT_TITLE="WALNUTS GPU Certification Runner"
    REPORT_INTRO="This artifact verifies the internal GPU certification runner for the narrow WALNUTS CUDA seam. It is not a public sampler benchmark."
    EXECUTE_PROBE_NAME="gpu_cert_runner_probe.json"
    EXECUTE_PROBE_SCHEMA="nextstat.walnuts_gpu_cert_runner_probe.v1"
    REPORT_SCHEMA="nextstat.walnuts_gpu_cert_runner_report.v1"
    TEST_MATRIX_JSON='[
      {
        "name": "cuda_hmc_stepper::tests::bench_cuda_stdnormal_walnuts_cpu_vs_gpu",
        "output": "bench_cuda_stdnormal_walnuts_cpu_vs_gpu.txt",
        "ignored": true,
        "extract_json_prefix": "NEXTSTAT_WALNUTS_GPU_CERT_JSON="
      }
    ]'
    REPORT_NOTES_JSON='[
      "This is an internal GPU certification runner artifact for the narrow StdNormal CUDA seam, not a public sampler benchmark.",
      "The submit host build intentionally uses NS_COMPUTE_FORCE_STUB_PTX=1 because nextstat-bench lacks nvcc.",
      "A green result means the execute node successfully JIT-compiled the narrow StdNormal kernel via NVRTC and emitted the internal CUDA benchmark payload."
    ]'
    REPORT_VERDICT_JSON='[
      "Transfer-built internal GPU benchmark runner is viable from `nextstat-bench` via HTCondor.",
      "Current certification scope is narrow: StdNormal target, diagonal metric, and internal CUDA seam only.",
      "Public GPU SOTA still requires a broader benchmark contract and sampler surface."
    ]'
    DEFAULT_CARGO_FILTER="cuda_stdnormal"
    ;;
  *)
    echo "Unknown BENCH_PROFILE=${BENCH_PROFILE}; expected smoke or cert" >&2
    exit 1
    ;;
esac

TEST_MATRIX_JSON="${BENCH_TEST_MATRIX_JSON:-${TEST_MATRIX_JSON}}"
TEST_MATRIX_B64="$(printf '%s' "${TEST_MATRIX_JSON}" | base64)"
REPORT_NOTES_B64="$(printf '%s' "${REPORT_NOTES_JSON}" | base64)"
REPORT_VERDICT_B64="$(printf '%s' "${REPORT_VERDICT_JSON}" | base64)"

BENCH_HOST="${BENCH_HOST:-nextstat-bench}"
BENCH_SSH_USER="${BENCH_SSH_USER:-}"
BENCH_SSH_PORT="${BENCH_SSH_PORT:-}"
BENCH_SSH_KEY="${BENCH_SSH_KEY:-}"
BENCH_REMOTE_REPO="${BENCH_REMOTE_REPO:-/tmp/nextstat_walnuts_cuda_gpu_smoke_repo}"
BENCH_REMOTE_TARGET="${BENCH_REMOTE_TARGET:-/tmp/nextstat_walnuts_cuda_gpu_smoke_target}"
BENCH_REMOTE_BASE="${BENCH_REMOTE_BASE:-/home/actions-runner}"
BENCH_REMOTE_RUN="${BENCH_REMOTE_RUN:-${BENCH_REMOTE_BASE}/bench_${REPORT_STEM}_${STAMP}}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-$ROOT_DIR/tmp/${REPORT_STEM}_${STAMP}/${BENCH_HOST}}"
BENCH_REQUEST_GPUS="${BENCH_REQUEST_GPUS:-1}"
BENCH_REQUIREMENTS="${BENCH_REQUIREMENTS:-(GPUs >= 1)}"
BENCH_POLL_SECS="${BENCH_POLL_SECS:-5}"
BENCH_TIMEOUT_SECS="${BENCH_TIMEOUT_SECS:-1800}"
BENCH_CARGO_FILTER="${BENCH_CARGO_FILTER:-${DEFAULT_CARGO_FILTER}}"

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

echo "[${LOG_PREFIX}] profile=${BENCH_PROFILE}"
echo "[${LOG_PREFIX}] host=${REMOTE_SPEC}"
echo "[${LOG_PREFIX}] remote_repo=${BENCH_REMOTE_REPO}"
echo "[${LOG_PREFIX}] remote_target=${BENCH_REMOTE_TARGET}"
echo "[${LOG_PREFIX}] remote_run=${BENCH_REMOTE_RUN}"
echo "[${LOG_PREFIX}] local_out=${BENCH_LOCAL_OUT}"
echo "[${LOG_PREFIX}] request_gpus=${BENCH_REQUEST_GPUS}"
echo "[${LOG_PREFIX}] requirements=${BENCH_REQUIREMENTS}"

echo "[${LOG_PREFIX}] probe submit host..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; python3 --version; cargo --version; rustc --version; command -v nvcc || true"

echo "[${LOG_PREFIX}] create remote directories..."
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_REPO}" "${BENCH_REMOTE_TARGET}" "${BENCH_REMOTE_RUN}" <<'EOS'
set -euo pipefail
REPO="$1"
TARGET="$2"
RUN="$3"
mkdir -p "$REPO" "$TARGET"
sudo -u actions-runner -H mkdir -p "$RUN"
EOS

echo "[${LOG_PREFIX}] rsync snapshot..."
RSYNC_RSH_CMD="${RSYNC_RSH[*]}"
rsync -az \
  --delete \
  --rsh="${RSYNC_RSH_CMD}" \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude '.nextstat-cargo-target/' \
  --exclude 'node_modules/' \
  --exclude 'benchmarks/artifacts/' \
  --exclude 'benchmarks/unbinned/artifacts/' \
  --exclude 'benchmarks/nextstat-public-benchmarks/.venv/' \
  --exclude 'bench_results/' \
  --exclude '.venv*/' \
  --exclude 'tmp/' \
  --exclude 'tmp*/' \
  --exclude 'docs/blog/artifacts/' \
  --exclude '**/__pycache__/' \
  --exclude '.DS_Store' \
  "${ROOT_DIR}/" \
  "${REMOTE_SPEC}:${BENCH_REMOTE_REPO}/"

echo "[${LOG_PREFIX}] build transfer binary on submit host..."
"${SSH_BASE[@]}" bash -s -- \
  "${BENCH_REMOTE_REPO}" \
  "${BENCH_REMOTE_TARGET}" \
  "${BENCH_REMOTE_RUN}" \
  "${BENCH_CARGO_FILTER}" \
  "${TEST_MATRIX_B64}" \
  "${EXECUTE_PROBE_NAME}" \
  "${EXECUTE_PROBE_SCHEMA}" <<'EOS'
set -euo pipefail
REPO="$1"
TARGET="$2"
RUN="$3"
CARGO_FILTER="$4"
TEST_MATRIX_B64="$5"
EXECUTE_PROBE_NAME="$6"
EXECUTE_PROBE_SCHEMA="$7"
python3 - <<'PY' "$RUN"
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
payload = {
    "schema_version": "nextstat.walnuts_gpu_transfer_submit_probe.v1",
    "submit_host": socket.gethostname(),
    "python_version": subprocess.check_output(["python3", "--version"], text=True).strip(),
    "cargo_path": shutil.which("cargo"),
    "cargo_version": subprocess.check_output(["cargo", "--version"], text=True).strip(),
    "rustc_path": shutil.which("rustc"),
    "rustc_version": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    "nvcc_path": shutil.which("nvcc"),
}
run_dir.joinpath("submit_host_probe.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
cd "$REPO"
export CARGO_TARGET_DIR="$TARGET"
export NS_COMPUTE_FORCE_STUB_PTX=1
cargo test -p ns-inference "$CARGO_FILTER" --features cuda --lib --no-run --message-format=json-render-diagnostics > "$RUN/build.jsonl"
python3 - <<'PY' "$RUN/build.jsonl" "$RUN/test_binary_path.txt"
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
executable = None
for line in jsonl_path.read_text().splitlines():
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        continue
    if payload.get("reason") != "compiler-artifact":
        continue
    candidate = payload.get("executable")
    target = payload.get("target") or {}
    if candidate and target.get("kind") == ["lib"]:
        executable = candidate
if executable is None:
    raise SystemExit("failed to locate ns-inference lib test executable in cargo JSON output")
out_path.write_text(executable + "\n")
print(executable)
PY
TEST_BIN="$(cat "$RUN/test_binary_path.txt")"
sudo -u actions-runner -H install -m 0755 "$TEST_BIN" "$RUN/ns-inference-cuda-tests"
sudo -u actions-runner -H python3 - <<'PY' "$RUN" "$TEST_MATRIX_B64" "$EXECUTE_PROBE_NAME" "$EXECUTE_PROBE_SCHEMA"
import base64
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
tests = json.loads(base64.b64decode(sys.argv[2]).decode("utf-8"))
probe_name = sys.argv[3]
probe_schema = sys.argv[4]

job_lines = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    f"export TEST_MATRIX_B64='{sys.argv[2]}'",
    f"export EXECUTE_PROBE_NAME='{probe_name}'",
    f"export EXECUTE_PROBE_SCHEMA='{probe_schema}'",
    "python3 - <<'PY'",
    "import base64",
    "import ctypes.util",
    "import json",
    "import os",
    "import platform",
    "import socket",
    "import subprocess",
    "import time",
    "from pathlib import Path",
    "",
    "tests = json.loads(base64.b64decode(os.environ['TEST_MATRIX_B64']).decode('utf-8'))",
    "report = {",
    "    'schema_version': os.environ['EXECUTE_PROBE_SCHEMA'],",
    "    'execute_host': socket.gethostname(),",
    "    'execute_user': subprocess.check_output(['whoami'], text=True).strip(),",
    "    'platform': {",
    "        'system': platform.system(),",
    "        'release': platform.release(),",
    "        'machine': platform.machine(),",
    "    },",
    "    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),",
    "    'condor_slot': os.environ.get('_CONDOR_SLOT'),",
    "    'nvcc_path': subprocess.run(['bash', '-lc', 'command -v nvcc || true'], capture_output=True, text=True).stdout.strip() or None,",
    "    'libnvrtc': ctypes.util.find_library('nvrtc'),",
    "    'libcuda': ctypes.util.find_library('cuda'),",
    "    'nvidia_smi': subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().splitlines(),",
    "    'tests': [],",
    "}",
    "",
    "all_passed = True",
    "for item in tests:",
    "    test_name = item['name']",
    "    output_name = item['output']",
    "    started = time.time()",
    "    cmd = ['./ns-inference-cuda-tests', '--exact', test_name, '--nocapture', '--test-threads=1']",
    "    if item.get('ignored'):",
    "        cmd.insert(3, '--ignored')",
    "    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)",
    "    Path(output_name).write_text(completed.stdout)",
    "    ok = completed.returncode == 0",
    "    all_passed = all_passed and ok",
    "    report['tests'].append({",
    "        'name': test_name,",
    "        'ok': ok,",
    "        'returncode': completed.returncode,",
    "        'duration_s': round(time.time() - started, 6),",
    "        'output_file': output_name,",
    "        'extract_json_prefix': item.get('extract_json_prefix'),",
    "    })",
    "",
    "report['all_passed'] = all_passed",
    "Path(os.environ['EXECUTE_PROBE_NAME']).write_text(json.dumps(report, indent=2, sort_keys=True) + '\\n')",
    "if not all_passed:",
    "    raise SystemExit(1)",
    "PY",
]
(run_dir / "job.sh").write_text("\n".join(job_lines) + "\n")
(run_dir / "job.sh").chmod(0o755)

transfer_output_files = [probe_name] + [item["output"] for item in tests]
submit_lines = [
    "universe = vanilla",
    "executable = /bin/bash",
    "arguments = job.sh",
    "should_transfer_files = YES",
    "when_to_transfer_output = ON_EXIT",
    "transfer_input_files = job.sh,ns-inference-cuda-tests",
    f"transfer_output_files = {','.join(transfer_output_files)}",
    "output = job.out",
    "error = job.err",
    "log = job.log",
    "request_gpus = 1",
    "requirements = (GPUs >= 1)",
    "queue",
]
(run_dir / "job.sub").write_text("\n".join(submit_lines) + "\n")
PY
EOS

echo "[${LOG_PREFIX}] patch job submit config..."
BENCH_REQUIREMENTS_B64="$(printf '%s' "${BENCH_REQUIREMENTS}" | base64)"
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_RUN}" "${BENCH_REQUEST_GPUS}" "${BENCH_REQUIREMENTS_B64}" <<'EOS'
set -euo pipefail
RUN="$1"
REQUEST_GPUS="$2"
REQUIREMENTS="$(printf '%s' "$3" | base64 --decode)"
sudo -u actions-runner -H python3 - <<'PY' "$RUN/job.sub" "$REQUEST_GPUS" "$REQUIREMENTS"
from pathlib import Path
import sys

path = Path(sys.argv[1])
request_gpus = sys.argv[2]
requirements = sys.argv[3]
text = path.read_text()
text = text.replace("request_gpus = 1", f"request_gpus = {request_gpus}")
text = text.replace("requirements = (GPUs >= 1)", f"requirements = {requirements}")
path.write_text(text)
PY
EOS

echo "[${LOG_PREFIX}] submit job..."
submit_output="$("${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_RUN}" <<'EOS'
set -euo pipefail
RUN="$1"
sudo -u actions-runner -H bash -s -- "$RUN" <<'INNER'
set -euo pipefail
RUN="$1"
cd "$RUN"
condor_submit job.sub
INNER
EOS
)"
printf '%s\n' "$submit_output"
cluster_id="$(printf '%s\n' "$submit_output" | sed -n 's/.*cluster \([0-9][0-9]*\).*/\1/p' | tail -n1)"
if [[ -z "${cluster_id}" ]]; then
  echo "[${LOG_PREFIX}] failed to parse cluster id" >&2
  exit 1
fi

echo "[${LOG_PREFIX}] cluster_id=${cluster_id}"
start_ts="$(date +%s)"
final_history=""
while true; do
  queue_status="$("${SSH_BASE[@]}" "condor_q ${cluster_id} -autoformat JobStatus HoldReason LastRemoteHost 2>/dev/null || true")"
  if [[ -z "${queue_status}" ]]; then
    final_history="$("${SSH_BASE[@]}" "condor_history ${cluster_id} -limit 1 -autoformat ClusterId ProcId JobStatus LastRemoteHost ExitCode 2>/dev/null || true")"
    break
  fi
  job_status="$(printf '%s\n' "${queue_status}" | awk 'NR==1 {print $1}')"
  if [[ "${job_status}" == "5" ]]; then
    echo "[${LOG_PREFIX}] job held"
    echo "${queue_status}"
    exit 1
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts > BENCH_TIMEOUT_SECS )); then
    echo "[${LOG_PREFIX}] timeout waiting for cluster ${cluster_id}" >&2
    "${SSH_BASE[@]}" "sudo -u actions-runner -H condor_rm ${cluster_id} >/dev/null 2>&1 || true"
    exit 1
  fi
  sleep "${BENCH_POLL_SECS}"
done

echo "[${LOG_PREFIX}] sync artifacts..."
rsync -az \
  --rsh="${RSYNC_RSH_CMD}" \
  --include '*.json' \
  --include '*.txt' \
  --include 'build.jsonl' \
  --include 'job.out' \
  --include 'job.err' \
  --include 'job.log' \
  --include 'job.sh' \
  --include 'job.sub' \
  --exclude '*' \
  "${REMOTE_SPEC}:${BENCH_REMOTE_RUN}/" \
  "${BENCH_LOCAL_OUT}/"

echo "[${LOG_PREFIX}] render local report..."
python3 - <<'PY' \
  "${BENCH_LOCAL_OUT}" \
  "${BENCH_HOST}" \
  "${cluster_id}" \
  "${BENCH_REQUIREMENTS}" \
  "${BENCH_REQUEST_GPUS}" \
  "${LOCAL_GIT_COMMIT}" \
  "${final_history}" \
  "${BENCH_PROFILE}" \
  "${EXECUTE_PROBE_NAME}" \
  "${REPORT_SCHEMA}" \
  "${REPORT_STEM}" \
  "${REPORT_TITLE}" \
  "${REPORT_INTRO}" \
  "${REPORT_NOTES_B64}" \
  "${REPORT_VERDICT_B64}"
import base64
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
submit_host = sys.argv[2]
cluster_id = int(sys.argv[3])
requirements = sys.argv[4]
request_gpus = int(sys.argv[5])
git_commit = sys.argv[6]
history_line = sys.argv[7].strip()
profile = sys.argv[8]
execute_probe_name = sys.argv[9]
report_schema = sys.argv[10]
report_stem = sys.argv[11]
report_title = sys.argv[12]
report_intro = sys.argv[13]
report_notes = json.loads(base64.b64decode(sys.argv[14]).decode("utf-8"))
report_verdict = json.loads(base64.b64decode(sys.argv[15]).decode("utf-8"))

submit_probe = json.loads((out_dir / "submit_host_probe.json").read_text())
execute_probe = json.loads((out_dir / execute_probe_name).read_text())

history = None
if history_line:
    parts = history_line.split(maxsplit=4)
    history = {
        "cluster_id": int(parts[0]) if len(parts) > 0 else cluster_id,
        "proc_id": int(parts[1]) if len(parts) > 1 else 0,
        "job_status": int(parts[2]) if len(parts) > 2 else None,
        "last_remote_host": parts[3] if len(parts) > 3 else None,
        "exit_code": int(parts[4]) if len(parts) > 4 else None,
    }

benchmark_payloads = []
for item in execute_probe.get("tests", []):
    prefix = item.get("extract_json_prefix")
    output_name = item.get("output_file")
    if not prefix or not output_name:
        continue
    output_path = out_dir / output_name
    if not output_path.exists():
        continue
    for line in output_path.read_text().splitlines():
        idx = line.find(prefix)
        if idx >= 0:
            payload = json.loads(line[idx + len(prefix):])
            benchmark_payloads.append({"test_name": item["name"], "report": payload})
            item["benchmark_report_schema"] = payload.get("schema_version")
            break

report = {
    "schema_version": report_schema,
    "profile": profile,
    "git_commit": git_commit,
    "submit_host": submit_host,
    "cluster_id": cluster_id,
    "scheduler": "htcondor",
    "request_gpus": request_gpus,
    "requirements": requirements,
    "build_mode": "submit_host_stub_ptx_plus_execute_node_nvrtc",
    "history": history,
    "submit_probe": submit_probe,
    "execute_probe": execute_probe,
    "benchmark_payloads": benchmark_payloads,
    "notes": report_notes,
}
out_dir.joinpath(f"{report_stem}.json").write_text(json.dumps(report, indent=2) + "\n")

md = [
    f"# {report_title}",
    "",
    report_intro,
    "",
    f"- git_commit: `{git_commit}`",
    f"- submit_host: `{submit_host}`",
    f"- execute_host: `{execute_probe.get('execute_host')}`",
    f"- condor_last_remote_host: `{history.get('last_remote_host') if history else None}`",
    f"- cluster_id: `{cluster_id}`",
    f"- scheduler: `htcondor`",
    f"- request_gpus: `{request_gpus}`",
    f"- requirements: `{requirements}`",
    f"- build_mode: `submit_host_stub_ptx_plus_execute_node_nvrtc`",
    "",
    "## Submit Host",
    "",
    f"- cargo: `{submit_probe.get('cargo_version')}`",
    f"- rustc: `{submit_probe.get('rustc_version')}`",
    f"- nvcc_path: `{submit_probe.get('nvcc_path')}`",
    "",
    "## Execute Node",
    "",
    f"- user: `{execute_probe.get('execute_user')}`",
    f"- condor_slot: `{execute_probe.get('condor_slot')}`",
    f"- cuda_visible_devices: `{execute_probe.get('cuda_visible_devices')}`",
    f"- nvcc_path: `{execute_probe.get('nvcc_path')}`",
    f"- libnvrtc: `{execute_probe.get('libnvrtc')}`",
    "",
    "## GPU Query",
    "",
]
gpu_lines = execute_probe.get("nvidia_smi") or []
if gpu_lines:
    md.extend([f"- `{line}`" for line in gpu_lines])
else:
    md.append("- `nvidia-smi` returned no GPU lines")

md.extend(["", "## Tests", ""])
for item in execute_probe.get("tests", []):
    line = (
        f"- `{item['name']}`: ok=`{item['ok']}`, returncode=`{item['returncode']}`, "
        f"duration_s=`{item['duration_s']}`, output=`{item['output_file']}`"
    )
    if item.get("benchmark_report_schema"):
        line += f", benchmark_schema=`{item['benchmark_report_schema']}`"
    md.append(line)

if benchmark_payloads:
    md.extend(["", "## Benchmarks", ""])
    for entry in benchmark_payloads:
        payload = entry["report"]
        one_step = payload.get("one_step", {})
        transition = payload.get("walnuts_transition", {})
        md.append(f"- `{entry['test_name']}` schema=`{payload.get('schema_version')}`")
        if one_step:
            md.append(
                f"- one_step: cpu_steps_per_sec=`{one_step['cpu']['steps_per_sec']:.2f}`, "
                f"gpu_steps_per_sec=`{one_step['gpu']['steps_per_sec']:.2f}`, "
                f"gpu_over_cpu=`{one_step['gpu_over_cpu_throughput']:.4f}`"
            )
        if transition:
            md.append(
                f"- walnuts_transition: cpu_transitions_per_sec=`{transition['cpu']['transitions_per_sec']:.2f}`, "
                f"gpu_transitions_per_sec=`{transition['gpu']['transitions_per_sec']:.2f}`, "
                f"gpu_over_cpu=`{transition['gpu_over_cpu_throughput']:.4f}`"
            )
            md.append(
                f"- walnuts_transition LF/s: cpu=`{transition['cpu']['leapfrogs_per_sec']:.2f}`, "
                f"gpu=`{transition['gpu']['leapfrogs_per_sec']:.2f}`"
            )

md.extend(["", "## Verdict", ""])
md.extend([f"- {line}" for line in report_verdict])
out_dir.joinpath(f"{report_stem}.md").write_text("\n".join(md) + "\n")
PY

echo "[${LOG_PREFIX}] done: ${BENCH_LOCAL_OUT}"
