#!/usr/bin/env bash
set -euo pipefail

# Internal HTCondor-backed GPU-lane preflight for WALNUTS.
#
# Submit host remains `nextstat-bench`; execution happens on any HTCondor node
# that satisfies the GPU requirement. This script proves the lane exists and
# records the execute-node contract without pretending to certify sampler
# quality.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

BENCH_HOST="${BENCH_HOST:-nextstat-bench}"
BENCH_SSH_USER="${BENCH_SSH_USER:-}"
BENCH_SSH_PORT="${BENCH_SSH_PORT:-}"
BENCH_SSH_KEY="${BENCH_SSH_KEY:-}"
BENCH_REMOTE_BASE="${BENCH_REMOTE_BASE:-/home/actions-runner}"
BENCH_REMOTE_RUN="${BENCH_REMOTE_RUN:-${BENCH_REMOTE_BASE}/bench_walnuts_gpu_lane_preflight_${STAMP}}"
BENCH_LOCAL_OUT="${BENCH_LOCAL_OUT:-$ROOT_DIR/tmp/bench_walnuts_gpu_lane_preflight_${STAMP}/${BENCH_HOST}}"
BENCH_REQUEST_GPUS="${BENCH_REQUEST_GPUS:-1}"
BENCH_REQUIREMENTS="${BENCH_REQUIREMENTS:-(GPUs >= 1)}"
BENCH_POLL_SECS="${BENCH_POLL_SECS:-5}"
BENCH_TIMEOUT_SECS="${BENCH_TIMEOUT_SECS:-600}"

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

echo "[walnuts-gpu-preflight] host=${REMOTE_SPEC}"
echo "[walnuts-gpu-preflight] remote_run=${BENCH_REMOTE_RUN}"
echo "[walnuts-gpu-preflight] local_out=${BENCH_LOCAL_OUT}"
echo "[walnuts-gpu-preflight] request_gpus=${BENCH_REQUEST_GPUS}"
echo "[walnuts-gpu-preflight] requirements=${BENCH_REQUIREMENTS}"

echo "[walnuts-gpu-preflight] probe submit host..."
"${SSH_BASE[@]}" "set -euo pipefail; hostname; whoami; python3 --version; condor_status -compact -af Name DetectedGPUs GPUs 2>/dev/null || true"

echo "[walnuts-gpu-preflight] create remote run dir as actions-runner..."
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_RUN}" <<'EOS'
set -euo pipefail
RUN="$1"
sudo -u actions-runner -H mkdir -p "$RUN"
EOS

echo "[walnuts-gpu-preflight] render remote job..."
BENCH_REQUIREMENTS_B64="$(printf '%s' "${BENCH_REQUIREMENTS}" | base64)"
"${SSH_BASE[@]}" bash -s -- "${BENCH_REMOTE_RUN}" "${BENCH_REQUEST_GPUS}" "${BENCH_REQUIREMENTS_B64}" <<'EOS'
set -euo pipefail
RUN="$1"
REQUEST_GPUS="$2"
REQUIREMENTS="$(printf '%s' "$3" | base64 --decode)"
sudo -u actions-runner -H bash -s -- "$RUN" "$REQUEST_GPUS" "$REQUIREMENTS" <<'INNER'
set -euo pipefail
RUN="$1"
REQUEST_GPUS="$2"
REQUIREMENTS="$3"
cat > "$RUN/job.sh" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json
import os
import platform
import shutil
import socket
import subprocess
import sys


def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "output": out.strip()}
    except Exception as exc:
        return {"ok": False, "output": str(exc)}


gpu_list = run(["nvidia-smi", "-L"])
gpu_query = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
payload = {
    "schema_version": "nextstat.walnuts_gpu_lane_preflight.v1",
    "execute_host": socket.gethostname(),
    "execute_user": subprocess.check_output(["whoami"], text=True).strip(),
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    },
    "python_version": sys.version.replace("\n", " "),
    "cargo_path": shutil.which("cargo"),
    "rustc_path": shutil.which("rustc"),
    "nvidia_smi_list": gpu_list["output"].splitlines() if gpu_list["ok"] and gpu_list["output"] else [],
    "nvidia_smi_query": gpu_query["output"].splitlines() if gpu_query["ok"] and gpu_query["output"] else [],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "condor_slot": os.environ.get("_CONDOR_SLOT"),
}
with open("gpu_lane_probe.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
JOB
chmod +x "$RUN/job.sh"
cat > "$RUN/job.sub" <<SUB
universe = vanilla
executable = /bin/bash
arguments = job.sh
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = job.sh
output = job.out
error = job.err
log = job.log
request_gpus = ${REQUEST_GPUS}
requirements = ${REQUIREMENTS}
queue
SUB
INNER
EOS

echo "[walnuts-gpu-preflight] submit job..."
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
  echo "[walnuts-gpu-preflight] failed to parse cluster id" >&2
  exit 1
fi

echo "[walnuts-gpu-preflight] cluster_id=${cluster_id}"
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
    echo "[walnuts-gpu-preflight] job held"
    echo "${queue_status}"
    break
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts > BENCH_TIMEOUT_SECS )); then
    echo "[walnuts-gpu-preflight] timeout waiting for cluster ${cluster_id}" >&2
    "${SSH_BASE[@]}" "sudo -u actions-runner -H condor_rm ${cluster_id} >/dev/null 2>&1 || true"
    exit 1
  fi
  sleep "${BENCH_POLL_SECS}"
done

echo "[walnuts-gpu-preflight] sync artifacts..."
RSYNC_RSH_CMD="${RSYNC_RSH[*]}"
rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_RUN}/" "${BENCH_LOCAL_OUT}/"

echo "[walnuts-gpu-preflight] render local report..."
python3 - <<'PY' "${BENCH_LOCAL_OUT}" "${BENCH_HOST}" "${cluster_id}" "${BENCH_REQUIREMENTS}" "${BENCH_REQUEST_GPUS}" "${final_history}"
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
submit_host = sys.argv[2]
cluster_id = int(sys.argv[3])
requirements = sys.argv[4]
request_gpus = int(sys.argv[5])
history_line = sys.argv[6].strip()

probe_path = out_dir / "gpu_lane_probe.json"
payload = json.loads(probe_path.read_text()) if probe_path.exists() else {}

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

report = {
    "schema_version": "nextstat.walnuts_gpu_lane_preflight_report.v1",
    "submit_host": submit_host,
    "cluster_id": cluster_id,
    "scheduler": "htcondor",
    "request_gpus": request_gpus,
    "requirements": requirements,
    "history": history,
    "probe": payload,
    "notes": [
        "This is an infrastructure preflight, not a sampler-quality artifact.",
        "A green result means nextstat-bench can submit a GPU HTCondor job and collect output.",
        "If cargo/rustc are absent on the execute node, the future WALNUTS GPU lane must transfer prebuilt artifacts from the submit host.",
    ],
}

(out_dir / "walnuts_gpu_lane_preflight.json").write_text(json.dumps(report, indent=2) + "\n")

gpu_lines = payload.get("nvidia_smi_list") or []
gpu_query = payload.get("nvidia_smi_query") or []
history_remote = history.get("last_remote_host") if history else None
execute_host = payload.get("execute_host")
md = [
    "# WALNUTS GPU Lane Preflight",
    "",
    "This artifact verifies infrastructure only. It is not a sampler benchmark.",
    "",
    f"- submit_host: `{submit_host}`",
    f"- execute_host: `{execute_host}`",
    f"- condor_last_remote_host: `{history_remote}`",
    f"- cluster_id: `{cluster_id}`",
    f"- scheduler: `htcondor`",
    f"- request_gpus: `{request_gpus}`",
    f"- requirements: `{requirements}`",
    "",
    "## Execute Node",
    "",
    f"- user: `{payload.get('execute_user')}`",
    f"- python: `{payload.get('python_version')}`",
    f"- cargo_path: `{payload.get('cargo_path')}`",
    f"- rustc_path: `{payload.get('rustc_path')}`",
    f"- cuda_visible_devices: `{payload.get('cuda_visible_devices')}`",
    f"- condor_slot: `{payload.get('condor_slot')}`",
    "",
    "## GPUs",
    "",
]
if gpu_lines:
    md.extend([f"- `{line}`" for line in gpu_lines])
else:
    md.append("- `nvidia-smi -L` returned no GPUs")
md.extend(["", "## GPU Query", ""])
if gpu_query:
    md.extend([f"- `{line}`" for line in gpu_query])
else:
    md.append("- no GPU query rows")
md.extend(
    [
        "",
        "## Verdict",
        "",
        "- HTCondor GPU execute path is reachable from `nextstat-bench`." if execute_host else "- HTCondor GPU execute path did not return an execute host.",
        "- Execute node currently lacks `cargo`/`rustc`; GPU benchmark jobs must use transfer-built artifacts." if not payload.get("cargo_path") or not payload.get("rustc_path") else "- Execute node has Rust toolchain available.",
    ]
)
(out_dir / "walnuts_gpu_lane_preflight.md").write_text("\n".join(md) + "\n")
PY

echo "[walnuts-gpu-preflight] done: ${BENCH_LOCAL_OUT}"
