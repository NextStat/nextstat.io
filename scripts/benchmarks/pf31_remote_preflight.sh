#!/usr/bin/env bash
set -euo pipefail

# PF3.1 remote CUDA preflight for expensive multi-GPU stands.
# Usage:
#   PF31_HOST=<host> PF31_PORT=<port> PF31_KEY=~/.ssh/<key> \
#   bash scripts/benchmarks/pf31_remote_preflight.sh
#
# Optional overrides:
#   PF31_USER, PF31_REMOTE_REPO, PF31_BIN, PF31_SPEC
#   PF31_MIN_GPUS, PF31_DEVICE_SETS, PF31_SMOKE_TOYS, PF31_TIMEOUT
#   PF31_LOCAL_DIR, PF31_REMOTE_DIR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PF31_HOST="${PF31_HOST:-88.198.23.172}"
PF31_PORT="${PF31_PORT:-22}"
PF31_USER="${PF31_USER:-root}"
PF31_KEY="${PF31_KEY:-$HOME/.ssh/rundesk_hetzner}"
PF31_REMOTE_REPO="${PF31_REMOTE_REPO:-/root/nextstat.io}"
PF31_BIN="${PF31_BIN:-$PF31_REMOTE_REPO/target/release/nextstat}"
PF31_SPEC="${PF31_SPEC:-$PF31_REMOTE_REPO/benchmarks/unbinned/specs/pf31_gauss_exp_10k.json}"
PF31_MIN_GPUS="${PF31_MIN_GPUS:-2}"
PF31_DEVICE_SETS="${PF31_DEVICE_SETS:-}"
PF31_SMOKE_TOYS="${PF31_SMOKE_TOYS:-32}"
PF31_TIMEOUT="${PF31_TIMEOUT:-600}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DATE_TAG="$(date -u +%Y-%m-%d)"
PF31_LOCAL_DIR="${PF31_LOCAL_DIR:-$ROOT_DIR/benchmarks/unbinned/artifacts/${DATE_TAG}/pf31_preflight_${STAMP}}"
PF31_REMOTE_DIR="${PF31_REMOTE_DIR:-/tmp/pf31_preflight_${STAMP}}"

SSH_BASE=(ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i "$PF31_KEY" -p "$PF31_PORT" "$PF31_USER@$PF31_HOST")

mkdir -p "$PF31_LOCAL_DIR"

{
  echo "utc_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$PF31_USER@$PF31_HOST:$PF31_PORT"
  echo "remote_repo=$PF31_REMOTE_REPO"
  echo "remote_bin=$PF31_BIN"
  echo "remote_spec=$PF31_SPEC"
  echo "remote_dir=$PF31_REMOTE_DIR"
  echo "min_gpus=$PF31_MIN_GPUS"
  echo "smoke_toys=$PF31_SMOKE_TOYS"
  echo "timeout_s=$PF31_TIMEOUT"
} > "$PF31_LOCAL_DIR/run.meta"

echo "[pf31-preflight] host=${PF31_USER}@${PF31_HOST}:${PF31_PORT}"
echo "[pf31-preflight] local_dir=${PF31_LOCAL_DIR}"
echo "[pf31-preflight] remote_dir=${PF31_REMOTE_DIR}"

"${SSH_BASE[@]}" "mkdir -p '$PF31_REMOTE_DIR'"

# Capture static environment and inventory.
"${SSH_BASE[@]}" "set -euo pipefail; \
  echo '## uname'; uname -a; \
  echo '## date'; date -u +%Y-%m-%dT%H:%M:%SZ; \
  echo '## lscpu'; lscpu; \
  echo '## mem'; free -h; \
  echo '## nvidia-smi gpu'; nvidia-smi --query-gpu=index,name,memory.total,driver_version,pstate,temperature.gpu,utilization.gpu --format=csv,noheader; \
  echo '## nvidia-smi topo'; nvidia-smi topo -m || true; \
  echo '## nvidia-smi compute apps'; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true; \
  echo '## nvcc'; (command -v nvcc >/dev/null && nvcc --version) || echo 'nvcc: not found'; \
  echo '## rust'; (source /root/.cargo/env >/dev/null 2>&1 || true; rustc --version; cargo --version) 2>/dev/null || true; \
  echo '## nextstat'; '$PF31_BIN' --version" > "$PF31_LOCAL_DIR/remote_env.txt"

gpu_count="$("${SSH_BASE[@]}" "nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '" || echo 0)"
gpu_count="${gpu_count:-0}"
echo "gpu_count=${gpu_count}" >> "$PF31_LOCAL_DIR/run.meta"

if [[ "$gpu_count" -lt "$PF31_MIN_GPUS" ]]; then
  echo "[pf31-preflight] FAIL: detected ${gpu_count} GPUs, required >= ${PF31_MIN_GPUS}" | tee -a "$PF31_LOCAL_DIR/summary.md"
  exit 2
fi

if [[ -z "$PF31_DEVICE_SETS" ]]; then
  if [[ "$gpu_count" -ge 4 ]]; then
    PF31_DEVICE_SETS="0;0,1;0,1,2,3"
  elif [[ "$gpu_count" -ge 2 ]]; then
    PF31_DEVICE_SETS="0;0,1"
  else
    PF31_DEVICE_SETS="0"
  fi
fi
echo "device_sets=${PF31_DEVICE_SETS}" >> "$PF31_LOCAL_DIR/run.meta"
echo "[pf31-preflight] device_sets=${PF31_DEVICE_SETS}"

declare -a CHECK_NAMES=()
declare -a CHECK_RCS=()
declare -a CHECK_ARGS=()

run_check() {
  local name="$1"
  shift
  local args=("$@")
  local cmd=(
    "$PF31_BIN" unbinned-fit-toys
    --config "$PF31_SPEC"
    --n-toys "$PF31_SMOKE_TOYS"
    --seed 42
    --threads 1
    --log-level warn
    --json-metrics "$PF31_REMOTE_DIR/${name}.metrics.json"
  )
  if ((${#args[@]} > 0)); then
    cmd+=("${args[@]}")
  fi

  CHECK_NAMES+=("$name")
  CHECK_ARGS+=("${args[*]-}")

  echo "[pf31-preflight] RUN ${name} ${args[*]-}"
  set +e
  "${SSH_BASE[@]}" bash -s -- "$PF31_REMOTE_DIR" "$name" "$PF31_TIMEOUT" "${cmd[@]}" <<'EOS'
set -euo pipefail
RUN_DIR="$1"
NAME="$2"
TIMEOUT_S="$3"
shift 3
CMD=("$@")
mkdir -p "$RUN_DIR"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
set +e
timeout "$TIMEOUT_S" "${CMD[@]}" >"$RUN_DIR/$NAME.out.json" 2>"$RUN_DIR/$NAME.err"
rc=$?
set -e
echo "$rc" > "$RUN_DIR/$NAME.rc"
exit "$rc"
EOS
  local rc=$?
  set -e
  CHECK_RCS+=("$rc")
  echo "[pf31-preflight] ${name} rc=${rc}"
}

# Baseline single-GPU host path.
run_check "cuda1_host_smoke" --gpu cuda --gpu-devices 0

# Device-resident path checks for configured device sets.
IFS=';' read -r -a DEVICE_SET_LIST <<< "$PF31_DEVICE_SETS"
for dev_set in "${DEVICE_SET_LIST[@]}"; do
  dev_set="$(echo "$dev_set" | xargs)"
  if [[ -z "$dev_set" ]]; then
    continue
  fi
  IFS=',' read -r -a DEVS <<< "$dev_set"
  n_devs="${#DEVS[@]}"
  shards="$n_devs"
  if [[ "$shards" -lt 2 ]]; then
    shards=2
  fi
  run_check "cuda${n_devs}_device_smoke" --gpu cuda --gpu-devices "$dev_set" --gpu-sample-toys --gpu-shards "$shards"
done

# Sync remote preflight artifacts locally immediately.
set +e
"${SSH_BASE[@]}" "tar -C /tmp -cf - '$(basename "$PF31_REMOTE_DIR")' 2>/dev/null" | tar -C "$PF31_LOCAL_DIR" -xf -
sync_rc=$?
set -e
if [[ "$sync_rc" -ne 0 ]]; then
  echo "[pf31-preflight] WARN: sync failed (rc=${sync_rc})"
fi

LOCAL_REMOTE_DIR="$PF31_LOCAL_DIR/$(basename "$PF31_REMOTE_DIR")"
SUMMARY_MD="$PF31_LOCAL_DIR/summary.md"
SUMMARY_JSON="$PF31_LOCAL_DIR/summary.json"

{
  echo "# PF3.1 Remote CUDA Preflight"
  echo
  echo "- Host: \`$PF31_USER@$PF31_HOST:$PF31_PORT\`"
  echo "- Remote bin: \`$PF31_BIN\`"
  echo "- Remote spec: \`$PF31_SPEC\`"
  echo "- Remote artifact dir: \`$PF31_REMOTE_DIR\`"
  echo "- Local artifact dir: \`$PF31_LOCAL_DIR\`"
  echo "- GPU count detected: \`$gpu_count\`"
  echo
  echo "| check | rc | args |"
  echo "|---|---:|---|"
  for i in "${!CHECK_NAMES[@]}"; do
    echo "| ${CHECK_NAMES[$i]} | ${CHECK_RCS[$i]} | \`${CHECK_ARGS[$i]}\` |"
  done
} > "$SUMMARY_MD"

python3 - "$SUMMARY_JSON" "$gpu_count" "$PF31_MIN_GPUS" "$PF31_DEVICE_SETS" "$PF31_REMOTE_DIR" "$LOCAL_REMOTE_DIR" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
gpu_count = int(sys.argv[2])
min_gpus = int(sys.argv[3])
device_sets = sys.argv[4]
remote_dir = sys.argv[5]
local_remote_dir = sys.argv[6]
rows = []
for rc_file in sorted(pathlib.Path(local_remote_dir).glob("*.rc")):
    name = rc_file.stem
    try:
        rc = int(rc_file.read_text(encoding="utf-8").strip())
    except Exception:
        rc = 255
    rows.append({"name": name, "rc": rc})
overall_ok = gpu_count >= min_gpus and all(r["rc"] == 0 for r in rows)
obj = {
    "schema_version": "nextstat.pf31_preflight.v1",
    "gpu_count": gpu_count,
    "min_gpus": min_gpus,
    "device_sets": device_sets,
    "remote_dir": remote_dir,
    "local_remote_dir": local_remote_dir,
    "overall_ok": overall_ok,
    "checks": rows,
}
out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
PY

overall_ok=1
if [[ "$gpu_count" -lt "$PF31_MIN_GPUS" ]]; then
  overall_ok=0
fi
for rc in "${CHECK_RCS[@]}"; do
  if [[ "$rc" -ne 0 ]]; then
    overall_ok=0
  fi
done

if [[ "$overall_ok" -eq 1 ]]; then
  echo "[pf31-preflight] PASS"
  echo "[pf31-preflight] summary: $SUMMARY_JSON"
  exit 0
fi

echo "[pf31-preflight] FAIL"
echo "[pf31-preflight] summary: $SUMMARY_JSON"
exit 3
