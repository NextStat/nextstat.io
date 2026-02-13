#!/usr/bin/env bash
set -euo pipefail

# PF3.1 remote benchmark matrix runner with immediate artifact sync.
# Usage:
#   bash scripts/benchmarks/pf31_remote_matrix.sh
# Optional overrides:
#   PF31_HOST, PF31_PORT, PF31_USER, PF31_KEY
#   PF31_BIN (remote nextstat binary)
#   PF31_SPEC (remote spec path)
#   PF31_TOYS (comma list, default: 10000,50000,100000)
#   PF31_SHARDS (comma list, default: 2,4,8)
#   PF31_GPU_DEVICE_SETS (semicolon list, example: "0;0,1"; auto-detect if unset)
#   PF31_HOST_FIT_MODES (comma list: host,native; default: host,native)
#   PF31_INCLUDE_HOST_SHARDED (1 to include --gpu-shards without --gpu-sample-toys)
#   PF31_LOCAL_DIR (local artifact dir)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PF31_HOST="${PF31_HOST:-88.198.23.172}"
PF31_PORT="${PF31_PORT:-22}"
PF31_USER="${PF31_USER:-root}"
PF31_KEY="${PF31_KEY:-$HOME/.ssh/rundesk_hetzner}"
PF31_BIN="${PF31_BIN:-/root/nextstat.io/target/release/nextstat}"
PF31_SPEC="${PF31_SPEC:-/root/nextstat.io/benchmarks/unbinned/specs/pf31_gauss_exp_10k.json}"
PF31_TOYS="${PF31_TOYS:-10000,50000,100000}"
PF31_SHARDS="${PF31_SHARDS:-2,4,8}"
PF31_GPU_DEVICE_SETS="${PF31_GPU_DEVICE_SETS:-}"
PF31_HOST_FIT_MODES="${PF31_HOST_FIT_MODES:-host,native}"
PF31_INCLUDE_HOST_SHARDED="${PF31_INCLUDE_HOST_SHARDED:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_RUN_DIR="${PF31_REMOTE_RUN_DIR:-/workspace/gpu_test/pf31_matrix_${STAMP}}"
LOCAL_RUN_DIR="${PF31_LOCAL_DIR:-$ROOT_DIR/benchmarks/unbinned/artifacts/2026-02-11/pf31_matrix_${STAMP}}"

SSH_BASE=(ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i "$PF31_KEY" -p "$PF31_PORT" "$PF31_USER@$PF31_HOST")

mkdir -p "$LOCAL_RUN_DIR"

echo "[pf31] host=${PF31_USER}@${PF31_HOST}:${PF31_PORT}"
echo "[pf31] remote_run_dir=${REMOTE_RUN_DIR}"
echo "[pf31] local_run_dir=${LOCAL_RUN_DIR}"
echo "[pf31] bin=${PF31_BIN}"
echo "[pf31] spec=${PF31_SPEC}"
echo "[pf31] toys=${PF31_TOYS}"
echo "[pf31] shards=${PF31_SHARDS}"
echo "[pf31] host_fit_modes=${PF31_HOST_FIT_MODES}"

"${SSH_BASE[@]}" "mkdir -p '${REMOTE_RUN_DIR}'"

if [[ -z "${PF31_GPU_DEVICE_SETS}" ]]; then
  gpu_count="$("${SSH_BASE[@]}" "nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '")"
  if [[ -z "${gpu_count}" || "${gpu_count}" -lt 1 ]]; then
    echo "[pf31] ERROR: could not detect any CUDA GPU on remote host"
    exit 1
  fi
  if [[ "${gpu_count}" -ge 2 ]]; then
    PF31_GPU_DEVICE_SETS="0;0,1"
  else
    PF31_GPU_DEVICE_SETS="0"
  fi
fi
echo "[pf31] gpu_device_sets=${PF31_GPU_DEVICE_SETS}"

run_case() {
  local name="$1"
  local n_toys="$2"
  shift 2
  local args=("$@")
  local all_remote_args=("$REMOTE_RUN_DIR" "$name" "$PF31_SPEC" "$n_toys" "$PF31_BIN")
  if ((${#args[@]} > 0)); then
    all_remote_args+=("${args[@]}")
  fi

  echo "[pf31] RUN ${name} (n_toys=${n_toys}) args='${args[*]-}'"

  set +e
  "${SSH_BASE[@]}" bash -s -- "${all_remote_args[@]}" <<'EOS'
set -euo pipefail
RUN_DIR="$1"
NAME="$2"
SPEC="$3"
N_TOYS="$4"
BIN="$5"
shift 5
ARGS=("$@")

mkdir -p "$RUN_DIR"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

start="$(date +%s)"
set +e
timeout 21600 "$BIN" unbinned-fit-toys \
  --config "$SPEC" \
  --log-level warn \
  --threads 1 \
  --json-metrics "$RUN_DIR/$NAME.metrics.json" \
  --n-toys "$N_TOYS" \
  "${ARGS[@]}" \
  >"$RUN_DIR/$NAME.out.json" \
  2>"$RUN_DIR/$NAME.err"
rc=$?
set -e
end="$(date +%s)"
dur="$((end - start))"

python3 - "$RUN_DIR/$NAME.meta.json" "$NAME" "$SPEC" "$N_TOYS" "$rc" "$dur" "${ARGS[@]}" <<'PY'
import json, sys, time
meta_path, name, spec, n_toys, rc, dur, *args = sys.argv[1:]
obj = {
    "name": name,
    "spec": spec,
    "n_toys": int(n_toys),
    "rc": int(rc),
    "elapsed_s": int(dur),
    "args": args,
    "ts_unix": int(time.time()),
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2)
PY
EOS
  ssh_rc=$?
  set -e

  # Pull artifacts immediately (temporary stand safety)
  set +e
  "${SSH_BASE[@]}" "tar -C '${REMOTE_RUN_DIR}' -cf - '${name}.meta.json' '${name}.metrics.json' '${name}.out.json' '${name}.err' 2>/dev/null" \
    | tar -C "$LOCAL_RUN_DIR" -xf -
  sync_rc=$?
  set -e

  if [[ $sync_rc -ne 0 ]]; then
    echo "[pf31] WARN sync failed for ${name} (sync_rc=${sync_rc})"
  fi
  if [[ $ssh_rc -ne 0 ]]; then
    echo "[pf31] WARN remote wrapper failed for ${name} (ssh_rc=${ssh_rc})"
  fi
  echo "[pf31] DONE ${name}"
}

device_count_from_set() {
  local set="$1"
  IFS=',' read -r -a devs <<< "$set"
  echo "${#devs[@]}"
}

IFS=',' read -r -a TOY_LIST <<< "$PF31_TOYS"
IFS=';' read -r -a DEVICE_SET_LIST <<< "$PF31_GPU_DEVICE_SETS"
IFS=',' read -r -a SHARD_LIST <<< "$PF31_SHARDS"
IFS=',' read -r -a HOST_FIT_MODE_LIST <<< "$PF31_HOST_FIT_MODES"

for toys in "${TOY_LIST[@]}"; do
  toys="$(echo "$toys" | xargs)"
  run_case "cpu_t${toys}" "$toys"
  for dev_set in "${DEVICE_SET_LIST[@]}"; do
    dev_set="$(echo "$dev_set" | xargs)"
    if [[ -z "$dev_set" ]]; then
      continue
    fi
    n_devs="$(device_count_from_set "$dev_set")"
    for fit_mode in "${HOST_FIT_MODE_LIST[@]}"; do
      fit_mode="$(echo "$fit_mode" | xargs)"
      if [[ -z "$fit_mode" ]]; then
        continue
      fi
      case "$fit_mode" in
        host)
          run_case "cuda${n_devs}_host_t${toys}" "$toys" --gpu cuda --gpu-devices "$dev_set"
          ;;
        native)
          run_case "cuda${n_devs}_native_t${toys}" "$toys" --gpu cuda --gpu-devices "$dev_set" --gpu-native
          ;;
        *)
          echo "[pf31] WARN unknown PF31_HOST_FIT_MODES entry '${fit_mode}', skipping"
          ;;
      esac
    done
    for shard in "${SHARD_LIST[@]}"; do
      shard="$(echo "$shard" | xargs)"
      if [[ -z "$shard" ]]; then
        continue
      fi
      if [[ "${PF31_INCLUDE_HOST_SHARDED}" == "1" ]]; then
        run_case "cuda${n_devs}_host_sh${shard}_t${toys}" "$toys" --gpu cuda --gpu-devices "$dev_set" --gpu-shards "$shard"
      fi
      run_case "cuda${n_devs}_device_sh${shard}_t${toys}" "$toys" --gpu cuda --gpu-devices "$dev_set" --gpu-sample-toys --gpu-shards "$shard"
    done
  done
done

python3 - "$LOCAL_RUN_DIR" <<'PY'
import glob, json, os, sys
base = sys.argv[1]
rows = []

def classify_case(name: str) -> str:
    if "_device_sh" in name:
        return "device_sharded"
    if "_host_sh" in name:
        return "host_sharded"
    if "_native_t" in name:
        return "native"
    if "_host_t" in name:
        return "host"
    if name.startswith("cpu_"):
        return "cpu"
    return "unknown"

for meta in sorted(glob.glob(os.path.join(base, "*.meta.json"))):
    d = json.load(open(meta, "r", encoding="utf-8"))
    mpath = meta.replace(".meta.json", ".metrics.json")
    d["fit_mode"] = classify_case(d.get("name", ""))
    if os.path.exists(mpath):
        m = json.load(open(mpath, "r", encoding="utf-8"))
        t = (m.get("timing") or {})
        toys = ((t.get("breakdown") or {}).get("toys") or {})
        d["wall_time_s"] = t.get("wall_time_s")
        d["pipeline"] = toys.get("pipeline")
        d["device_ids"] = toys.get("device_ids")
        d["device_shard_plan"] = toys.get("device_shard_plan")
        d["sample_s"] = toys.get("sample_s")
        d["batch_build_s"] = toys.get("batch_build_s")
        d["batch_fit_s"] = toys.get("batch_fit_s")
        d["poi_sigma_s"] = toys.get("poi_sigma_s")
    rows.append(d)

summary_path = os.path.join(base, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
print(f"[pf31] wrote {summary_path}")
print(json.dumps(rows, indent=2))
PY

echo "[pf31] complete"
