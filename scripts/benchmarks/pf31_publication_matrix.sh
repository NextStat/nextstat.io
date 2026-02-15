#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/scripts/benchmarks"

MATRIX_JSON="${PF31_PUBLICATION_MATRIX:-$ROOT_DIR/benchmarks/unbinned/matrices/pf31_publication_v1.json}"
PF31_HOST="${PF31_HOST:-88.198.23.172}"
PF31_PORT="${PF31_PORT:-22}"
PF31_USER="${PF31_USER:-root}"
PF31_KEY="${PF31_KEY:-$HOME/.ssh/rundesk_hetzner}"
PF31_REMOTE_REPO="${PF31_REMOTE_REPO:-/root/nextstat.io}"
PF31_BIN="${PF31_BIN:-$PF31_REMOTE_REPO/target/release/nextstat}"
PF31_GPU_DEVICE_SETS="${PF31_GPU_DEVICE_SETS:-}"
PF31_THREADS="${PF31_THREADS:-1}"
PF31_DRY_RUN="${PF31_DRY_RUN:-1}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DATE_TAG="$(date -u +%Y-%m-%d)"
RUN_ID="${PF31_RUN_ID:-pf31_publication_${STAMP}}"
REMOTE_ROOT="${PF31_REMOTE_ROOT:-/workspace/gpu_test/${RUN_ID}}"
LOCAL_ROOT="${PF31_LOCAL_ROOT:-$ROOT_DIR/benchmarks/unbinned/artifacts/${DATE_TAG}/${RUN_ID}}"

mkdir -p "$LOCAL_ROOT"

if [[ ! -f "$MATRIX_JSON" ]]; then
  echo "[pf31-pub] ERROR: matrix file not found: $MATRIX_JSON" >&2
  exit 1
fi

cp "$MATRIX_JSON" "$LOCAL_ROOT/matrix.json"

{
  echo "run_id=$RUN_ID"
  echo "utc_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "matrix_json=$MATRIX_JSON"
  echo "host=$PF31_USER@$PF31_HOST:$PF31_PORT"
  echo "remote_repo=$PF31_REMOTE_REPO"
  echo "remote_bin=$PF31_BIN"
  echo "remote_root=$REMOTE_ROOT"
  echo "local_root=$LOCAL_ROOT"
  echo "dry_run=$PF31_DRY_RUN"
  echo "gpu_device_sets=${PF31_GPU_DEVICE_SETS:-auto}"
  echo "threads=$PF31_THREADS"
  echo "local_git_head=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$LOCAL_ROOT/run.meta"

git -C "$ROOT_DIR" status --short > "$LOCAL_ROOT/local_git_status.txt" || true

# Emit an explicit execution plan up front.
python3 - "$MATRIX_JSON" > "$LOCAL_ROOT/planned_cases.tsv" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
for c in m.get("cases", []):
    print("\t".join([
        c["id"],
        c["spec_rel"],
        c["toys"],
        c["shards"],
        c.get("host_fit_modes", "host,native"),
        str(int(c.get("include_host_sharded", 0))),
    ]))
PY

echo "[pf31-pub] run_id=${RUN_ID}"
echo "[pf31-pub] local_root=${LOCAL_ROOT}"
echo "[pf31-pub] remote_root=${REMOTE_ROOT}"
echo "[pf31-pub] dry_run=${PF31_DRY_RUN}"

if [[ "$PF31_DRY_RUN" == "1" ]]; then
  echo "[pf31-pub] DRY RUN: command expansion only (no SSH, no benchmark run)."
  while IFS=$'\t' read -r case_id spec_rel toys shards host_modes include_host_sharded; do
    spec_remote="$PF31_REMOTE_REPO/$spec_rel"
    echo "[pf31-pub] case=${case_id}"
    echo "  PF31_SPEC='${spec_remote}' PF31_TOYS='${toys}' PF31_SHARDS='${shards}' PF31_HOST_FIT_MODES='${host_modes}' PF31_INCLUDE_HOST_SHARDED='${include_host_sharded}' PF31_THREADS='${PF31_THREADS}'"
    echo "  PF31_LOCAL_DIR='${LOCAL_ROOT}/${case_id}' PF31_REMOTE_RUN_DIR='${REMOTE_ROOT}/${case_id}' bash scripts/benchmarks/pf31_remote_matrix.sh"
  done < "$LOCAL_ROOT/planned_cases.tsv"
  echo "[pf31-pub] dry-run complete"
  exit 0
fi

SSH_BASE=(ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i "$PF31_KEY" -p "$PF31_PORT" "$PF31_USER@$PF31_HOST")

# Capture remote environment manifest once per publication run.
"${SSH_BASE[@]}" "set -euo pipefail; mkdir -p '$REMOTE_ROOT'; \
  echo '## uname'; uname -a; \
  echo '## lscpu'; lscpu; \
  echo '## mem'; free -h; \
  echo '## nvidia-smi'; nvidia-smi --query-gpu=index,name,memory.total,driver_version,utilization.gpu --format=csv,noheader; \
  echo '## nvcc'; (command -v nvcc >/dev/null && nvcc --version) || echo 'nvcc: not found'; \
  echo '## rust'; (source /root/.cargo/env >/dev/null 2>&1 || true; rustc --version; cargo --version) 2>/dev/null || true; \
  echo '## nextstat version'; '$PF31_BIN' --version" > "$LOCAL_ROOT/remote_env.txt"

while IFS=$'\t' read -r case_id spec_rel toys shards host_modes include_host_sharded; do
  spec_remote="$PF31_REMOTE_REPO/$spec_rel"
  case_local="$LOCAL_ROOT/$case_id"
  case_remote="$REMOTE_ROOT/$case_id"

  mkdir -p "$case_local"

  echo "[pf31-pub] running case=${case_id} spec=${spec_remote} toys=${toys} shards=${shards}"
  # NOTE: pf31_remote_matrix.sh uses ssh; ssh can consume stdin and break this while-read loop.
  # Force a clean stdin so we don't stop after the first case.
  env \
    PF31_HOST="$PF31_HOST" \
    PF31_PORT="$PF31_PORT" \
    PF31_USER="$PF31_USER" \
    PF31_KEY="$PF31_KEY" \
    PF31_BIN="$PF31_BIN" \
    PF31_SPEC="$spec_remote" \
    PF31_TOYS="$toys" \
    PF31_SHARDS="$shards" \
    PF31_GPU_DEVICE_SETS="$PF31_GPU_DEVICE_SETS" \
    PF31_HOST_FIT_MODES="$host_modes" \
    PF31_INCLUDE_HOST_SHARDED="$include_host_sharded" \
    PF31_THREADS="$PF31_THREADS" \
    PF31_LOCAL_DIR="$case_local" \
    PF31_REMOTE_RUN_DIR="$case_remote" \
    bash "$SCRIPT_DIR/pf31_remote_matrix.sh" </dev/null
done < "$LOCAL_ROOT/planned_cases.tsv"

python3 "$SCRIPT_DIR/pf31_publication_report.py" \
  --run-root "$LOCAL_ROOT" \
  --matrix "$LOCAL_ROOT/matrix.json" \
  --out-json "$LOCAL_ROOT/publication_summary.json" \
  --out-md "$LOCAL_ROOT/publication_summary.md"

python3 "$SCRIPT_DIR/write_snapshot_index.py" \
  --suite "pf31-unbinned-publication" \
  --snapshot-id "$RUN_ID" \
  --artifacts-dir "$LOCAL_ROOT" \
  --out "$LOCAL_ROOT/snapshot_index.json"

echo "[pf31-pub] complete"
echo "[pf31-pub] bundle: $LOCAL_ROOT"
