#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
render_validation_pack.sh

Build a "validation pack" (Apex2 master report + unified validation_report.json + publishable PDF).

Usage:
  bash validation-pack/render_validation_pack.sh [options]

Options:
  --out-dir DIR            Output directory (default: tmp/validation_pack)
  --workspace PATH         Workspace JSON to fingerprint (default: tests/fixtures/complex_workspace.json)
  --apex2-master PATH      Use an existing Apex2 master JSON instead of running Apex2
  --python PATH            Python interpreter to run Apex2 + PDF renderer (default: .venv/bin/python or python3)
  --nextstat-bin PATH      nextstat CLI binary (default: target/release/nextstat, target/debug/nextstat, or nextstat in PATH)
  --json-only              Generate validation_report.json only (skip PDF rendering and matplotlib requirement)
  --deterministic          Deterministic JSON/PDF output (default)
  --non-deterministic      Allow timestamps/timings in outputs
  --nuts-quality           Also run NUTS quality report (can be slower)
  --root-search-dir PATH   Auto-discover ROOT cases by scanning for combination.xml under PATH
  -h, --help               Show this help
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

out_dir="tmp/validation_pack"
workspace="tests/fixtures/complex_workspace.json"
apex2_master_in=""
py=""
nextstat_bin=""
deterministic=1
render_pdf=1
run_nuts_quality=0
root_search_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --workspace)
      workspace="$2"
      shift 2
      ;;
    --apex2-master)
      apex2_master_in="$2"
      shift 2
      ;;
    --python)
      py="$2"
      shift 2
      ;;
    --nextstat-bin)
      nextstat_bin="$2"
      shift 2
      ;;
    --json-only)
      render_pdf=0
      shift 1
      ;;
    --deterministic)
      deterministic=1
      shift 1
      ;;
    --non-deterministic)
      deterministic=0
      shift 1
      ;;
    --nuts-quality)
      run_nuts_quality=1
      shift 1
      ;;
    --root-search-dir)
      root_search_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$py" ]]; then
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    py="$repo_root/.venv/bin/python"
  else
    py="python3"
  fi
fi

if [[ -z "$nextstat_bin" ]]; then
  if [[ -x "$repo_root/target/release/nextstat" ]]; then
    nextstat_bin="$repo_root/target/release/nextstat"
  elif [[ -x "$repo_root/target/debug/nextstat" ]]; then
    nextstat_bin="$repo_root/target/debug/nextstat"
  else
    nextstat_bin="nextstat"
  fi
fi

# Resolve paths relative to repo root.
if [[ "$out_dir" != /* ]]; then
  out_dir="$repo_root/$out_dir"
fi
if [[ "$workspace" != /* ]]; then
  workspace="$repo_root/$workspace"
fi
if [[ -n "$root_search_dir" && "$root_search_dir" != /* ]]; then
  root_search_dir="$repo_root/$root_search_dir"
fi
if [[ -n "$apex2_master_in" && "$apex2_master_in" != /* ]]; then
  apex2_master_in="$repo_root/$apex2_master_in"
fi

mkdir -p "$out_dir"

if [[ ! -f "$workspace" ]]; then
  echo "Workspace not found: $workspace" >&2
  exit 2
fi

echo "Using python: $py" >&2
echo "Using nextstat: $nextstat_bin" >&2
echo "Output dir: $out_dir" >&2

if [[ "$render_pdf" == "1" ]]; then
  # PDF render requires matplotlib (via nextstat[viz] or a dev env).
  if ! "$py" -c 'import matplotlib' >/dev/null 2>&1; then
    echo "Missing dependency: matplotlib (required to render validation_report.pdf)." >&2
    echo "Install with: pip install 'nextstat[viz]'  (or install matplotlib into your venv)" >&2
    echo "Alternatively, re-run with --json-only to skip PDF rendering." >&2
    exit 2
  fi
fi

apex2_master="$out_dir/apex2_master_report.json"

if [[ -n "$apex2_master_in" ]]; then
  if [[ ! -f "$apex2_master_in" ]]; then
    echo "Apex2 master report not found: $apex2_master_in" >&2
    exit 2
  fi
  if [[ "$(cd "$(dirname "$apex2_master_in")" && pwd)/$(basename "$apex2_master_in")" != "$apex2_master" ]]; then
    cp "$apex2_master_in" "$apex2_master"
  fi
else
  cmd=(
    "$py"
    "$repo_root/tests/apex2_master_report.py"
    --out "$apex2_master"
    --pyhf-out "$out_dir/apex2_pyhf_report.json"
    --nuts-quality-out "$out_dir/apex2_nuts_quality_report.json"
    --root-out "$out_dir/apex2_root_suite_report.json"
    --survival-statsmodels-out "$out_dir/apex2_survival_statsmodels_report.json"
    --p6-glm-bench-out "$out_dir/p6_glm_fit_predict.json"
    --p6-glm-bench-report-out "$out_dir/apex2_p6_glm_bench_report.json"
    --bias-pulls-out "$out_dir/apex2_bias_pulls_report.json"
    --sbc-out "$out_dir/apex2_sbc_report.json"
  )

  if [[ "$deterministic" == "1" ]]; then
    cmd+=(--deterministic)
  fi
  if [[ "$run_nuts_quality" == "1" ]]; then
    cmd+=(--nuts-quality)
  fi
  if [[ -n "$root_search_dir" ]]; then
    cmd+=(--root-search-dir "$root_search_dir" --root-cases-out "$out_dir/apex2_root_cases.json")
  fi

  echo "Running Apex2 master..." >&2
  set +e
  "${cmd[@]}"
  apex2_rc=$?
  set -e
  if [[ "$apex2_rc" != "0" ]]; then
    echo "Apex2 master exited non-zero (rc=$apex2_rc). Continuing to render validation report." >&2
  fi
fi

schema_src="$repo_root/docs/schemas/validation/validation_report_v1.schema.json"
if [[ -f "$schema_src" ]]; then
  cp "$schema_src" "$out_dir/validation_report_v1.schema.json"
fi

validation_json="$out_dir/validation_report.json"
validation_pdf="$out_dir/validation_report.pdf"
manifest_json="$out_dir/validation_pack_manifest.json"

echo "Rendering unified validation report..." >&2
if [[ "$nextstat_bin" == "nextstat" ]]; then
  if command -v nextstat >/dev/null 2>&1; then
    ns_cmd=(nextstat)
  else
    ns_cmd=(cargo run -p ns-cli --quiet --)
  fi
else
  ns_cmd=("$nextstat_bin")
fi

ns_args=(validation-report --apex2 "$apex2_master" --workspace "$workspace" --out "$validation_json")
if [[ "$render_pdf" == "1" ]]; then
  ns_args+=(--pdf "$validation_pdf" --python "$py")
fi
if [[ "$deterministic" == "1" ]]; then
  ns_args+=(--deterministic)
fi

# Avoid matplotlib cache warnings in locked-down environments.
mplconfig="$out_dir/mplconfig"
mkdir -p "$mplconfig"
set +e
MPLCONFIGDIR="$mplconfig" "${ns_cmd[@]}" "${ns_args[@]}"
report_rc=$?
set -e

manifest_files=("apex2_master_report.json" "validation_report.json")
if [[ "$render_pdf" == "1" ]]; then
  manifest_files+=("validation_report.pdf")
fi
if [[ -f "$out_dir/validation_report_v1.schema.json" ]]; then
  manifest_files+=("validation_report_v1.schema.json")
fi

"$py" - "$out_dir" "$deterministic" "${manifest_files[@]}" >"$manifest_json" <<'PY'
import hashlib
import json
import os
import sys
from typing import Any

out_dir = sys.argv[1]
deterministic = sys.argv[2] == "1"
files = sys.argv[3:]

def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

entries: list[dict[str, Any]] = []
for rel in sorted(set(files)):
    path = os.path.join(out_dir, rel)
    st = os.stat(path)
    entries.append({"path": rel, "bytes": st.st_size, "sha256": sha256_file(path)})

doc: dict[str, Any] = {
    "schema_version": "validation_pack_manifest_v1",
    "deterministic": deterministic,
    "files": entries,
}
json.dump(doc, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY

echo "Wrote:" >&2
echo "  $apex2_master" >&2
echo "  $validation_json" >&2
if [[ "$render_pdf" == "1" ]]; then
  echo "  $validation_pdf" >&2
fi
echo "  $manifest_json" >&2
if [[ -f "$out_dir/validation_report_v1.schema.json" ]]; then
  echo "  $out_dir/validation_report_v1.schema.json" >&2
fi

if [[ -n "${apex2_rc:-}" && "$apex2_rc" != "0" ]]; then
  exit "$apex2_rc"
fi
exit "$report_rc"
