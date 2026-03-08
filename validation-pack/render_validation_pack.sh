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
  --sign-gpg               Produce a detached ASCII-armored signature for validation_pack_manifest.json (requires gpg)
  --gpg-key KEYID          Optional: key id/email/fingerprint to use with --sign-gpg (default: gpg default)
  --gpg-home DIR           Optional: GNUPGHOME to use with --sign-gpg (default: --out-dir/.gnupg)
  --sign-openssl-key PATH  Produce a detached binary signature for validation_pack_manifest.json (requires openssl)
  --sign-openssl-pub PATH  Optional: copy the corresponding public key to --out-dir (as validation_pack_manifest.pub.pem)
  --deterministic          Deterministic JSON/PDF output (default)
  --non-deterministic      Allow timestamps/timings in outputs
  --nuts-quality           Also run NUTS quality report (can be slower)
  --root-search-dir PATH   Auto-discover ROOT cases by scanning for combination.xml under PATH
  --skip-pharma-validation Skip pharma IQ/OQ/PQ validation runner
  --m15-config PATH        Also generate M15 assessment-table / MAP / MAR / profile diff / bundle artifacts from this config
  --bayesian-design-report PATH
                           Also package a frozen Bayesian design report plus regulatory appendix artifacts
  -h, --help               Show this help

Examples:
  # Full pack (JSON + PDF):
  bash validation-pack/render_validation_pack.sh --out-dir tmp/validation_pack --deterministic

  # JSON only (no PDF / no matplotlib):
  bash validation-pack/render_validation_pack.sh --out-dir tmp/validation_pack --deterministic --json-only

  # Fast, fixture-driven pack (use existing Apex2 master input):
  bash validation-pack/render_validation_pack.sh \
    --out-dir tmp/validation_pack_fixture \
    --workspace tests/fixtures/simple_workspace.json \
    --apex2-master tests/fixtures/apex2_master_min_plus.json \
    --deterministic

  # JSON-only validation pack + M15 artifacts:
  bash validation-pack/render_validation_pack.sh \
    --out-dir tmp/validation_pack_m15 \
    --workspace tests/fixtures/simple_workspace.json \
    --apex2-master tests/fixtures/apex2_master_min_plus.json \
    --m15-config docs/specs/m15_config_v1.example.json \
    --deterministic --json-only

  # Full validation pack + publishable M15 report (Markdown/PDF/DOCX):
  bash validation-pack/render_validation_pack.sh \
    --out-dir tmp/validation_pack_m15_publishable \
    --workspace tests/fixtures/simple_workspace.json \
    --apex2-master tests/fixtures/apex2_master_min_plus.json \
    --m15-config docs/specs/m15_config_v1.example.json \
    --deterministic

  # JSON-only validation pack + frozen Bayesian design appendix artifacts:
  bash validation-pack/render_validation_pack.sh \
    --out-dir tmp/validation_pack_bayes \
    --workspace tests/fixtures/simple_workspace.json \
    --apex2-master tests/fixtures/apex2_master_min_plus.json \
    --bayesian-design-report docs/specs/pharma/beta_binomial_design_report_v0.example.json \
    --deterministic --json-only
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
sign_gpg=0
gpg_key=""
gpg_home=""
openssl_key=""
openssl_pub=""
run_nuts_quality=0
root_search_dir=""
skip_pharma_validation=0
m15_config=""
bayesian_design_report=""

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
    --sign-gpg)
      sign_gpg=1
      shift 1
      ;;
    --gpg-key)
      gpg_key="$2"
      shift 2
      ;;
    --gpg-home)
      gpg_home="$2"
      shift 2
      ;;
    --sign-openssl-key)
      openssl_key="$2"
      shift 2
      ;;
    --sign-openssl-pub)
      openssl_pub="$2"
      shift 2
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
    --skip-pharma-validation)
      skip_pharma_validation=1
      shift 1
      ;;
    --m15-config)
      m15_config="$2"
      shift 2
      ;;
    --bayesian-design-report)
      bayesian_design_report="$2"
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
if [[ -n "$m15_config" && "$m15_config" != /* ]]; then
  m15_config="$repo_root/$m15_config"
fi
if [[ -n "$bayesian_design_report" && "$bayesian_design_report" != /* ]]; then
  bayesian_design_report="$repo_root/$bayesian_design_report"
fi

mkdir -p "$out_dir"

if [[ ! -f "$workspace" ]]; then
  echo "Workspace not found: $workspace" >&2
  exit 2
fi
if [[ -n "$m15_config" && ! -f "$m15_config" ]]; then
  echo "M15 config not found: $m15_config" >&2
  exit 2
fi
if [[ -n "$bayesian_design_report" && ! -f "$bayesian_design_report" ]]; then
  echo "Bayesian design report not found: $bayesian_design_report" >&2
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

pharma_validation_json="$out_dir/pharma_validation.json"
if [[ "$skip_pharma_validation" == "0" ]]; then
  pharma_runner="$repo_root/tests/pharma_validation/runner.py"
  if [[ -f "$pharma_runner" ]]; then
    echo "Running pharma IQ/OQ/PQ validation..." >&2
    pharma_cmd=("$py" "$pharma_runner" --out "$pharma_validation_json")
    if [[ "$deterministic" == "1" ]]; then
      pharma_cmd+=(--deterministic)
    fi
    set +e
    # IMPORTANT: run against the installed `nextstat` wheel, not the repo's
    # `bindings/ns-py/python` sources. This avoids accidentally importing a stale
    # in-tree `nextstat/_core*.so` and makes CI reflect the published artifact.
    PYTHONPATH="$repo_root/tests${PYTHONPATH:+:$PYTHONPATH}" "${pharma_cmd[@]}"
    pharma_rc=$?
    set -e
    if [[ "$pharma_rc" != "0" ]]; then
      echo "Pharma validation runner exited non-zero (rc=$pharma_rc). Continuing." >&2
    fi
  else
    echo "Pharma validation runner not found: $pharma_runner (skipping)" >&2
  fi
fi

schema_src="$repo_root/docs/schemas/validation/validation_report_v1.schema.json"
if [[ -f "$schema_src" ]]; then
  cp "$schema_src" "$out_dir/validation_report_v1.schema.json"
fi

validation_json="$out_dir/validation_report.json"
validation_pdf="$out_dir/validation_report.pdf"
manifest_json="$out_dir/validation_pack_manifest.json"
m15_config_out="$out_dir/m15_config.json"
m15_profile_diff_json="$out_dir/m15_profile_diff_report.json"
m15_assessment_json="$out_dir/m15_assessment_table.json"
m15_map_json="$out_dir/m15_map.json"
m15_mar_json="$out_dir/m15_mar.json"
m15_bundle_json="$out_dir/m15_bundle_manifest.json"
m15_report_markdown="$out_dir/m15_report.md"
m15_report_pdf="$out_dir/m15_report.pdf"
m15_report_docx="$out_dir/m15_report.docx"
m15_profile_diff_schema_out="$out_dir/m15_profile_diff_report_v1.schema.json"
m15_bundle_schema_out="$out_dir/m15_bundle_manifest_v1.schema.json"
bayesian_design_report_out="$out_dir/bayesian_design_report.json"
bayesian_design_report_schema_out=""
bayesian_design_appendix_json="$out_dir/bayesian_design_regulatory_appendix.json"
bayesian_design_appendix_markdown="$out_dir/bayesian_design_regulatory_appendix.md"
bayesian_design_appendix_pdf="$out_dir/bayesian_design_regulatory_appendix.pdf"
bayesian_design_appendix_schema_out="$out_dir/bayesian_design_regulatory_appendix_v0.schema.json"

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

if [[ -n "$bayesian_design_report" ]]; then
  bayesian_design_schema_filename="$("$py" - "$bayesian_design_report" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
schema_version = str(report.get("schema_version"))
design_family = str(report.get("design_family"))

if schema_version == "nextstat_beta_binomial_design_report_v0" and design_family == "beta_binomial":
    print("beta_binomial_design_report_v0.schema.json")
elif schema_version == "nextstat_normal_normal_design_report_v0" and design_family == "normal_normal":
    print("normal_normal_design_report_v0.schema.json")
else:
    raise SystemExit(
        "Unsupported Bayesian design report: expected frozen beta-binomial or normal-normal "
        "*_design_report_v0 artifact"
    )
PY
)"

  if [[ "$(cd "$(dirname "$bayesian_design_report")" && pwd)/$(basename "$bayesian_design_report")" != "$bayesian_design_report_out" ]]; then
    cp "$bayesian_design_report" "$bayesian_design_report_out"
  fi

  bayesian_design_report_schema_src="$repo_root/docs/schemas/pharma/$bayesian_design_schema_filename"
  if [[ ! -f "$bayesian_design_report_schema_src" ]]; then
    echo "Bayesian design report schema not found: $bayesian_design_report_schema_src" >&2
    exit 2
  fi
  bayesian_design_report_schema_out="$out_dir/$bayesian_design_schema_filename"
  cp "$bayesian_design_report_schema_src" "$bayesian_design_report_schema_out"

  bayesian_design_appendix_schema_src="$repo_root/docs/schemas/pharma/bayesian_design_regulatory_appendix_v0.schema.json"
  if [[ ! -f "$bayesian_design_appendix_schema_src" ]]; then
    echo "Bayesian design appendix schema not found: $bayesian_design_appendix_schema_src" >&2
    exit 2
  fi
  cp "$bayesian_design_appendix_schema_src" "$bayesian_design_appendix_schema_out"

  echo "Rendering Bayesian regulatory appendix..." >&2
  # IMPORTANT: use the installed package / caller-provided import environment.
  # Do not prepend `bindings/ns-py/python` here; that source-tree wrapper can
  # shadow the compiled wheel and break release-grade validation semantics.
  "$py" - "$bayesian_design_report_out" "$bayesian_design_appendix_json" <<'PY'
import json
import sys
from pathlib import Path

from nextstat import bayes_design

report_path = Path(sys.argv[1])
appendix_path = Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))
schema_version = str(report.get("schema_version"))

if schema_version == "nextstat_beta_binomial_design_report_v0":
    appendix = bayes_design.build_beta_binomial_regulatory_appendix(report)
elif schema_version == "nextstat_normal_normal_design_report_v0":
    appendix = bayes_design.build_normal_normal_regulatory_appendix(report)
else:
    raise SystemExit(
        "Unsupported Bayesian design report: expected frozen beta-binomial or normal-normal "
        "*_design_report_v0 artifact"
    )

appendix_path.write_text(
    json.dumps(appendix, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

  echo "Rendering Bayesian regulatory appendix Markdown/PDF..." >&2
  "$py" - "$bayesian_design_appendix_json" "$bayesian_design_appendix_markdown" "$bayesian_design_appendix_pdf" "$render_pdf" <<'PY'
import json
import sys
from pathlib import Path

from nextstat import bayes_design

appendix_path = Path(sys.argv[1])
markdown_path = Path(sys.argv[2])
pdf_path = Path(sys.argv[3])
render_pdf = sys.argv[4] == "1"
appendix = json.loads(appendix_path.read_text(encoding="utf-8"))

markdown_path.write_text(
    bayes_design.render_bayesian_regulatory_appendix_markdown(appendix),
    encoding="utf-8",
)
if render_pdf:
    bayes_design.write_bayesian_regulatory_appendix_pdf(pdf_path, appendix)
PY
fi

if [[ -n "$m15_config" ]]; then
  if [[ "$render_pdf" == "1" ]]; then
    if ! "$py" -c 'import docx' >/dev/null 2>&1; then
      echo "Missing dependency: python-docx (required to render m15_report.docx)." >&2
      echo "Install with: pip install python-docx" >&2
      echo "Alternatively, re-run with --json-only to skip M15 publishable PDF/DOCX rendering." >&2
      exit 2
    fi
  fi
  if [[ ! -f "$pharma_validation_json" ]]; then
    echo "M15 artifact generation requires $pharma_validation_json." >&2
    echo "Either omit --skip-pharma-validation or pre-seed that file before re-rendering." >&2
    exit 2
  fi
  if [[ "$report_rc" != "0" ]]; then
    echo "validation-report exited non-zero (rc=$report_rc); refusing to build M15 artifacts." >&2
    exit "$report_rc"
  fi

  if [[ "$(cd "$(dirname "$m15_config")" && pwd)/$(basename "$m15_config")" != "$m15_config_out" ]]; then
    cp "$m15_config" "$m15_config_out"
  fi

  m15_profile_diff_schema_src="$repo_root/docs/schemas/validation/m15_profile_diff_report_v1.schema.json"
  if [[ -f "$m15_profile_diff_schema_src" ]]; then
    cp "$m15_profile_diff_schema_src" "$m15_profile_diff_schema_out"
  fi
  m15_schema_src="$repo_root/docs/schemas/validation/m15_bundle_manifest_v1.schema.json"
  if [[ -f "$m15_schema_src" ]]; then
    cp "$m15_schema_src" "$m15_bundle_schema_out"
  fi

  echo "Rendering M15 profile diff report..." >&2
  m15_profile_diff_cmd=(
    "${ns_cmd[@]}"
    m15 profile-diff
    --config "$m15_config_out"
    --output "$m15_profile_diff_json"
  )
  if [[ "$deterministic" == "1" ]]; then
    m15_profile_diff_cmd+=(--deterministic)
  fi
  "${m15_profile_diff_cmd[@]}"

  echo "Rendering M15 assessment-table..." >&2
  m15_assessment_cmd=(
    "${ns_cmd[@]}"
    m15 assessment-table
    --config "$m15_config_out"
    --validation-report "$validation_json"
    --pharma-validation "$pharma_validation_json"
    --output "$m15_assessment_json"
  )
  if [[ "$deterministic" == "1" ]]; then
    m15_assessment_cmd+=(--deterministic)
  fi
  "${m15_assessment_cmd[@]}"

  echo "Rendering M15 MAP..." >&2
  m15_map_cmd=(
    "${ns_cmd[@]}"
    m15 map
    --config "$m15_config_out"
    --assessment-table "$m15_assessment_json"
    --output "$m15_map_json"
  )
  if [[ "$deterministic" == "1" ]]; then
    m15_map_cmd+=(--deterministic)
  fi
  "${m15_map_cmd[@]}"

  echo "Rendering M15 MAR..." >&2
  m15_mar_cmd=(
    "${ns_cmd[@]}"
    m15 mar
    --map "$m15_map_json"
    --assessment-table "$m15_assessment_json"
    --validation-report "$validation_json"
    --pharma-validation "$pharma_validation_json"
    --output "$m15_mar_json"
  )
  if [[ "$deterministic" == "1" ]]; then
    m15_mar_cmd+=(--deterministic)
  fi
  "${m15_mar_cmd[@]}"

  echo "Rendering M15 bundle manifest..." >&2
  m15_bundle_cmd=(
    "${ns_cmd[@]}"
    m15 bundle
    --config "$m15_config_out"
    --assessment-table "$m15_assessment_json"
    --map "$m15_map_json"
    --mar "$m15_mar_json"
    --validation-report "$validation_json"
    --pharma-validation "$pharma_validation_json"
    --output "$m15_bundle_json"
  )
  if [[ "$deterministic" == "1" ]]; then
    m15_bundle_cmd+=(--deterministic)
  fi
  "${m15_bundle_cmd[@]}"

  if [[ "$render_pdf" == "1" ]]; then
    echo "Rendering publishable M15 report..." >&2
    "$py" -m nextstat.m15_report render \
      --assessment-table "$m15_assessment_json" \
      --map "$m15_map_json" \
      --mar "$m15_mar_json" \
      --profile-diff "$m15_profile_diff_json" \
      --bundle "$m15_bundle_json" \
      --markdown "$m15_report_markdown" \
      --pdf "$m15_report_pdf" \
      --docx "$m15_report_docx"
  fi
fi

manifest_files=("apex2_master_report.json" "validation_report.json")
if [[ "$render_pdf" == "1" ]]; then
  manifest_files+=("validation_report.pdf")
fi
if [[ -f "$out_dir/validation_report_v1.schema.json" ]]; then
  manifest_files+=("validation_report_v1.schema.json")
fi
if [[ -f "$pharma_validation_json" ]]; then
  manifest_files+=("pharma_validation.json")
fi
if [[ -f "$bayesian_design_report_out" ]]; then
  manifest_files+=("bayesian_design_report.json")
fi
if [[ -n "$bayesian_design_report_schema_out" && -f "$bayesian_design_report_schema_out" ]]; then
  manifest_files+=("$(basename "$bayesian_design_report_schema_out")")
fi
if [[ -f "$bayesian_design_appendix_json" ]]; then
  manifest_files+=("bayesian_design_regulatory_appendix.json")
fi
if [[ -f "$bayesian_design_appendix_markdown" ]]; then
  manifest_files+=("bayesian_design_regulatory_appendix.md")
fi
if [[ -f "$bayesian_design_appendix_pdf" ]]; then
  manifest_files+=("bayesian_design_regulatory_appendix.pdf")
fi
if [[ -f "$bayesian_design_appendix_schema_out" ]]; then
  manifest_files+=("bayesian_design_regulatory_appendix_v0.schema.json")
fi
if [[ -f "$m15_config_out" ]]; then
  manifest_files+=("m15_config.json")
fi
if [[ -f "$m15_profile_diff_json" ]]; then
  manifest_files+=("m15_profile_diff_report.json")
fi
if [[ -f "$m15_assessment_json" ]]; then
  manifest_files+=("m15_assessment_table.json")
fi
if [[ -f "$m15_map_json" ]]; then
  manifest_files+=("m15_map.json")
fi
if [[ -f "$m15_mar_json" ]]; then
  manifest_files+=("m15_mar.json")
fi
if [[ -f "$m15_bundle_json" ]]; then
  manifest_files+=("m15_bundle_manifest.json")
fi
if [[ -f "$m15_report_markdown" ]]; then
  manifest_files+=("m15_report.md")
fi
if [[ -f "$m15_report_pdf" ]]; then
  manifest_files+=("m15_report.pdf")
fi
if [[ -f "$m15_report_docx" ]]; then
  manifest_files+=("m15_report.docx")
fi
if [[ -f "$m15_profile_diff_schema_out" ]]; then
  manifest_files+=("m15_profile_diff_report_v1.schema.json")
fi
if [[ -f "$m15_bundle_schema_out" ]]; then
  manifest_files+=("m15_bundle_manifest_v1.schema.json")
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

manifest_sig="$manifest_json.asc"
if [[ "$sign_gpg" == "1" ]]; then
  if ! command -v gpg >/dev/null 2>&1; then
    echo "Missing dependency: gpg (required for --sign-gpg)." >&2
    exit 2
  fi
  if [[ -z "$gpg_home" ]]; then
    gpg_home="$out_dir/.gnupg"
  fi
  mkdir -p "$gpg_home"
  chmod 700 "$gpg_home" || true
  gpg_args=(--batch --yes --pinentry-mode loopback --armor --detach-sign --output "$manifest_sig")
  if [[ -n "$gpg_key" ]]; then
    gpg_args+=(--local-user "$gpg_key")
  fi
  GNUPGHOME="$gpg_home" gpg "${gpg_args[@]}" "$manifest_json"
fi

manifest_sig_openssl="$manifest_json.sig"
manifest_sha256_hex="$out_dir/validation_pack_manifest.sha256"
manifest_sha256_bin="$out_dir/validation_pack_manifest.sha256.bin"
manifest_pub_openssl="$out_dir/validation_pack_manifest.pub.pem"
if [[ -n "$openssl_key" ]]; then
  if [[ ! -f "$openssl_key" ]]; then
    echo "OpenSSL key not found: $openssl_key" >&2
    exit 2
  fi
  if ! command -v openssl >/dev/null 2>&1; then
    echo "Missing dependency: openssl (required for --sign-openssl-key)." >&2
    exit 2
  fi
  # Sign SHA-256 digest bytes (raw) to support both RSA/ECDSA and Ed25519 keys.
  openssl dgst -sha256 -hex "$manifest_json" | awk '{print $2}' >"$manifest_sha256_hex"
  openssl dgst -sha256 -binary "$manifest_json" >"$manifest_sha256_bin"
  openssl pkeyutl -sign -inkey "$openssl_key" -rawin -in "$manifest_sha256_bin" -out "$manifest_sig_openssl"
  if [[ -n "$openssl_pub" ]]; then
    if [[ ! -f "$openssl_pub" ]]; then
      echo "OpenSSL public key not found: $openssl_pub" >&2
      exit 2
    fi
    cp "$openssl_pub" "$manifest_pub_openssl"
  fi
fi

echo "Wrote:" >&2
echo "  $apex2_master" >&2
echo "  $validation_json" >&2
if [[ "$render_pdf" == "1" ]]; then
  echo "  $validation_pdf" >&2
fi
if [[ -f "$bayesian_design_report_out" ]]; then
  echo "  $bayesian_design_report_out" >&2
fi
if [[ -f "$bayesian_design_appendix_json" ]]; then
  echo "  $bayesian_design_appendix_json" >&2
fi
if [[ -f "$bayesian_design_appendix_markdown" ]]; then
  echo "  $bayesian_design_appendix_markdown" >&2
fi
if [[ -f "$bayesian_design_appendix_pdf" ]]; then
  echo "  $bayesian_design_appendix_pdf" >&2
fi
echo "  $manifest_json" >&2
if [[ "$sign_gpg" == "1" ]]; then
  echo "  $manifest_sig" >&2
fi
if [[ -n "$openssl_key" ]]; then
  echo "  $manifest_sig_openssl" >&2
  echo "  $manifest_sha256_hex" >&2
  if [[ -n "$openssl_pub" ]]; then
    echo "  $manifest_pub_openssl" >&2
  fi
fi
if [[ -f "$out_dir/validation_report_v1.schema.json" ]]; then
  echo "  $out_dir/validation_report_v1.schema.json" >&2
fi
if [[ -f "$pharma_validation_json" ]]; then
  echo "  $pharma_validation_json" >&2
fi
if [[ -f "$m15_config_out" ]]; then
  echo "  $m15_config_out" >&2
fi
if [[ -f "$m15_profile_diff_json" ]]; then
  echo "  $m15_profile_diff_json" >&2
fi
if [[ -f "$m15_assessment_json" ]]; then
  echo "  $m15_assessment_json" >&2
fi
if [[ -f "$m15_map_json" ]]; then
  echo "  $m15_map_json" >&2
fi
if [[ -f "$m15_mar_json" ]]; then
  echo "  $m15_mar_json" >&2
fi
if [[ -f "$m15_bundle_json" ]]; then
  echo "  $m15_bundle_json" >&2
fi
if [[ -f "$m15_report_markdown" ]]; then
  echo "  $m15_report_markdown" >&2
fi
if [[ -f "$m15_report_pdf" ]]; then
  echo "  $m15_report_pdf" >&2
fi
if [[ -f "$m15_report_docx" ]]; then
  echo "  $m15_report_docx" >&2
fi
if [[ -f "$m15_profile_diff_schema_out" ]]; then
  echo "  $m15_profile_diff_schema_out" >&2
fi
if [[ -f "$m15_bundle_schema_out" ]]; then
  echo "  $m15_bundle_schema_out" >&2
fi

if [[ -n "${apex2_rc:-}" && "$apex2_rc" != "0" ]]; then
  exit "$apex2_rc"
fi
exit "$report_rc"
