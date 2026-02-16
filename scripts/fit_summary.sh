#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/fit_summary.sh --input <fit.json> [--out-dir <dir>] [--prefix <name>] [--top <N>]

Description:
  Generates compact artifacts from a NextStat fit JSON:
    - <prefix>.summary.json
    - <prefix>.params.tsv
    - <prefix>.top_unc.tsv
    - <prefix>.top_pull.tsv

Options:
  --input <path>   Path to fit JSON (required)
  --out-dir <dir>  Output directory (default: input file directory)
  --prefix <name>  Output file prefix (default: input basename without extension)
  --top <N>        Number of rows for top tables (default: 30)
  -h, --help       Show this help
EOF
}

INPUT=""
OUT_DIR=""
PREFIX=""
TOP_N=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --prefix)
      PREFIX="${2:-}"
      shift 2
      ;;
    --top)
      TOP_N="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${INPUT}" ]]; then
  echo "ERROR: --input is required." >&2
  usage
  exit 2
fi

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: input file not found: ${INPUT}" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required." >&2
  exit 2
fi

if ! [[ "${TOP_N}" =~ ^[0-9]+$ ]] || [[ "${TOP_N}" -le 0 ]]; then
  echo "ERROR: --top must be a positive integer, got: ${TOP_N}" >&2
  exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="$(cd "$(dirname "${INPUT}")" && pwd)"
fi

if [[ -z "${PREFIX}" ]]; then
  base="$(basename "${INPUT}")"
  PREFIX="${base%.*}"
fi

mkdir -p "${OUT_DIR}"

if ! jq -e '
  (.parameter_names|type=="array") and
  (.bestfit|type=="array") and
  (.uncertainties|type=="array") and
  ((.parameter_names|length)==(.bestfit|length)) and
  ((.parameter_names|length)==(.uncertainties|length))
' "${INPUT}" >/dev/null; then
  echo "ERROR: fit JSON is missing expected arrays or they have different lengths." >&2
  exit 2
fi

SUMMARY_JSON="${OUT_DIR}/${PREFIX}.summary.json"
PARAMS_TSV="${OUT_DIR}/${PREFIX}.params.tsv"
TOP_UNC_TSV="${OUT_DIR}/${PREFIX}.top_unc.tsv"
TOP_PULL_TSV="${OUT_DIR}/${PREFIX}.top_pull.tsv"

jq '
{
  converged,
  n_iter,
  nll,
  twice_nll,
  edm,
  final_grad_norm,
  n_active_bounds,
  termination_reason,
  poi_index,
  poi: (if (.poi_index|type=="number") then .parameter_names[.poi_index] else null end),
  poi_hat: (if (.poi_index|type=="number") then .bestfit[.poi_index] else null end),
  poi_unc: (if (.poi_index|type=="number") then .uncertainties[.poi_index] else null end),
  n_params: (.parameter_names|length),
  fit_regions,
  validation_regions
}
' "${INPUT}" > "${SUMMARY_JSON}"

{
  printf 'name\tvalue\tuncertainty\n'
  jq -r '
    [.parameter_names, .bestfit, .uncertainties]
    | transpose[]
    | "\(.[0])\t\(.[1])\t\(.[2])"
  ' "${INPUT}"
} > "${PARAMS_TSV}"

{
  printf 'name\tvalue\tuncertainty\n'
  jq -r --argjson top_n "${TOP_N}" '
    [.parameter_names, .bestfit, .uncertainties]
    | transpose
    | map({name: .[0], value: .[1], unc: .[2]})
    | sort_by(.unc) | reverse | .[:$top_n]
    | .[]
    | "\(.name)\t\(.value)\t\(.unc)"
  ' "${INPUT}"
} > "${TOP_UNC_TSV}"

{
  printf 'name\tvalue\tuncertainty\tpull\n'
  jq -r --argjson top_n "${TOP_N}" '
    [.parameter_names, .bestfit, .uncertainties]
    | transpose
    | map({
        name: .[0],
        value: .[1],
        unc: .[2],
        pull: (
          if .[2] <= 0 then null
          elif (.[0] | startswith("alpha_")) then (.[1] / .[2])
          elif (.[0] | startswith("gamma_")) then ((.[1] - 1.0) / .[2])
          else null
          end
        )
      })
    | map(select(.pull != null))
    | sort_by(.pull | abs) | reverse | .[:$top_n]
    | .[]
    | "\(.name)\t\(.value)\t\(.unc)\t\(.pull)"
  ' "${INPUT}"
} > "${TOP_PULL_TSV}"

echo "Wrote:"
echo "  ${SUMMARY_JSON}"
echo "  ${PARAMS_TSV}"
echo "  ${TOP_UNC_TSV}"
echo "  ${TOP_PULL_TSV}"
