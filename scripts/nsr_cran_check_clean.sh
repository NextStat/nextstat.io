#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/nsr_cran_check_clean.sh

Runs a "clean" CRAN-style check for bindings/ns-r with required tooling present:
  - pandoc
  - checkbashisms
  - R packages: testthat, knitr, rmarkdown

The script builds from source and runs:
  R CMD build bindings/ns-r
  R CMD check --as-cran --no-manual <tarball>
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
pkg_dir="${repo_root}/bindings/ns-r"
work_dir="${repo_root}/tmp/r-cran-clean"

if ! command -v R >/dev/null 2>&1; then
  echo "ERROR: R is not installed or not on PATH" >&2
  exit 1
fi
if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc is required for clean R CMD check logs" >&2
  exit 1
fi
if ! command -v checkbashisms >/dev/null 2>&1; then
  echo "ERROR: checkbashisms is required for clean R CMD check logs" >&2
  exit 1
fi

Rscript -e '
required <- c("testthat", "knitr", "rmarkdown")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop(sprintf("Missing R packages: %s", paste(missing, collapse = ", ")), call. = FALSE)
}
'

bash "${repo_root}/scripts/nsr_vendor_sync.sh" --check

mkdir -p "${work_dir}"
cd "${work_dir}"

rm -rf nextstat.Rcheck
R CMD build "${pkg_dir}"
tarball="$(ls -1t nextstat_*.tar.gz | head -n1)"
R CMD check --as-cran --no-manual "${tarball}"
