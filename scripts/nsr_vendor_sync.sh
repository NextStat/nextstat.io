#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/nsr_vendor_sync.sh
  bash scripts/nsr_vendor_sync.sh --check

Syncs vendored Rust crates used by bindings/ns-r from root /crates into:
  bindings/ns-r/src/crates

Also syncs:
  Cargo.lock -> bindings/ns-r/src/Cargo.lock

Modes:
  (default)   apply sync
  --check     verify no drift (non-zero exit on mismatch)
EOF
}

mode="sync"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--check" ]]; then
  mode="check"
  shift
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

source_crates_dir="${repo_root}/crates"
vendor_root="${repo_root}/bindings/ns-r/src"
vendor_crates_dir="${vendor_root}/crates"
source_lock="${repo_root}/Cargo.lock"
vendor_lock="${vendor_root}/Cargo.lock"

vendored_crates=(
  ns-ad
  ns-compute
  ns-core
  ns-inference
  ns-prob
  ns-root
  ns-translate
  ns-unbinned
  ns-zstd
)

exclude_args=(
  --exclude ".git"
  --exclude ".github"
  --exclude ".DS_Store"
  --exclude "target"
  --exclude ".cargo-ok"
)

for crate in "${vendored_crates[@]}"; do
  if [[ ! -d "${source_crates_dir}/${crate}" ]]; then
    echo "ERROR: missing source crate: ${source_crates_dir}/${crate}" >&2
    exit 1
  fi
done

if [[ ! -f "${source_lock}" ]]; then
  echo "ERROR: missing source Cargo.lock: ${source_lock}" >&2
  exit 1
fi

is_vendored_crate() {
  local needle="$1"
  local crate
  for crate in "${vendored_crates[@]}"; do
    if [[ "${crate}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "${mode}" == "check" ]]; then
  drift=0

  for crate in "${vendored_crates[@]}"; do
    src="${source_crates_dir}/${crate}/"
    dst="${vendor_crates_dir}/${crate}/"
    rsync_out="$(
      rsync -ani --delete "${exclude_args[@]}" "${src}" "${dst}" || true
    )"
    if [[ -n "${rsync_out}" ]]; then
      echo "DRIFT: bindings/ns-r/src/crates/${crate}" >&2
      drift=1
    fi
  done

  if [[ -d "${vendor_crates_dir}" ]]; then
    while IFS= read -r extra_dir; do
      extra_name="$(basename "${extra_dir}")"
      if ! is_vendored_crate "${extra_name}"; then
        echo "DRIFT: extra vendored crate directory: bindings/ns-r/src/crates/${extra_name}" >&2
        drift=1
      fi
    done < <(find "${vendor_crates_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
  fi

  if ! cmp -s "${source_lock}" "${vendor_lock}"; then
    echo "DRIFT: bindings/ns-r/src/Cargo.lock differs from root Cargo.lock" >&2
    drift=1
  fi

  if [[ "${drift}" -ne 0 ]]; then
    echo >&2
    echo "Run to sync:" >&2
    echo "  bash scripts/nsr_vendor_sync.sh" >&2
    exit 1
  fi

  echo "OK: bindings/ns-r vendored crates are in sync."
  exit 0
fi

mkdir -p "${vendor_crates_dir}"

for crate in "${vendored_crates[@]}"; do
  src="${source_crates_dir}/${crate}/"
  dst="${vendor_crates_dir}/${crate}/"
  rsync -a --delete "${exclude_args[@]}" "${src}" "${dst}"
done

if [[ -d "${vendor_crates_dir}" ]]; then
  while IFS= read -r existing_dir; do
    existing_name="$(basename "${existing_dir}")"
    if ! is_vendored_crate "${existing_name}"; then
      rm -rf "${existing_dir}"
      echo "Pruned extra vendored crate: bindings/ns-r/src/crates/${existing_name}"
    fi
  done < <(find "${vendor_crates_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

cp "${source_lock}" "${vendor_lock}"
echo "Synced bindings/ns-r vendored crates and Cargo.lock."
