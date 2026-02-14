#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "Missing wasm-bindgen CLI." >&2
  echo "Install: cargo install wasm-bindgen-cli --version 0.2.108" >&2
  exit 2
fi

echo "Building ns-wasm (release, wasm32-unknown-unknown)…"
# wasm-bindgen's runtime enables its externref-table implementation via
# `CARGO_CFG_TARGET_FEATURE` ("reference-types"). On Rust 1.93 this needs to be
# explicitly enabled via `RUSTFLAGS` for the wasm target, otherwise
# wasm-bindgen-cli fails during transform.
WASM_RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+reference-types"
RUSTFLAGS="$WASM_RUSTFLAGS" cargo build -p ns-wasm --target wasm32-unknown-unknown --profile release-wasm

# Resolve actual target-dir (may be overridden in .cargo/config.toml)
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])')"
IN_WASM="$TARGET_DIR/wasm32-unknown-unknown/release-wasm/ns_wasm.wasm"

if [ ! -f "$IN_WASM" ]; then
  echo "ERROR: WASM binary not found at $IN_WASM" >&2
  exit 1
fi
OUT_DIR="playground/pkg"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Running wasm-bindgen…"
wasm-bindgen "$IN_WASM" --target web --no-typescript --out-dir "$OUT_DIR"

echo "Done: $OUT_DIR"
