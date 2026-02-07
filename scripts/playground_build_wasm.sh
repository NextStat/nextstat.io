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
cargo build -p ns-wasm --target wasm32-unknown-unknown --release

IN_WASM="target/wasm32-unknown-unknown/release/ns_wasm.wasm"
OUT_DIR="playground/pkg"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Running wasm-bindgen…"
wasm-bindgen "$IN_WASM" --target web --no-typescript --out-dir "$OUT_DIR"

echo "Done: $OUT_DIR"

