#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$ROOT_DIR/.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
MLX_ROOT="$ROOT_DIR/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
OUT_DIR="$ROOT_DIR/.build/release"

if [[ ! -d "$KERNEL_DIR" ]]; then
  echo "KERNEL_DIR not found: $KERNEL_DIR" >&2
  echo "Run 'swift build -c release' first." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

AIR_DIR="$ROOT_DIR/.build/tmp_metal_air"
rm -rf "$AIR_DIR"
mkdir -p "$AIR_DIR"

METAL_FLAGS=(
  -x metal
  -Wall -Wextra
  -fno-fast-math
  -Wno-c++17-extensions
  -Wno-c++20-extensions
  "-mmacosx-version-min=14.0"
)

# Prefer a locally mounted MetalToolchain if available (Xcode may not ship it by default).
CANDIDATE_TOOLCHAINS=()
if [[ -n "${METAL_TOOLCHAIN_ROOT:-}" ]]; then
  CANDIDATE_TOOLCHAINS+=("$METAL_TOOLCHAIN_ROOT")
fi
CANDIDATE_TOOLCHAINS+=(
  "/Library/Developer/Toolchains/Metal.xctoolchain"
  "$HOME/Library/Developer/Toolchains/Metal.xctoolchain"
  "/tmp/metaltoolchain_dmg/Metal.xctoolchain"
)

FOUND_TOOLCHAIN=""
for tc in "${CANDIDATE_TOOLCHAINS[@]}"; do
  if [[ -x "$tc/usr/bin/metal" && -x "$tc/usr/bin/metallib" ]]; then
    FOUND_TOOLCHAIN="$tc"
    break
  fi
done

if [[ -n "$FOUND_TOOLCHAIN" ]]; then
  METAL_CMD=("$FOUND_TOOLCHAIN/usr/bin/metal")
  METALLIB_CMD=("$FOUND_TOOLCHAIN/usr/bin/metallib")
else
  # Fall back to Xcode toolchain via xcrun.
  if ! xcrun -sdk macosx metal -v >/dev/null 2>&1; then
    echo "ERROR: Metal compiler is not available via xcrun." >&2
    echo "Fix options:" >&2
    echo "  1) Install/enable MetalToolchain in Xcode (Apple) so 'xcrun metal' works." >&2
    echo "  2) Mount MetalToolchain DMG and export METAL_TOOLCHAIN_ROOT=/path/to/Metal.xctoolchain" >&2
    echo "     (This script also checks /tmp/metaltoolchain_dmg/Metal.xctoolchain)." >&2
    exit 2
  fi
  METAL_CMD=(xcrun -sdk macosx metal)
  METALLIB_CMD=(xcrun -sdk macosx metallib)
fi

# Compile all .metal sources under kernels/ (fast enough; avoids missing a required kernel).
METAL_SRCS_FILE="$AIR_DIR/metal_srcs.txt"
find "$KERNEL_DIR" -type f -name "*.metal" \
  ! -name "fence.metal" \
  ! -name "*_nax.metal" \
  | sort > "$METAL_SRCS_FILE"
N_SRCS="$(wc -l < "$METAL_SRCS_FILE" | tr -d ' ')"

if [[ "$N_SRCS" -eq 0 ]]; then
  echo "No .metal sources found under: $KERNEL_DIR" >&2
  exit 2
fi

echo "Compiling $N_SRCS .metal files..." >&2

AIR_FILES=()
while IFS= read -r src; do
  rel="${src#$KERNEL_DIR/}"
  base="${rel//\//_}"
  air="$AIR_DIR/${base%.metal}.air"
  "${METAL_CMD[@]}" "${METAL_FLAGS[@]}" -c "$src" -I"$MLX_ROOT" -I"$KERNEL_DIR" -o "$air"
  AIR_FILES+=("$air")
done < "$METAL_SRCS_FILE"

echo "Linking mlx.metallib..." >&2
 "${METALLIB_CMD[@]}" "${AIR_FILES[@]}" -o "$OUT_DIR/mlx.metallib"

echo "Wrote: $OUT_DIR/mlx.metallib" >&2
