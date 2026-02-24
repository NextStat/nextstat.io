#!/usr/bin/env bash
set -euo pipefail

# One-shot helper for macOS dev machines where `xcrun metal` is missing.
# It downloads the MetalToolchain asset (via Xcode), mounts the DMG,
# and builds `mlx.metallib` using the toolchain directly.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if xcrun -sdk macosx metal -v >/dev/null 2>&1; then
  echo "xcrun metal is available; building mlx.metallib with system toolchain." >&2
  exec "$ROOT_DIR/build_mlx_metallib.sh"
fi

echo "xcrun metal not available; downloading MetalToolchain via Xcode..." >&2
xcodebuild -downloadComponent MetalToolchain
xcodebuild -runFirstLaunch

# Find the newest MetalToolchain DMG in AssetsV2.
DMG="$(ls -t /System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain/*.asset/AssetData/Restore/*.dmg 2>/dev/null | head -n 1 || true)"
if [[ -z "$DMG" ]]; then
  echo "ERROR: Could not locate downloaded MetalToolchain DMG under /System/Library/AssetsV2." >&2
  exit 2
fi

echo "Mounting: $DMG" >&2
MOUNT=/tmp/metaltoolchain_dmg
mkdir -p "$MOUNT"
hdiutil attach "$DMG" -nobrowse -readonly -mountpoint "$MOUNT" >/dev/null

export METAL_TOOLCHAIN_ROOT="$MOUNT/Metal.xctoolchain"

echo "Using METAL_TOOLCHAIN_ROOT=$METAL_TOOLCHAIN_ROOT" >&2
"$ROOT_DIR/build_mlx_metallib.sh"

# Best-effort unmount.
hdiutil detach "$MOUNT" >/dev/null || true
