#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${1:-8000}"
echo "Serving playground at http://localhost:${PORT}/"
python3 -m http.server --directory playground "$PORT"

