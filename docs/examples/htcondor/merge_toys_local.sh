#!/bin/bash
# Merge toy results on the submit node (universe = local).
# Assumes nextstat-env is available and toys_*.json are in CWD.

set -euo pipefail

source nextstat-env/bin/activate 2>/dev/null || export PATH="$(pwd)/nextstat-env/bin:$PATH"

python3 merge_toys.py --glob "toys_*.json" -o toys_merged.json
