#!/bin/bash
# NextStat HTCondor wrapper — portable venv (Strategy B)
#
# Unpacks a pre-built virtualenv tarball and runs a Python analysis script.
# Used with transfer_input_files = nextstat-env.tar.gz, <your_script.py>, ...
#
# Usage in .sub file:
#   executable = wrapper.sh
#   arguments  = my_analysis.py $(Process)

set -euo pipefail

SCRIPT="$1"
shift

# Unpack venv (transferred by HTCondor)
tar xzf nextstat-env.tar.gz

# Activate
export PATH="$(pwd)/nextstat-env/bin:$PATH"
export VIRTUAL_ENV="$(pwd)/nextstat-env"

# Run analysis
python3 "$SCRIPT" "$@"
