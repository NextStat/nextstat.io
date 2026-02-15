#!/usr/bin/env python3
"""Single profile scan point — called by scan_array.sub via wrapper.sh.

Usage:
    python3 scan_point.py <process_id>

Reads workspace.json from CWD (transferred by HTCondor).
Writes scan_<process_id>.json with the scan result at one mu value.
"""

import json
import sys

import nextstat

N_POINTS = 201  # Must match "queue N" in scan_array.sub
MU_MIN = 0.0
MU_MAX = 5.0


def main():
    process_id = int(sys.argv[1])
    mu = MU_MIN + process_id * (MU_MAX - MU_MIN) / (N_POINTS - 1)

    with open("workspace.json") as f:
        ws_str = f.read()

    model = nextstat.from_pyhf(ws_str)
    result = nextstat.profile_scan(model, [mu])

    out = {"process_id": process_id, "mu": mu}
    out.update(result)

    with open(f"scan_{process_id}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
