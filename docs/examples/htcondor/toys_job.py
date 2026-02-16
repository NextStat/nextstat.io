#!/usr/bin/env python3
"""Run a batch of toy CLs — called by toys_batch.sub via wrapper.sh.

Usage:
    python3 toys_job.py <process_id> <n_toys> <base_seed>

Reads workspace.json from CWD (transferred by HTCondor).
Writes toys_<process_id>.json with toy test statistics.
"""

import json
import sys

import nextstat


def main():
    process_id = int(sys.argv[1])
    n_toys = int(sys.argv[2])
    base_seed = int(sys.argv[3])

    # Each job gets a unique seed: spacing of 1000 avoids overlap for n_toys <= 1000.
    # If n_toys > 1000, increase the multiplier accordingly.
    seed = base_seed + process_id * 1000

    with open("workspace.json") as f:
        ws_str = f.read()

    model = nextstat.from_pyhf(ws_str)
    result = nextstat.hypotest_toys(
        1.0,
        model,
        n_toys=n_toys,
        seed=seed,
        expected_set=True,
    )

    out = {
        "process_id": process_id,
        "n_toys": n_toys,
        "seed": seed,
    }
    out.update(result)

    with open(f"toys_{process_id}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Job {process_id}: {n_toys} toys completed (seed={seed})")


if __name__ == "__main__":
    main()
