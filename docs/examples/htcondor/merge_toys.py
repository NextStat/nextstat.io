#!/usr/bin/env python3
"""Merge individual toy job JSON files into a single result.

Usage:
    python3 merge_toys.py [--glob 'toys_*.json'] [-o toys_merged.json]
"""

import argparse
import glob
import json


def main():
    parser = argparse.ArgumentParser(description="Merge toy job JSONs")
    parser.add_argument("--glob", default="toys_*.json", help="Glob pattern")
    parser.add_argument("-o", "--output", default="toys_merged.json")
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No files matching '{args.glob}'")
        return

    all_toys = []
    total_n_toys = 0
    for path in files:
        with open(path) as f:
            data = json.load(f)
        all_toys.append(data)
        total_n_toys += data.get("n_toys", 0)

    merged = {
        "n_jobs": len(all_toys),
        "total_n_toys": total_n_toys,
        "jobs": all_toys,
    }

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(all_toys)} jobs ({total_n_toys} total toys) -> {args.output}")


if __name__ == "__main__":
    main()
