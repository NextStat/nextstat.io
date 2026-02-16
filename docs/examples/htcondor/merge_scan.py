#!/usr/bin/env python3
"""Merge individual scan point JSON files into a single scan result.

Usage:
    python3 merge_scan.py scan_*.json -o scan_merged.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Merge scan point JSONs")
    parser.add_argument("files", nargs="+", help="scan_*.json files")
    parser.add_argument("-o", "--output", default="scan_merged.json")
    args = parser.parse_args()

    points = []
    for path in sorted(args.files):
        with open(path) as f:
            points.append(json.load(f))

    points.sort(key=lambda p: p["mu"])

    merged = {
        "n_points": len(points),
        "mu_values": [p["mu"] for p in points],
        "points": points,
    }

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(points)} scan points -> {args.output}")


if __name__ == "__main__":
    main()
