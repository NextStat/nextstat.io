#!/usr/bin/env python3
"""
Example: write a histogram table to Parquet + manifest, then ingest with NextStat.

Requires:
  pip install "nextstat[io]"
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore

import nextstat
from nextstat.arrow_io import validate_histograms_parquet_manifest, write_histograms_parquet


def main() -> int:
    out = Path("tmp/arrow_parquet_example").resolve()
    out.mkdir(parents=True, exist_ok=True)

    pq_path = out / "histograms.parquet"

    table = pa.table(
        {
            "channel": ["SR", "SR", "CR"],
            "sample": ["signal", "background", "background"],
            "yields": [[5.0, 10.0, 15.0], [100.0, 200.0, 150.0], [500.0, 600.0]],
        }
    )

    manifest = write_histograms_parquet(table, pq_path, compression="zstd")
    validate_histograms_parquet_manifest(manifest)

    model = nextstat.from_parquet(str(pq_path), poi="mu")
    fit = nextstat.fit(model)
    print("converged:", fit.converged, "nll:", fit.nll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

