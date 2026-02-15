"""Create histograms.parquet from the example CSV table.

Requires: pip install pyarrow
"""
from pathlib import Path
import ast

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

src = Path(__file__).with_name("histograms_table.example.csv")
tbl = pv.read_csv(src)

channels = tbl.column("channel").to_pylist()
samples = tbl.column("sample").to_pylist()
yields = [ast.literal_eval(v) for v in tbl.column("yields").to_pylist()]
stat_error = [ast.literal_eval(v) for v in tbl.column("stat_error").to_pylist()]

out = pa.table({
    "channel": channels,
    "sample": samples,
    "yields": yields,
    "stat_error": stat_error,
})

pq.write_table(out, Path(__file__).with_name("histograms.parquet"), compression="snappy")
print("wrote", Path(__file__).with_name("histograms.parquet"))
