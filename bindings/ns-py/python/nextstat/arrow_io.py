"""Arrow/Parquet authoring helpers (optional).

NextStat can *ingest* Parquet histogram tables via the Rust `parquet` crate
(`nextstat.from_parquet`) without requiring PyArrow on the machine.

However, to *create* Parquet/Arrow files from Python, you typically use PyArrow
or an Arrow-backed dataframe library (Polars, DuckDB, Spark). This module
provides:

- schema validation for the histogram table contract
- manifest writing/validation for reproducible pipelines

Requires `pyarrow` (install with: `pip install "nextstat[io]"`).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


HISTOGRAM_TABLE_MANIFEST_V1 = "nextstat.histograms_parquet_manifest.v1"


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_utc_iso() -> str:
    # Keep it JSON-friendly and stable (no timezone locale surprises).
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class HistogramTableStats:
    has_stat_error: bool
    channels: Dict[str, int]  # channel -> n_bins
    n_rows: int


@dataclass(frozen=True)
class ModifiersTableStats:
    n_rows: int


def _require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
    except Exception as e:
        raise ImportError(
            "nextstat.arrow_io requires pyarrow. Install: pip install 'nextstat[io]'"
        ) from e
    return pa


def validate_histogram_table(table) -> HistogramTableStats:
    """Validate the histogram table contract.

    Required columns:
    - channel: Utf8/LargeUtf8
    - sample:  Utf8/LargeUtf8
    - yields:  List<Float64> or LargeList<Float64>

    Optional:
    - stat_error: List<Float64> or LargeList<Float64>

    Additionally, for each *channel*, all rows must have consistent bin count
    (length of yields list).
    """
    pa = _require_pyarrow()

    if isinstance(table, pa.RecordBatch):
        table = pa.Table.from_batches([table])

    schema = table.schema
    names = set(schema.names)
    for col in ("channel", "sample", "yields"):
        if col not in names:
            raise ValueError(f"missing required column: {col!r}")

    def _is_utf8(t) -> bool:
        return pa.types.is_string(t) or pa.types.is_large_string(t)

    def _is_list_f64(t) -> bool:
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            return pa.types.is_float64(t.value_type)
        return False

    if not _is_utf8(schema.field("channel").type):
        raise TypeError(f"column 'channel' must be Utf8, got {schema.field('channel').type}")
    if not _is_utf8(schema.field("sample").type):
        raise TypeError(f"column 'sample' must be Utf8, got {schema.field('sample').type}")
    if not _is_list_f64(schema.field("yields").type):
        raise TypeError(
            f"column 'yields' must be List<Float64>, got {schema.field('yields').type}"
        )

    has_stat_error = "stat_error" in names
    if has_stat_error and not _is_list_f64(schema.field("stat_error").type):
        raise TypeError(
            f"column 'stat_error' must be List<Float64>, got {schema.field('stat_error').type}"
        )

    # Per-channel bin count consistency.
    channel_arr = table.column("channel").combine_chunks()
    yields_arr = table.column("yields").combine_chunks()

    # yields_arr is a ListArray; for each row i, yields_arr[i] is a float array.
    chan_bins: Dict[str, int] = {}
    for i in range(table.num_rows):
        ch = channel_arr[i].as_py()
        y = yields_arr[i].as_py()
        if y is None:
            raise ValueError(f"row {i}: yields is null")
        n = len(y)
        if n <= 0:
            raise ValueError(f"row {i}: yields is empty for channel={ch!r}")
        prev = chan_bins.get(ch)
        if prev is None:
            chan_bins[ch] = n
        elif prev != n:
            raise ValueError(
                f"inconsistent bin count in channel={ch!r}: expected {prev}, got {n}"
            )

    return HistogramTableStats(
        has_stat_error=has_stat_error,
        channels=chan_bins,
        n_rows=int(table.num_rows),
    )


def validate_modifiers_table(table) -> ModifiersTableStats:
    """Validate the modifiers table contract (binned Parquet v2).

    Required columns:
    - channel: Utf8/LargeUtf8
    - sample: Utf8/LargeUtf8
    - modifier_name: Utf8/LargeUtf8
    - modifier_type: Utf8/LargeUtf8

    Optional:
    - data_hi: List<Float64> or LargeList<Float64>
    - data_lo: List<Float64> or LargeList<Float64>
    - data: List<Float64> or LargeList<Float64>
    """
    pa = _require_pyarrow()

    if isinstance(table, pa.RecordBatch):
        table = pa.Table.from_batches([table])

    schema = table.schema
    names = set(schema.names)
    for col in ("channel", "sample", "modifier_name", "modifier_type"):
        if col not in names:
            raise ValueError(f"missing required column: {col!r}")

    def _is_utf8(t) -> bool:
        return pa.types.is_string(t) or pa.types.is_large_string(t)

    def _is_list_f64(t) -> bool:
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            return pa.types.is_float64(t.value_type)
        return False

    for col in ("channel", "sample", "modifier_name", "modifier_type"):
        if not _is_utf8(schema.field(col).type):
            raise TypeError(f"column {col!r} must be Utf8, got {schema.field(col).type}")

    for col in ("data_hi", "data_lo", "data"):
        if col in names and not _is_list_f64(schema.field(col).type):
            raise TypeError(f"column {col!r} must be List<Float64>, got {schema.field(col).type}")

    return ModifiersTableStats(n_rows=int(table.num_rows))


def write_histograms_parquet(
    table,
    path: str | Path,
    *,
    compression: str = "zstd",
    write_manifest: bool = True,
    manifest_path: str | Path | None = None,
    poi: str = "mu",
    observations: Optional[dict[str, list[float]]] = None,
    observations_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a histogram table to Parquet and optionally write a manifest JSON.

    Returns the manifest dict (even if write_manifest=False).
    """
    pa = _require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore

    if isinstance(path, str):
        path = Path(path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(table, pa.RecordBatch):
        table = pa.Table.from_batches([table])

    stats = validate_histogram_table(table)

    pq.write_table(table, str(path), compression=compression)
    pq_sha256 = _sha256_file(path)

    obs_ref = None
    if observations is not None:
        if observations_path is None:
            observations_path = path.with_suffix(path.suffix + ".observations.json")
        obs_path = Path(observations_path).resolve()
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        obs_path.write_text(_pretty_json(observations), encoding="utf-8")
        obs_ref = str(obs_path)

    manifest: dict[str, Any] = {
        "schema_version": HISTOGRAM_TABLE_MANIFEST_V1,
        "created_at_utc": _now_utc_iso(),
        "poi": str(poi),
        "parquet_path": str(path),
        "parquet_sha256": pq_sha256,
        "parquet_compression": str(compression),
        "table_schema": {
            "columns": [
                {"name": "channel", "type": "utf8", "required": True},
                {"name": "sample", "type": "utf8", "required": True},
                {"name": "yields", "type": "list<float64>", "required": True},
                {"name": "stat_error", "type": "list<float64>", "required": False},
            ],
            # Keep an implementation-independent fingerprint of the Arrow schema.
            "arrow_schema_str": str(table.schema),
            "arrow_schema_sha256": hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest(),
        },
        "stats": {
            "n_rows": stats.n_rows,
            "has_stat_error": stats.has_stat_error,
            "channels": [{"name": k, "n_bins": int(v)} for k, v in sorted(stats.channels.items())],
        },
        "observations_path": obs_ref,
    }

    if write_manifest:
        if manifest_path is None:
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        mpath = Path(manifest_path).resolve()
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(_pretty_json(manifest), encoding="utf-8")

    return manifest


def validate_histograms_parquet_manifest(
    manifest: dict[str, Any],
    *,
    check_sha256: bool = True,
) -> None:
    """Validate a manifest against the referenced Parquet file (best-effort)."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore

    if manifest.get("schema_version") != HISTOGRAM_TABLE_MANIFEST_V1:
        raise ValueError("unsupported manifest schema_version")

    pq_path = Path(manifest["parquet_path"]).resolve()
    if not pq_path.exists():
        raise FileNotFoundError(str(pq_path))

    if check_sha256:
        want = str(manifest.get("parquet_sha256") or "")
        got = _sha256_file(pq_path)
        if want and got != want:
            raise ValueError(f"parquet sha256 mismatch: want={want} got={got}")

    table = pq.read_table(str(pq_path))
    stats = validate_histogram_table(table)

    want_rows = int(((manifest.get("stats") or {}).get("n_rows")) or 0)
    if want_rows and stats.n_rows != want_rows:
        raise ValueError(f"n_rows mismatch: want={want_rows} got={stats.n_rows}")

    want_ch = (manifest.get("stats") or {}).get("channels") or []
    want_map = {c["name"]: int(c["n_bins"]) for c in want_ch if isinstance(c, dict) and "name" in c}
    if want_map:
        if stats.channels != want_map:
            raise ValueError(f"channel bin counts mismatch: want={want_map} got={stats.channels}")

    # Optional observations payload sanity.
    obs_path = manifest.get("observations_path")
    if obs_path:
        op = Path(obs_path).resolve()
        if not op.exists():
            raise FileNotFoundError(str(op))
        obs = json.loads(op.read_text(encoding="utf-8"))
        if not isinstance(obs, dict):
            raise TypeError("observations JSON must be an object")

        for ch, bins in obs.items():
            if ch not in stats.channels:
                raise ValueError(f"observations contain unknown channel: {ch!r}")
            if not isinstance(bins, list):
                raise TypeError(f"observations[{ch!r}] must be a list")
            if len(bins) != stats.channels[ch]:
                raise ValueError(
                    f"observations bin count mismatch for {ch!r}: "
                    f"want {stats.channels[ch]}, got {len(bins)}"
                )


# ---------------------------------------------------------------------------
# Unbinned event-level Parquet (nextstat_unbinned_events_v1)
# ---------------------------------------------------------------------------

UNBINNED_EVENTS_SCHEMA_V1 = "nextstat_unbinned_events_v1"
_WEIGHT_COLUMN = "_weight"
_CHANNEL_COLUMN = "_channel"


@dataclass(frozen=True)
class EventTableStats:
    observable_names: list[str]
    has_weight: bool
    has_channel: bool
    channels: Dict[str, int]  # channel -> n_events (or {"_default": n} if no _channel col)
    n_rows: int


def validate_event_table(table) -> EventTableStats:
    """Validate an unbinned event table contract.

    Required: at least one Float64 column that is not ``_weight`` / ``_channel``.

    Optional:
    - ``_weight``:  Float64, per-event weight.
    - ``_channel``: Utf8, channel label (for multi-channel files).
    """
    pa = _require_pyarrow()

    if isinstance(table, pa.RecordBatch):
        table = pa.Table.from_batches([table])

    schema = table.schema
    names = set(schema.names)

    observable_names: list[str] = []
    for field in schema:
        if field.name in (_WEIGHT_COLUMN, _CHANNEL_COLUMN):
            continue
        if not pa.types.is_float64(field.type):
            raise TypeError(
                f"observable column '{field.name}' must be Float64, got {field.type}"
            )
        observable_names.append(field.name)

    if not observable_names:
        raise ValueError("event table must have at least one Float64 observable column")

    has_weight = _WEIGHT_COLUMN in names
    if has_weight and not pa.types.is_float64(schema.field(_WEIGHT_COLUMN).type):
        raise TypeError(
            f"column '{_WEIGHT_COLUMN}' must be Float64, "
            f"got {schema.field(_WEIGHT_COLUMN).type}"
        )

    has_channel = _CHANNEL_COLUMN in names
    if has_channel:
        ct = schema.field(_CHANNEL_COLUMN).type
        if not (pa.types.is_string(ct) or pa.types.is_large_string(ct)):
            raise TypeError(f"column '{_CHANNEL_COLUMN}' must be Utf8, got {ct}")

    channels: Dict[str, int] = {}
    if has_channel:
        ch_arr = table.column(_CHANNEL_COLUMN).combine_chunks()
        for i in range(table.num_rows):
            ch = ch_arr[i].as_py()
            channels[ch] = channels.get(ch, 0) + 1
    else:
        channels["_default"] = int(table.num_rows)

    return EventTableStats(
        observable_names=observable_names,
        has_weight=has_weight,
        has_channel=has_channel,
        channels=channels,
        n_rows=int(table.num_rows),
    )


def write_events_parquet(
    table,
    path: str | Path,
    *,
    observables: Optional[list[dict[str, Any]]] = None,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Write an unbinned event table to Parquet with NextStat metadata.

    Parameters
    ----------
    table : pyarrow.Table or pyarrow.RecordBatch
        Event data.  Must pass :func:`validate_event_table`.
    path : str or Path
        Output Parquet file path.
    observables : list of dict, optional
        Observable metadata: ``[{"name": "mass", "bounds": [100, 180]}, ...]``.
        If *None*, all Float64 columns are observables with ``[-inf, inf]`` bounds.
    compression : str
        Parquet compression codec (default ``"zstd"``).

    Returns
    -------
    dict
        Metadata dict written into the Parquet file footer.
    """
    pa = _require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore

    if isinstance(path, str):
        path = Path(path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(table, pa.RecordBatch):
        table = pa.Table.from_batches([table])

    stats = validate_event_table(table)

    if observables is None:
        observables = [
            {"name": n, "bounds": [float("-inf"), float("inf")]}
            for n in stats.observable_names
        ]

    meta_kv = {
        UNBINNED_EVENTS_SCHEMA_V1.encode(): b"",  # marker
        b"nextstat.schema_version": UNBINNED_EVENTS_SCHEMA_V1.encode(),
        b"nextstat.observables": json.dumps(observables, separators=(",", ":")).encode(),
    }

    existing = table.schema.metadata or {}
    merged_meta = {**existing, **meta_kv}
    table = table.replace_schema_metadata(merged_meta)

    pq.write_table(table, str(path), compression=compression)

    return {
        "schema_version": UNBINNED_EVENTS_SCHEMA_V1,
        "observables": observables,
        "stats": {
            "n_rows": stats.n_rows,
            "observable_names": stats.observable_names,
            "has_weight": stats.has_weight,
            "has_channel": stats.has_channel,
            "channels": stats.channels,
        },
        "parquet_path": str(path),
        "parquet_compression": compression,
    }


def load_parquet_as_histfactory_model(
    path: str | Path,
    *,
    poi: str = "mu",
    observations: Optional[dict[str, list[float]]] = None,
):
    """Convenience: validate Parquet schema with PyArrow, then call nextstat.from_parquet()."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore

    from . import from_parquet as _from_parquet  # lazy import to keep module import light

    p = Path(path).resolve()
    table = pq.read_table(str(p))
    validate_histogram_table(table)
    return _from_parquet(str(p), poi=poi, observations=observations)


def load_parquet_v2_as_histfactory_model(
    yields_path: str | Path,
    modifiers_path: str | Path,
    *,
    poi: str = "mu",
    observations: Optional[dict[str, list[float]]] = None,
):
    """Convenience: validate Parquet v2 schemas, then call nextstat.from_parquet_with_modifiers()."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore

    from . import (  # lazy import to keep module import light
        from_parquet_with_modifiers as _from_parquet_with_modifiers,
    )

    y = Path(yields_path).resolve()
    m = Path(modifiers_path).resolve()
    yields_table = pq.read_table(str(y))
    modifiers_table = pq.read_table(str(m))

    validate_histogram_table(yields_table)
    validate_modifiers_table(modifiers_table)

    return _from_parquet_with_modifiers(str(y), str(m), poi=poi, observations=observations)
