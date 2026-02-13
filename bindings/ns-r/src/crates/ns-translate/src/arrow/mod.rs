//! Apache Arrow / Parquet integration for NextStat.
//!
//! Provides zero-copy data interchange between NextStat's HistFactory model
//! and the Arrow columnar format used by Polars, DuckDB, Spark, and the
//! broader ML ecosystem.
//!
//! # Architecture
//!
//! The bridge uses Arrow IPC (Inter-Process Communication) as the serialization
//! format between Python (PyArrow) and Rust. This avoids pyo3 version conflicts
//! and requires only a single memcpy at the language boundary.
//!
//! ```text
//! Python (PyArrow)          Rust (arrow crate)
//! ────────────────          ──────────────────
//! table.serialize() ──IPC bytes──▶ ipc::StreamReader ──▶ RecordBatch
//!                                                           │
//!                                                    ingest / export
//!                                                           │
//! pa.ipc.open_stream() ◀──IPC bytes── ipc::StreamWriter ◀──┘
//! ```
//!
//! # Histogram Yields Table Schema
//!
//! The ingest path expects an Arrow table with the following columns:
//!
//! | Column       | Type              | Required | Description                         |
//! |-------------|-------------------|----------|-------------------------------------|
//! | `channel`   | `Utf8`            | yes      | Channel (region) name               |
//! | `sample`    | `Utf8`            | yes      | Sample (process) name               |
//! | `yields`    | `List<Float64>`   | yes      | Expected event counts per bin       |
//! | `stat_error`| `List<Float64>`   | no       | Per-bin statistical uncertainties    |
//!
//! # Modifier Table Schema (v2)
//!
//! One row per (channel, sample, modifier) triple:
//!
//! | Column          | Type            | Required | Description                              |
//! |-----------------|-----------------|----------|------------------------------------------|
//! | `channel`       | `Utf8`          | yes      | Channel name                             |
//! | `sample`        | `Utf8`          | yes      | Sample name                              |
//! | `modifier_name` | `Utf8`          | yes      | Modifier parameter name                  |
//! | `modifier_type` | `Utf8`          | yes      | pyhf type tag (normsys, histosys, …)     |
//! | `data_hi`       | `List<Float64>` | no       | Up-variation (histosys hi, normsys [hi]) |
//! | `data_lo`       | `List<Float64>` | no       | Down-variation (histosys lo, normsys [lo])|
//! | `data`          | `List<Float64>` | no       | Generic data (shapesys σ, staterror σ)   |
//!
//! # Modules
//!
//! - [`ingest`] — Arrow RecordBatch → pyhf [`Workspace`](crate::pyhf::Workspace)
//! - [`export`] — [`HistFactoryModel`](crate::pyhf::HistFactoryModel) / [`Workspace`](crate::pyhf::Workspace) → Arrow RecordBatch
//! - [`parquet`] — Read/write Parquet files (yields + modifiers roundtrip)
//! - [`results`] — Fit results / toy studies / scan points → Arrow / Parquet

pub mod export;
pub mod ingest;
pub mod parquet;
pub mod results;
#[cfg(feature = "arrow-io-unbinned")]
pub mod unbinned;

#[cfg(test)]
mod tests;
