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
//! # Histogram Table Schema
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
//! # Modules
//!
//! - [`ingest`] — Arrow RecordBatch → pyhf [`Workspace`](crate::pyhf::Workspace)
//! - [`export`] — [`HistFactoryModel`](crate::pyhf::HistFactoryModel) → Arrow RecordBatch
//! - [`parquet`] — Read/write Parquet files

pub mod export;
pub mod ingest;
pub mod parquet;

#[cfg(test)]
mod tests;
