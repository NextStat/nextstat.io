//! Parquet file read/write for histogram data.
//!
//! Uses the `parquet` crate to read/write Parquet files containing
//! histogram tables in the schema expected by [`super::ingest`].

use std::fs::File;
use std::path::Path;

use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::export::ArrowExportError;
use super::ingest::{ArrowIngestConfig, ArrowIngestError};
use crate::pyhf::schema::Workspace;

/// Error type for Parquet operations.
#[derive(Debug, thiserror::Error)]
pub enum ParquetError {
    #[error("Parquet read/write error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Ingest error: {0}")]
    Ingest(#[from] ArrowIngestError),

    #[error("Export error: {0}")]
    Export(#[from] ArrowExportError),
}

/// Read a Parquet file into Arrow RecordBatches.
pub fn read_parquet_batches(path: &Path) -> Result<Vec<RecordBatch>, ParquetError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok(batches?)
}

/// Read Parquet data from an in-memory byte slice into Arrow RecordBatches.
pub fn read_parquet_bytes(data: &[u8]) -> Result<Vec<RecordBatch>, ParquetError> {
    // `ParquetRecordBatchReaderBuilder::try_new` accepts `impl ChunkReader`.
    // `bytes::Bytes` implements `ChunkReader`. The `parquet` crate depends on
    // `bytes` internally; we add it as an explicit dep for this one call.
    let buf = bytes::Bytes::copy_from_slice(data);
    let builder = ParquetRecordBatchReaderBuilder::try_new(buf)?;
    let reader = builder.build()?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok(batches?)
}

/// Read a Parquet file and convert to a pyhf Workspace.
///
/// The Parquet file must contain the histogram table schema:
/// `channel (Utf8)`, `sample (Utf8)`, `yields (List<Float64>)`,
/// optionally `stat_error (List<Float64>)`.
pub fn from_parquet(path: &Path, config: &ArrowIngestConfig) -> Result<Workspace, ParquetError> {
    let batches = read_parquet_batches(path)?;
    let ws = super::ingest::from_record_batches(&batches, config)?;
    Ok(ws)
}

/// Read Parquet bytes and convert to a pyhf Workspace.
pub fn from_parquet_bytes(
    data: &[u8],
    config: &ArrowIngestConfig,
) -> Result<Workspace, ParquetError> {
    let batches = read_parquet_bytes(data)?;
    let ws = super::ingest::from_record_batches(&batches, config)?;
    Ok(ws)
}

fn default_compression() -> Compression {
    // Keep `arrow-io` usable without `parquet/zstd` (which requires zstd at compile time).
    // When enabled, prefer Zstd; otherwise fall back to Snappy (enabled via `parquet/snap`).
    #[cfg(feature = "arrow-io-zstd")]
    {
        Compression::ZSTD(Default::default())
    }
    #[cfg(not(feature = "arrow-io-zstd"))]
    {
        Compression::SNAPPY
    }
}

/// Write Arrow RecordBatches to a Parquet file.
///
/// Uses Zstd if the `arrow-io-zstd` feature is enabled, otherwise Snappy.
pub fn write_parquet(path: &Path, batches: &[RecordBatch]) -> Result<(), ParquetError> {
    if batches.is_empty() {
        return Ok(());
    }

    let schema = batches[0].schema();
    let props = WriterProperties::builder().set_compression(default_compression()).build();

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;

    for batch in batches {
        writer.write(batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Write Arrow RecordBatches to Parquet bytes in memory.
pub fn write_parquet_bytes(batches: &[RecordBatch]) -> Result<Vec<u8>, ParquetError> {
    if batches.is_empty() {
        return Ok(vec![]);
    }

    let schema = batches[0].schema();
    let props = WriterProperties::builder().set_compression(default_compression()).build();

    let mut buf = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
    }

    Ok(buf)
}
