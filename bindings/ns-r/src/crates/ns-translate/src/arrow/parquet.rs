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
use parquet::file::metadata::KeyValue;
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

/// Read a Parquet file with column projection (only read requested columns).
///
/// `columns`: list of column names to read. If empty, reads all columns.
/// `row_limit`: optional maximum number of rows to read.
pub fn read_parquet_projected(
    path: &Path,
    columns: &[&str],
    row_limit: Option<usize>,
) -> Result<Vec<RecordBatch>, ParquetError> {
    let file = File::open(path)?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    if !columns.is_empty() {
        let schema = builder.schema().clone();
        let mut indices = Vec::with_capacity(columns.len());
        for col_name in columns {
            if let Some((idx, _)) = schema.column_with_name(col_name) {
                indices.push(idx);
            }
        }
        let mask = parquet::arrow::ProjectionMask::leaves(builder.parquet_schema(), indices);
        builder = builder.with_projection(mask);
    }

    if let Some(limit) = row_limit {
        builder = builder.with_limit(limit);
    }

    let reader = builder.build()?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok(batches?)
}

/// Read Parquet bytes with column projection.
pub fn read_parquet_bytes_projected(
    data: &[u8],
    columns: &[&str],
    row_limit: Option<usize>,
) -> Result<Vec<RecordBatch>, ParquetError> {
    let buf = bytes::Bytes::copy_from_slice(data);
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(buf)?;

    if !columns.is_empty() {
        let schema = builder.schema().clone();
        let mut indices = Vec::with_capacity(columns.len());
        for col_name in columns {
            if let Some((idx, _)) = schema.column_with_name(col_name) {
                indices.push(idx);
            }
        }
        let mask = parquet::arrow::ProjectionMask::leaves(builder.parquet_schema(), indices);
        builder = builder.with_projection(mask);
    }

    if let Some(limit) = row_limit {
        builder = builder.with_limit(limit);
    }

    let reader = builder.build()?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok(batches?)
}

/// Memory-map a file and return it as a `Bytes` handle suitable for `ChunkReader`.
///
/// The mmap stays alive as long as the returned `Bytes` is alive.
/// On wasm32 this falls back to a regular read.
/// Newtype around `Mmap` that implements `AsRef<[u8]>` for `Bytes::from_owner`.
struct MmapOwner(memmap2::Mmap);

impl AsRef<[u8]> for MmapOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

fn mmap_file(path: &Path) -> Result<bytes::Bytes, ParquetError> {
    let file = File::open(path)?;
    // SAFETY: we treat the file as read-only and do not modify it while the
    // mmap is alive. The `Bytes` wrapper keeps the `MmapOwner` alive.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(bytes::Bytes::from_owner(MmapOwner(mmap)))
}

/// Read a Parquet file via mmap into Arrow RecordBatches (zero-copy, no read() syscalls).
///
/// Falls back to [`read_parquet_batches`] on platforms without mmap support.
pub fn read_parquet_mmap(path: &Path) -> Result<Vec<RecordBatch>, ParquetError> {
    let buf = mmap_file(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(buf)?;
    let reader = builder.build()?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok(batches?)
}

/// Read a Parquet file via mmap with column projection and row limit.
///
/// `columns`: column names to read (empty = all).
/// `row_limit`: optional maximum number of rows.
pub fn read_parquet_mmap_projected(
    path: &Path,
    columns: &[&str],
    row_limit: Option<usize>,
) -> Result<Vec<RecordBatch>, ParquetError> {
    let buf = mmap_file(path)?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(buf)?;

    if !columns.is_empty() {
        let schema = builder.schema().clone();
        let mut indices = Vec::with_capacity(columns.len());
        for col_name in columns {
            if let Some((idx, _)) = schema.column_with_name(col_name) {
                indices.push(idx);
            }
        }
        let mask = parquet::arrow::ProjectionMask::leaves(builder.parquet_schema(), indices);
        builder = builder.with_projection(mask);
    }

    if let Some(limit) = row_limit {
        builder = builder.with_limit(limit);
    }

    let reader = builder.build()?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok(batches?)
}

/// A column-level predicate for row group pruning.
///
/// Row groups whose `[min, max]` range for the named column does not overlap
/// `[lo, hi]` are skipped entirely — no page decode, no I/O.
#[derive(Debug, Clone)]
pub struct ColumnBound {
    /// Column name in the Parquet schema.
    pub column: String,
    /// Lower bound (inclusive).
    pub lo: f64,
    /// Upper bound (inclusive).
    pub hi: f64,
}

/// Read a Parquet file via mmap, skipping row groups that fail column bound predicates.
///
/// This is the fastest path for selective reads: mmap + row group pruning + projection + limit.
pub fn read_parquet_mmap_filtered(
    path: &Path,
    columns: &[&str],
    bounds: &[ColumnBound],
    row_limit: Option<usize>,
) -> Result<Vec<RecordBatch>, ParquetError> {
    let buf = mmap_file(path)?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(buf)?;

    // Row group pruning via min/max statistics.
    if !bounds.is_empty() {
        let metadata = builder.metadata().clone();
        let parquet_schema = metadata.file_metadata().schema_descr();

        let selected = select_row_groups_by_bounds(&metadata, parquet_schema, bounds);
        if selected.len() < metadata.num_row_groups() {
            builder = builder.with_row_groups(selected);
        }
    }

    if !columns.is_empty() {
        let schema = builder.schema().clone();
        let mut indices = Vec::with_capacity(columns.len());
        for col_name in columns {
            if let Some((idx, _)) = schema.column_with_name(col_name) {
                indices.push(idx);
            }
        }
        let mask = parquet::arrow::ProjectionMask::leaves(builder.parquet_schema(), indices);
        builder = builder.with_projection(mask);
    }

    if let Some(limit) = row_limit {
        builder = builder.with_limit(limit);
    }

    let reader = builder.build()?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok(batches?)
}

/// Read a Parquet file via mmap with **parallel** row group decoding (rayon).
///
/// Each row group is decoded in its own rayon task sharing the same mmap'd `Bytes`.
/// Combines: mmap zero-copy + predicate pushdown + column projection + row limit.
///
/// For files with a single row group this is equivalent to `read_parquet_mmap_filtered`.
/// The real speedup comes with multi-row-group files (e.g. large EventStore ntuples).
pub fn read_parquet_mmap_parallel(
    path: &Path,
    columns: &[&str],
    bounds: &[ColumnBound],
    row_limit: Option<usize>,
) -> Result<Vec<RecordBatch>, ParquetError> {
    use rayon::prelude::*;

    let buf = mmap_file(path)?;

    // Read metadata once to determine row groups + schema.
    let probe = ParquetRecordBatchReaderBuilder::try_new(buf.clone())?;
    let metadata = probe.metadata().clone();
    let n_rg = metadata.num_row_groups();

    // Single row group → fast path, no rayon overhead.
    if n_rg <= 1 {
        return read_parquet_mmap_filtered(path, columns, bounds, row_limit);
    }

    // Determine which row groups survive predicate pushdown.
    let selected: Vec<usize> = if bounds.is_empty() {
        (0..n_rg).collect()
    } else {
        let parquet_schema = metadata.file_metadata().schema_descr();
        select_row_groups_by_bounds(&metadata, parquet_schema, bounds)
    };

    if selected.is_empty() {
        return Ok(vec![]);
    }

    // Pre-compute column projection indices (shared across tasks).
    let col_indices: Vec<usize> = if columns.is_empty() {
        vec![]
    } else {
        let schema = probe.schema().clone();
        columns
            .iter()
            .filter_map(|name| schema.column_with_name(name).map(|(idx, _)| idx))
            .collect()
    };

    // Parallel decode: each task builds its own reader for one row group.
    let results: Vec<Result<Vec<RecordBatch>, ParquetError>> = selected
        .par_iter()
        .map(|&rg_idx| {
            let rg_buf = buf.clone(); // Bytes clone is O(1) — Arc bump.
            let mut rg_builder = ParquetRecordBatchReaderBuilder::try_new(rg_buf)?;

            rg_builder = rg_builder.with_row_groups(vec![rg_idx]);

            if !col_indices.is_empty() {
                let mask = parquet::arrow::ProjectionMask::leaves(
                    rg_builder.parquet_schema(),
                    col_indices.clone(),
                );
                rg_builder = rg_builder.with_projection(mask);
            }

            let reader = rg_builder.build()?;
            let batches: std::result::Result<Vec<_>, _> = reader.collect();
            Ok(batches?)
        })
        .collect();

    // Flatten and apply row limit.
    let mut all_batches = Vec::new();
    let mut rows_remaining = row_limit.unwrap_or(usize::MAX);

    for result in results {
        let batches = result?;
        for batch in batches {
            if rows_remaining == 0 {
                break;
            }
            if batch.num_rows() <= rows_remaining {
                rows_remaining -= batch.num_rows();
                all_batches.push(batch);
            } else {
                // Slice the last batch to exact row limit.
                all_batches.push(batch.slice(0, rows_remaining));
                rows_remaining = 0;
            }
        }
    }

    Ok(all_batches)
}

/// Select row groups whose statistics overlap all given column bounds.
///
/// Returns indices of row groups that should be read. If a row group lacks
/// statistics for a bounded column, it is conservatively included.
fn select_row_groups_by_bounds(
    metadata: &std::sync::Arc<parquet::file::metadata::ParquetMetaData>,
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    bounds: &[ColumnBound],
) -> Vec<usize> {
    use parquet::file::statistics::Statistics;

    let n_rg = metadata.num_row_groups();
    let mut selected = Vec::with_capacity(n_rg);

    // Map column names to Parquet column indices.
    let bound_col_indices: Vec<Option<usize>> = bounds
        .iter()
        .map(|b| {
            (0..parquet_schema.num_columns()).find(|&i| parquet_schema.column(i).name() == b.column)
        })
        .collect();

    'rg: for rg_idx in 0..n_rg {
        let rg_meta = metadata.row_group(rg_idx);

        for (bound, col_idx) in bounds.iter().zip(&bound_col_indices) {
            let Some(ci) = col_idx else {
                // Column not found in schema — skip this predicate.
                continue;
            };
            let col_meta = rg_meta.column(*ci);
            let Some(stats) = col_meta.statistics() else {
                // No statistics — conservatively include this row group.
                continue;
            };

            // Extract min/max as f64 via min_opt()/max_opt().
            let (stat_min, stat_max) = match stats {
                Statistics::Double(s) => match (s.min_opt(), s.max_opt()) {
                    (Some(&mn), Some(&mx)) => (mn, mx),
                    _ => continue,
                },
                Statistics::Float(s) => match (s.min_opt(), s.max_opt()) {
                    (Some(&mn), Some(&mx)) => (mn as f64, mx as f64),
                    _ => continue,
                },
                Statistics::Int64(s) => match (s.min_opt(), s.max_opt()) {
                    (Some(&mn), Some(&mx)) => (mn as f64, mx as f64),
                    _ => continue,
                },
                Statistics::Int32(s) => match (s.min_opt(), s.max_opt()) {
                    (Some(&mn), Some(&mx)) => (mn as f64, mx as f64),
                    _ => continue,
                },
                _ => continue, // Unsupported type — conservatively include.
            };

            // Overlap check: [stat_min, stat_max] ∩ [bound.lo, bound.hi] = ∅ ?
            if stat_max < bound.lo || stat_min > bound.hi {
                // No overlap — skip this row group.
                continue 'rg;
            }
        }

        selected.push(rg_idx);
    }

    selected
}

/// Event data in Structure-of-Arrays layout, ready for direct GPU upload.
///
/// Layout: `soa[obs_idx * n_events + event_idx]` — contiguous f64 buffer
/// matching the GPU kernel's expected SoA format.
#[derive(Debug, Clone)]
pub struct ParquetEventData {
    /// Flat SoA buffer `[n_obs × n_events]`.
    pub soa: Vec<f64>,
    /// Number of events (rows after filtering).
    pub n_events: usize,
    /// Number of observables (columns requested).
    pub n_obs: usize,
    /// Column names in the order they appear in `soa`.
    pub column_names: Vec<String>,
    /// Optional per-event weight column (length `n_events`).
    pub weights: Option<Vec<f64>>,
}

/// Read a Parquet file into GPU-ready SoA f64 layout using all optimizations.
///
/// Combines: mmap zero-copy + row group predicate pushdown + parallel decode +
/// Arrow → SoA flattening. The returned [`ParquetEventData`] can be uploaded
/// to CUDA (`clone_htod`) or Metal (`new_buffer_with_data`) with zero
/// intermediate copies.
///
/// `observable_columns` specifies which columns become observables in the SoA.
/// `weight_column` (if Some) extracts per-event weights separately.
pub fn read_parquet_events_soa(
    path: &Path,
    observable_columns: &[&str],
    bounds: &[ColumnBound],
    weight_column: Option<&str>,
    row_limit: Option<usize>,
) -> Result<ParquetEventData, ParquetError> {
    use arrow::array::Float64Array;

    // Build the full column list for projection.
    let mut all_cols: Vec<&str> = observable_columns.to_vec();
    if let Some(wc) = weight_column
        && !all_cols.contains(&wc)
    {
        all_cols.push(wc);
    }

    // Read with all optimizations.
    let batches = read_parquet_mmap_parallel(path, &all_cols, bounds, row_limit)?;

    if batches.is_empty() {
        return Ok(ParquetEventData {
            soa: vec![],
            n_events: 0,
            n_obs: observable_columns.len(),
            column_names: observable_columns.iter().map(|s| s.to_string()).collect(),
            weights: weight_column.map(|_| vec![]),
        });
    }

    // Count total events across all batches.
    let n_events: usize = batches.iter().map(|b| b.num_rows()).sum();
    let n_obs = observable_columns.len();

    // Pre-allocate SoA buffer: [n_obs × n_events].
    let mut soa = vec![0.0f64; n_obs * n_events];
    let mut weights: Option<Vec<f64>> = weight_column.map(|_| Vec::with_capacity(n_events));

    let mut event_offset = 0usize;

    for batch in &batches {
        let batch_len = batch.num_rows();

        // Extract observable columns into SoA layout.
        for (obs_idx, &col_name) in observable_columns.iter().enumerate() {
            let col_array = batch.column_by_name(col_name).ok_or_else(|| {
                ParquetError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("column '{}' not found in Parquet batch", col_name),
                ))
            })?;

            let f64_array = col_array.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                ParquetError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("column '{}' is not Float64", col_name),
                ))
            })?;

            // Direct memcpy from Arrow buffer into SoA slice.
            let src = f64_array.values();
            let dst_start = obs_idx * n_events + event_offset;
            soa[dst_start..dst_start + batch_len].copy_from_slice(src);
        }

        // Extract weight column if requested.
        if let (Some(wc), Some(wvec)) = (weight_column, &mut weights) {
            let col_array = batch.column_by_name(wc).ok_or_else(|| {
                ParquetError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("weight column '{}' not found", wc),
                ))
            })?;
            let f64_array = col_array.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                ParquetError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("weight column '{}' is not Float64", wc),
                ))
            })?;
            wvec.extend_from_slice(f64_array.values());
        }

        event_offset += batch_len;
    }

    Ok(ParquetEventData {
        soa,
        n_events,
        n_obs,
        column_names: observable_columns.iter().map(|s| s.to_string()).collect(),
        weights,
    })
}

/// Read a Parquet file and convert to a pyhf Workspace.
///
/// The Parquet file must contain the histogram table schema:
/// `channel (Utf8)`, `sample (Utf8)`, `yields (List<Float64>)`,
/// optionally `stat_error (List<Float64>)`.
pub fn from_parquet(path: &Path, config: &ArrowIngestConfig) -> Result<Workspace, ParquetError> {
    let batches = read_parquet_mmap(path)?;
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

/// Build default writer properties with NextStat metadata in the Parquet footer.
fn default_writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(default_compression())
        .set_key_value_metadata(Some(vec![
            KeyValue::new("nextstat.version".to_string(), ns_core::VERSION.to_string()),
            KeyValue::new("nextstat.created".to_string(), chrono_now_iso()),
        ]))
        .build()
}

/// ISO 8601 timestamp (best-effort, no chrono dep — uses compile-time fallback).
fn chrono_now_iso() -> String {
    // We avoid pulling in chrono just for this. Use a fixed marker that
    // downstream tooling can parse or override.
    std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .map(|epoch| format!("{epoch}"))
        .unwrap_or_else(|| {
            let d = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            format!("{}", d.as_secs())
        })
}

/// Write Arrow RecordBatches to a Parquet file.
///
/// Uses Zstd if the `arrow-io-zstd` feature is enabled, otherwise Snappy.
/// Embeds `nextstat.version` and `nextstat.created` in the Parquet footer metadata.
pub fn write_parquet(path: &Path, batches: &[RecordBatch]) -> Result<(), ParquetError> {
    if batches.is_empty() {
        return Ok(());
    }

    let schema = batches[0].schema();
    let props = default_writer_props();

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;

    for batch in batches {
        writer.write(batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Write a pyhf Workspace as two Parquet files: yields table + modifiers table.
///
/// - `yields_path`: path for the histogram yields table.
/// - `modifiers_path`: path for the modifier declarations table.
///
/// Both files embed `nextstat.version` footer metadata.
pub fn workspace_to_parquet(
    workspace: &Workspace,
    yields_path: &Path,
    modifiers_path: &Path,
) -> Result<(), ParquetError> {
    let yields_batch = super::export::yields_from_workspace(workspace)?;
    write_parquet(yields_path, &[yields_batch])?;

    let modifiers_batch = super::export::modifiers_to_record_batch(workspace)?;
    write_parquet(modifiers_path, &[modifiers_batch])?;

    Ok(())
}

/// Read yields + modifiers Parquet files and convert to a pyhf Workspace.
///
/// - `yields_path`: Parquet file with the histogram yields table.
/// - `modifiers_path`: Parquet file with the modifier declarations table.
pub fn from_parquet_with_modifiers(
    yields_path: &Path,
    modifiers_path: &Path,
    config: &ArrowIngestConfig,
) -> Result<Workspace, ParquetError> {
    let yield_batches = read_parquet_mmap(yields_path)?;
    let modifier_batches = read_parquet_mmap(modifiers_path)?;
    let ws = super::ingest::from_record_batches_with_modifiers(
        &yield_batches,
        &modifier_batches,
        config,
    )?;
    Ok(ws)
}

/// Write Arrow RecordBatches to Parquet bytes in memory.
pub fn write_parquet_bytes(batches: &[RecordBatch]) -> Result<Vec<u8>, ParquetError> {
    if batches.is_empty() {
        return Ok(vec![]);
    }

    let schema = batches[0].schema();
    let props = default_writer_props();

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
