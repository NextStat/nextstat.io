//! Criterion benchmarks for Parquet read paths.
//!
//! Compares: sequential file read, mmap read, mmap + projection,
//! mmap + predicate pushdown, and parallel row group decode.
//!
//! Run: `cargo bench -p ns-translate --features arrow-io --bench parquet_read`

#[cfg(feature = "arrow-io")]
use std::sync::Arc;

#[cfg(feature = "arrow-io")]
use arrow::array::{ArrayRef, Float64Array};
#[cfg(feature = "arrow-io")]
use arrow::datatypes::{DataType, Field, Schema};
#[cfg(feature = "arrow-io")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "arrow-io")]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(feature = "arrow-io")]
use ns_translate::arrow::parquet::{
    ColumnBound, read_parquet_batches, read_parquet_mmap, read_parquet_mmap_filtered,
    read_parquet_mmap_parallel, read_parquet_mmap_projected,
};

/// Temp dir wrapper with cleanup on drop.
#[cfg(feature = "arrow-io")]
struct BenchDir(std::path::PathBuf);

#[cfg(feature = "arrow-io")]
impl BenchDir {
    fn parquet_path(&self) -> std::path::PathBuf {
        self.0.join("bench.parquet")
    }
}

#[cfg(feature = "arrow-io")]
impl Drop for BenchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Generate a synthetic Parquet file with `n_rows` events and return its dir.
///
/// Schema: mass (Float64), pt (Float64), eta (Float64), weight (Float64).
#[cfg(feature = "arrow-io")]
fn make_bench_parquet(n_rows: usize, row_group_size: usize) -> BenchDir {
    let dir_path = std::env::temp_dir().join(format!("ns_bench_pq_{}_{}", n_rows, row_group_size));
    let _ = std::fs::create_dir_all(&dir_path);
    let path = dir_path.join("bench.parquet");

    let mass: Vec<f64> = (0..n_rows).map(|i| 100.0 + (i as f64 / n_rows as f64) * 80.0).collect();
    let pt: Vec<f64> = (0..n_rows).map(|i| 20.0 + (i as f64 / n_rows as f64) * 500.0).collect();
    let eta: Vec<f64> = (0..n_rows).map(|i| -2.5 + (i as f64 / n_rows as f64) * 5.0).collect();
    let weight: Vec<f64> = vec![1.0; n_rows];

    let schema = Arc::new(Schema::new(vec![
        Field::new("mass", DataType::Float64, false),
        Field::new("pt", DataType::Float64, false),
        Field::new("eta", DataType::Float64, false),
        Field::new("weight", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(mass)) as ArrayRef,
            Arc::new(Float64Array::from(pt)) as ArrayRef,
            Arc::new(Float64Array::from(eta)) as ArrayRef,
            Arc::new(Float64Array::from(weight)) as ArrayRef,
        ],
    )
    .unwrap();

    let props = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_size(row_group_size)
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk)
        .build();

    let file = std::fs::File::create(&path).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    BenchDir(dir_path)
}

#[cfg(feature = "arrow-io")]
fn bench_read_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_read_sequential");

    for &n_rows in &[1_000, 100_000] {
        let dir = make_bench_parquet(n_rows, 8192);
        let path = dir.parquet_path();

        group.bench_with_input(BenchmarkId::new("file_read", n_rows), &path, |b, path| {
            b.iter(|| read_parquet_batches(path).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("mmap_read", n_rows), &path, |b, path| {
            b.iter(|| read_parquet_mmap(path).unwrap())
        });
    }

    group.finish();
}

#[cfg(feature = "arrow-io")]
fn bench_read_projected(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_read_projected");

    for &n_rows in &[1_000, 100_000] {
        let dir = make_bench_parquet(n_rows, 8192);
        let path = dir.parquet_path();

        group.bench_with_input(BenchmarkId::new("all_cols", n_rows), &path, |b, path| {
            b.iter(|| read_parquet_mmap_projected(path, &[], None).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("2_of_4_cols", n_rows), &path, |b, path| {
            b.iter(|| read_parquet_mmap_projected(path, &["mass", "pt"], None).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("1_col", n_rows), &path, |b, path| {
            b.iter(|| read_parquet_mmap_projected(path, &["mass"], None).unwrap())
        });
    }

    group.finish();
}

#[cfg(feature = "arrow-io")]
fn bench_read_filtered(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_read_filtered");

    // 100k rows, 1000-row row groups → 100 row groups. Filter selects ~25%.
    let dir = make_bench_parquet(100_000, 1_000);
    let path = dir.parquet_path();

    let bounds_25pct = vec![ColumnBound {
        column: "mass".into(),
        lo: 100.0,
        hi: 120.0, // ~25% of [100, 180] range
    }];

    let bounds_10pct = vec![ColumnBound {
        column: "mass".into(),
        lo: 100.0,
        hi: 108.0, // ~10%
    }];

    group.bench_function("no_filter", |b| {
        b.iter(|| read_parquet_mmap_filtered(&path, &[], &[], None).unwrap())
    });

    group.bench_function("filter_25pct", |b| {
        b.iter(|| read_parquet_mmap_filtered(&path, &[], &bounds_25pct, None).unwrap())
    });

    group.bench_function("filter_10pct", |b| {
        b.iter(|| read_parquet_mmap_filtered(&path, &[], &bounds_10pct, None).unwrap())
    });

    group.finish();
}

#[cfg(feature = "arrow-io")]
fn bench_read_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_read_parallel");

    // 100k rows, 1000-row row groups → 100 row groups.
    let dir = make_bench_parquet(100_000, 1_000);
    let path = dir.parquet_path();

    let bounds = vec![ColumnBound { column: "mass".into(), lo: 100.0, hi: 120.0 }];

    group.bench_function("sequential_all", |b| {
        b.iter(|| read_parquet_mmap_filtered(&path, &[], &[], None).unwrap())
    });

    group.bench_function("parallel_all", |b| {
        b.iter(|| read_parquet_mmap_parallel(&path, &[], &[], None).unwrap())
    });

    group.bench_function("sequential_filtered", |b| {
        b.iter(|| read_parquet_mmap_filtered(&path, &[], &bounds, None).unwrap())
    });

    group.bench_function("parallel_filtered", |b| {
        b.iter(|| read_parquet_mmap_parallel(&path, &[], &bounds, None).unwrap())
    });

    group.finish();
}

#[cfg(feature = "arrow-io")]
criterion_group!(
    benches,
    bench_read_sequential,
    bench_read_projected,
    bench_read_filtered,
    bench_read_parallel,
);

#[cfg(feature = "arrow-io")]
criterion_main!(benches);

#[cfg(not(feature = "arrow-io"))]
fn main() {
    eprintln!("parquet_read benchmarks require --features arrow-io");
}
