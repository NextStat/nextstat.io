#![cfg(all(feature = "arrow-io-zstd", not(target_arch = "wasm32")))]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use parquet::basic::Compression;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn sample_path() -> PathBuf {
    std::env::var_os("NS_PARQUET_ZSTD_SAMPLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/tmp/ns_parquet_samples/byte_stream_split.zstd.parquet".into())
}

fn input_path() -> PathBuf {
    std::env::var_os("NS_PARQUET_INPUT_SAMPLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/tmp/ns_parquet_samples/alltypes_tiny_pages_plain.parquet".into())
}

fn ensure_sample_exists(path: &Path) {
    if path.exists() {
        return;
    }
    panic!(
        "missing parquet sample at {:?}\n\
Download a zstd-compressed parquet file, e.g.:\n\
  mkdir -p /tmp/ns_parquet_samples\n\
  curl -L -o /tmp/ns_parquet_samples/byte_stream_split.zstd.parquet \\\n\
    https://cdn.jsdelivr.net/gh/apache/parquet-testing@master/data/byte_stream_split.zstd.parquet\n\
Or set NS_PARQUET_ZSTD_SAMPLE to a local file path.",
        path
    );
}

fn assert_has_zstd_compressed_pages(path: &Path) {
    let file = std::fs::File::open(path).expect("open parquet sample");
    let reader = SerializedFileReader::new(file).expect("parquet metadata read");
    let md = reader.metadata();
    let mut saw_zstd = false;
    for rg in 0..md.num_row_groups() {
        let rg = md.row_group(rg);
        for col in 0..rg.num_columns() {
            if matches!(rg.column(col).compression(), Compression::ZSTD(_)) {
                saw_zstd = true;
                break;
            }
        }
        if saw_zstd {
            break;
        }
    }
    assert!(saw_zstd, "expected at least one ZSTD-compressed column chunk in {:?}", path);
}

fn assert_bytes_have_zstd_compressed_pages(parquet_bytes: &[u8]) {
    let buf = bytes::Bytes::copy_from_slice(parquet_bytes);
    let reader = SerializedFileReader::new(buf).expect("parquet metadata read");
    let md = reader.metadata();
    let mut saw_zstd = false;
    for rg in 0..md.num_row_groups() {
        let rg = md.row_group(rg);
        for col in 0..rg.num_columns() {
            if matches!(rg.column(col).compression(), Compression::ZSTD(_)) {
                saw_zstd = true;
                break;
            }
        }
        if saw_zstd {
            break;
        }
    }
    assert!(saw_zstd, "expected at least one ZSTD-compressed column chunk in output parquet");
}

#[test]
#[ignore]
fn parquet_external_zstd_smoke_read_and_roundtrip() {
    let path = sample_path();
    ensure_sample_exists(&path);
    assert_has_zstd_compressed_pages(&path);

    // Exercise file-path and in-memory APIs.
    let batches = ns_translate::arrow::parquet::read_parquet_batches(&path)
        .expect("read parquet sample into record batches");
    assert!(!batches.is_empty(), "expected non-empty record batches");

    let bytes = ns_translate::arrow::parquet::write_parquet_bytes(&batches)
        .expect("write record batches to parquet bytes");
    let batches2 = ns_translate::arrow::parquet::read_parquet_bytes(&bytes)
        .expect("read parquet bytes into record batches");

    let rows1: usize = batches.iter().map(|b| b.num_rows()).sum();
    let rows2: usize = batches2.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows1, rows2, "row count changed after roundtrip");
}

#[test]
#[ignore]
fn parquet_external_zstd_bench_in_memory_decode() {
    let path = sample_path();
    ensure_sample_exists(&path);
    assert_has_zstd_compressed_pages(&path);

    let iters: usize =
        std::env::var("NS_PARQUET_ZSTD_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);

    let data = fs::read(&path).expect("read parquet sample bytes");
    let bytes = data.as_slice();

    // Warmup
    for _ in 0..5 {
        let batches =
            ns_translate::arrow::parquet::read_parquet_bytes(bytes).expect("read parquet bytes");
        std::hint::black_box(batches.len());
    }

    let t0 = Instant::now();
    let mut rows = 0usize;
    for _ in 0..iters {
        let batches =
            ns_translate::arrow::parquet::read_parquet_bytes(bytes).expect("read parquet bytes");
        rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
        std::hint::black_box(&batches);
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "parquet_external_zstd: file_bytes={} iters={} total_rows={} elapsed={:.3}s",
        bytes.len(),
        iters,
        rows,
        dt
    );
}

#[test]
#[ignore]
fn parquet_external_input_to_zstd_roundtrip_and_bench() {
    let path = input_path();
    ensure_sample_exists(&path);

    let input_bytes = fs::read(&path).expect("read parquet input bytes");
    let input = input_bytes.as_slice();

    // Read external parquet (any compression), then write it out with Zstd enabled via feature flag.
    let batches =
        ns_translate::arrow::parquet::read_parquet_bytes(input).expect("read parquet input");
    assert!(!batches.is_empty(), "expected non-empty record batches");

    let repeat: usize =
        std::env::var("NS_PARQUET_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut repeated = Vec::with_capacity(batches.len() * repeat);
    for _ in 0..repeat {
        // RecordBatch is Arc-backed; cloning is cheap and lets us build a "large enough" file
        // for stable throughput measurements without downloading huge fixtures.
        repeated.extend(batches.iter().cloned());
    }

    let zstd_bytes =
        ns_translate::arrow::parquet::write_parquet_bytes(&repeated).expect("write parquet zstd");
    assert_bytes_have_zstd_compressed_pages(&zstd_bytes);

    let iters: usize =
        std::env::var("NS_PARQUET_ZSTD_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);

    // Warmup
    for _ in 0..5 {
        let out = ns_translate::arrow::parquet::read_parquet_bytes(&zstd_bytes)
            .expect("read zstd parquet bytes");
        std::hint::black_box(out.len());
    }

    let t0 = Instant::now();
    let mut rows = 0usize;
    for _ in 0..iters {
        let out = ns_translate::arrow::parquet::read_parquet_bytes(&zstd_bytes)
            .expect("read zstd parquet bytes");
        rows += out.iter().map(|b| b.num_rows()).sum::<usize>();
        std::hint::black_box(&out);
    }
    let dt = t0.elapsed().as_secs_f64();

    let mb = 1024.0 * 1024.0;
    let mbps_in = (zstd_bytes.len() as f64) * (iters as f64) / mb / dt;
    eprintln!(
        "parquet_external_input_to_zstd: input_file_bytes={} zstd_file_bytes={} iters={} total_rows={} elapsed={:.3}s ({:.1} MiB/s over zstd bytes)",
        input.len(),
        zstd_bytes.len(),
        iters,
        rows,
        dt,
        mbps_in
    );
}
