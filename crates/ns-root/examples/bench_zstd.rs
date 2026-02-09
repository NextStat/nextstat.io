//! Quick benchmark: ns_zstd (our fork) decompression throughput.
//! Run: cargo run -p ns-root --release --example bench_zstd

fn main() {
    // Generate compressible payload: mixed structured data similar to ROOT TTree
    // (integer event IDs, small jet counts, floating-point with repeating patterns)
    let n_events = 200_000;
    let mut raw = Vec::with_capacity(n_events * 20);
    for i in 0u32..n_events as u32 {
        // event ID (monotonically increasing — very compressible via delta)
        raw.extend_from_slice(&i.to_le_bytes());
        // njet: small integer 0-8 (highly compressible)
        raw.push((i % 9) as u8);
        raw.push(0);
        raw.push(0);
        raw.push(0);
        // pt: float with limited dynamic range
        let pt = 25.0_f32 + (i as f32 % 200.0);
        raw.extend_from_slice(&pt.to_le_bytes());
        // eta: float in [-2.5, 2.5]
        let eta = -2.5_f32 + (i as f32 % 50.0) * 0.1;
        raw.extend_from_slice(&eta.to_le_bytes());
        // weight: mostly 1.0 with occasional variations
        let w = if i % 7 == 0 { 1.05_f32 } else { 1.0_f32 };
        raw.extend_from_slice(&w.to_le_bytes());
    }
    let uncompressed_size = raw.len();

    // Compress with ns_zstd encoder (Fastest vs Default)
    let compressed_fast = ns_zstd::encoding::compress_to_vec(
        raw.as_slice(),
        ns_zstd::encoding::CompressionLevel::Fastest,
    );
    let compressed_def = ns_zstd::encoding::compress_to_vec(
        raw.as_slice(),
        ns_zstd::encoding::CompressionLevel::Default,
    );

    let ratio_fast = uncompressed_size as f64 / compressed_fast.len() as f64;
    let ratio_def = uncompressed_size as f64 / compressed_def.len() as f64;

    println!(
        "Payload: {} bytes uncompressed\n  Fastest: {} bytes (ratio {:.2}x)\n  Default: {} bytes (ratio {:.2}x)",
        uncompressed_size,
        compressed_fast.len(),
        ratio_fast,
        compressed_def.len(),
        ratio_def
    );

    // Verify correctness (both encodings)
    {
        for (name, compressed) in [("Fastest", &compressed_fast), ("Default", &compressed_def)] {
            let mut dec = ns_zstd::decoding::FrameDecoder::new();
            let mut out = Vec::with_capacity(uncompressed_size);
            dec.decode_all_to_vec(compressed, &mut out).unwrap();
            assert_eq!(out.len(), uncompressed_size, "{} decode length mismatch", name);
            assert_eq!(out, raw, "{} decode mismatch", name);
        }
    }

    let iterations = 500;
    let rounds = 5;

    // --- Benchmark 0: compression throughput ---
    // Keep iteration counts lower: compression is slower than decode and may allocate.
    println!("\n--- compress_to_vec throughput ---");
    for (name, level, iters) in [
        ("Fastest", ns_zstd::encoding::CompressionLevel::Fastest, 200usize),
        ("Default", ns_zstd::encoding::CompressionLevel::Default, 60usize),
    ] {
        let mut tps = Vec::new();
        for round in 0..rounds {
            for _ in 0..3 {
                let out = ns_zstd::encoding::compress_to_vec(raw.as_slice(), level);
                std::hint::black_box(&out);
            }
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let out = ns_zstd::encoding::compress_to_vec(raw.as_slice(), level);
                std::hint::black_box(&out);
            }
            let elapsed = start.elapsed();
            let total_bytes = uncompressed_size as f64 * iters as f64;
            let tp = total_bytes / elapsed.as_secs_f64() / 1e6;
            tps.push(tp);
            let per_iter_us = elapsed.as_micros() as f64 / iters as f64;
            println!(
                "  {} Round {}: {:.0} MB/s  ({:.0} µs/iter)",
                name,
                round + 1,
                tp,
                per_iter_us
            );
        }
        tps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {} median: {:.0} MB/s  best: {:.0} MB/s",
            name,
            tps[rounds / 2],
            tps[rounds - 1]
        );
    }

    // --- Benchmark 1: decode_all_to_vec (bulk, no streaming overhead) ---
    println!("\n--- decode_all_to_vec (bulk) ---");
    let mut throughputs = Vec::new();
    let mut decoder = ns_zstd::decoding::FrameDecoder::new();
    for round in 0..rounds {
        for _ in 0..3 {
            let mut out = Vec::with_capacity(uncompressed_size);
            decoder.decode_all_to_vec(&compressed_fast, &mut out).unwrap();
        }
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut out = Vec::with_capacity(uncompressed_size);
            decoder.decode_all_to_vec(&compressed_fast, &mut out).unwrap();
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed();
        let total_bytes = uncompressed_size as f64 * iterations as f64;
        let tp = total_bytes / elapsed.as_secs_f64() / 1e6;
        throughputs.push(tp);
        let per_iter_us = elapsed.as_micros() as f64 / iterations as f64;
        println!("  Round {}: {:.0} MB/s  ({:.0} µs/iter)", round + 1, tp, per_iter_us);
    }
    throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_bulk = throughputs[rounds / 2];
    let best_bulk = throughputs[rounds - 1];
    println!("Median: {:.0} MB/s  Best: {:.0} MB/s", median_bulk, best_bulk);

    // --- Benchmark 2: StreamingDecoder (read_to_end) ---
    println!("\n--- StreamingDecoder (read_to_end) ---");
    let mut throughputs = Vec::new();
    for round in 0..rounds {
        for _ in 0..3 {
            let mut dec =
                ns_zstd::decoding::StreamingDecoder::new(compressed_fast.as_slice()).unwrap();
            let mut out = Vec::with_capacity(uncompressed_size);
            std::io::Read::read_to_end(&mut dec, &mut out).unwrap();
        }
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut dec =
                ns_zstd::decoding::StreamingDecoder::new(compressed_fast.as_slice()).unwrap();
            let mut out = Vec::with_capacity(uncompressed_size);
            std::io::Read::read_to_end(&mut dec, &mut out).unwrap();
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed();
        let total_bytes = uncompressed_size as f64 * iterations as f64;
        let tp = total_bytes / elapsed.as_secs_f64() / 1e6;
        throughputs.push(tp);
        let per_iter_us = elapsed.as_micros() as f64 / iterations as f64;
        println!("  Round {}: {:.0} MB/s  ({:.0} µs/iter)", round + 1, tp, per_iter_us);
    }
    throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_stream = throughputs[rounds / 2];
    let best_stream = throughputs[rounds - 1];
    println!("Median: {:.0} MB/s  Best: {:.0} MB/s", median_stream, best_stream);

    println!("\n=== Summary ===");
    println!("Bulk:      {:.0} MB/s median, {:.0} MB/s best", median_bulk, best_bulk);
    println!("Streaming: {:.0} MB/s median, {:.0} MB/s best", median_stream, best_stream);
    println!("Reference: C libzstd ~1500-2000 MB/s");
}
