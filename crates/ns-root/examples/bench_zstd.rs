//! Quick benchmark: ruzstd (our fork) decompression throughput.
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
        raw.push(0); raw.push(0); raw.push(0);
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

    // Compress with ruzstd encoder
    let compressed = ruzstd::encoding::compress_to_vec(
        raw.as_slice(),
        ruzstd::encoding::CompressionLevel::Fastest,
    );
    let ratio = uncompressed_size as f64 / compressed.len() as f64;

    println!("Payload: {} bytes uncompressed, {} bytes compressed (ratio {:.2}x)",
        uncompressed_size, compressed.len(), ratio);

    // Warm up
    for _ in 0..5 {
        let mut dec = ruzstd::decoding::StreamingDecoder::new(compressed.as_slice()).unwrap();
        let mut out = Vec::with_capacity(uncompressed_size);
        std::io::Read::read_to_end(&mut dec, &mut out).unwrap();
        assert_eq!(out.len(), uncompressed_size);
    }

    // Benchmark: 3 rounds × N iterations, take median
    let iterations = 500;
    let rounds = 5;
    let mut throughputs = Vec::new();

    for round in 0..rounds {
        // Warm up
        for _ in 0..3 {
            let mut dec = ruzstd::decoding::StreamingDecoder::new(compressed.as_slice()).unwrap();
            let mut out = Vec::with_capacity(uncompressed_size);
            std::io::Read::read_to_end(&mut dec, &mut out).unwrap();
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut dec = ruzstd::decoding::StreamingDecoder::new(compressed.as_slice()).unwrap();
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
    let median = throughputs[rounds / 2];
    let best = throughputs[rounds - 1];

    println!();
    println!("Median: {:.0} MB/s", median);
    println!("Best:   {:.0} MB/s", best);
    println!();
    println!("Reference: C libzstd single-thread ~1500-2000 MB/s on similar data");
}
