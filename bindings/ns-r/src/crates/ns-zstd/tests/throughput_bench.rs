#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::io::Read as _;
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
}

fn gen_deterministic_bytes(len: usize) -> Vec<u8> {
    // Simple xorshift64* with a compression-friendly structure:
    // periodic low-entropy runs + deterministic "noise" blocks.
    let mut out = vec![0u8; len];
    let mut x: u64 = 0x1234_5678_9abc_def0;
    for (i, b) in out.iter_mut().enumerate() {
        // Every 4 KiB insert a short repetitive run.
        if (i & 4095) < 128 {
            *b = (i as u8).wrapping_mul(7);
            continue;
        }
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
        *b = (x >> 56) as u8;
    }
    out
}

fn gen_rootish_bytes(len: usize) -> Vec<u8> {
    // Similar to ns-root/examples/bench_zstd.rs, but sized by bytes instead of events.
    // Mix of compressible integer patterns + moderately varying floats.
    let mut out = Vec::with_capacity(len);
    let mut i: u32 = 0;
    while out.len() + 20 <= len {
        out.extend_from_slice(&i.to_le_bytes());
        out.push((i % 9) as u8);
        out.extend_from_slice(&[0, 0, 0]);

        let pt = 25.0_f32 + (i as f32 % 200.0);
        out.extend_from_slice(&pt.to_le_bytes());
        let eta = -2.5_f32 + (i as f32 % 50.0) * 0.1;
        out.extend_from_slice(&eta.to_le_bytes());
        let w = if i.is_multiple_of(7) { 1.05_f32 } else { 1.0_f32 };
        out.extend_from_slice(&w.to_le_bytes());

        i = i.wrapping_add(1);
    }
    out.resize(len, 0);
    out
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0) / secs
}

#[test]
#[ignore]
fn bench_decode_throughput_vs_c() {
    // Tweak via env vars:
    // - NS_ZSTD_BENCH_MB=64
    // - NS_ZSTD_BENCH_ITERS=20
    // - NS_ZSTD_BENCH_LEVEL=default|fastest|uncompressed
    let mb = env_usize("NS_ZSTD_BENCH_MB", 32);
    let iters = env_usize("NS_ZSTD_BENCH_ITERS", 15);
    let level = std::env::var("NS_ZSTD_BENCH_LEVEL").unwrap_or_else(|_| "default".to_string());

    let level = match level.as_str() {
        "fastest" => ns_zstd::encoding::CompressionLevel::Fastest,
        "uncompressed" => ns_zstd::encoding::CompressionLevel::Uncompressed,
        _ => ns_zstd::encoding::CompressionLevel::Default,
    };

    // Data generator selector:
    // - NS_ZSTD_BENCH_DATA=deterministic|rootish
    let data = std::env::var("NS_ZSTD_BENCH_DATA").unwrap_or_else(|_| "deterministic".to_string());
    let input = match data.as_str() {
        "rootish" => gen_rootish_bytes(mb * 1024 * 1024),
        _ => gen_deterministic_bytes(mb * 1024 * 1024),
    };
    let compressed = ns_zstd::encoding::compress_to_vec(input.as_slice(), level);
    eprintln!(
        "input={} MiB data={} compressed={} bytes ratio={:.3}",
        mb,
        data,
        compressed.len(),
        (compressed.len() as f64) / (input.len() as f64)
    );

    // Correctness warmups (also primes caches / JITs).
    {
        let mut dec = ns_zstd::decoding::StreamingDecoder::new(std::io::Cursor::new(&compressed))
            .expect("ns_zstd decoder init");
        let mut out = Vec::with_capacity(input.len());
        dec.read_to_end(&mut out).expect("ns_zstd decode");
        assert_eq!(out, input);
    }
    {
        let mut out = Vec::with_capacity(input.len().max(1));
        let mut dctx = zstd_safe::DCtx::default();
        dctx.decompress(&mut out, &compressed).expect("zstd_safe decode");
        assert_eq!(out, input);
    }

    // ns_zstd throughput
    let mut out = Vec::with_capacity(input.len());
    let mut ns_best = 0.0f64;
    let mut ns_total = 0.0f64;
    let mut frame = ns_zstd::decoding::FrameDecoder::new();
    for _ in 0..iters {
        out.clear();
        let t0 = Instant::now();
        frame.decode_all_to_vec(&compressed, &mut out).expect("ns_zstd decode");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(out.len());
        let tput = mbps(out.len(), dt);
        ns_total += tput;
        if tput > ns_best {
            ns_best = tput;
        }
    }

    // libzstd throughput (via zstd-safe)
    let mut out = Vec::with_capacity(input.len().max(1));
    let mut dctx = zstd_safe::DCtx::default();
    let mut c_best = 0.0f64;
    let mut c_total = 0.0f64;
    for _ in 0..iters {
        out.clear();
        let t0 = Instant::now();
        dctx.decompress(&mut out, &compressed).expect("zstd_safe decode");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(out.len());
        let tput = mbps(out.len(), dt);
        c_total += tput;
        if tput > c_best {
            c_best = tput;
        }
    }

    eprintln!(
        "ns_zstd: best={:.1} MiB/s avg={:.1} MiB/s (iters={})",
        ns_best,
        ns_total / (iters as f64),
        iters
    );
    eprintln!(
        "libzstd: best={:.1} MiB/s avg={:.1} MiB/s (iters={})",
        c_best,
        c_total / (iters as f64),
        iters
    );
}
