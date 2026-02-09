#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
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
fn bench_ratio_and_compress_throughput_vs_c() {
    // Env vars:
    // - NS_ZSTD_RATIO_MB=64
    // - NS_ZSTD_RATIO_ITERS=10
    let mb = env_usize("NS_ZSTD_RATIO_MB", 32);
    let iters = env_usize("NS_ZSTD_RATIO_ITERS", 10);

    let input = gen_rootish_bytes(mb * 1024 * 1024);
    let bound = zstd_safe::compress_bound(input.len());

    // --- Ratio snapshot (single run) ---
    let ns_fast = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Fastest,
    );
    let ns_def = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Default,
    );

    let mut cctx = zstd_safe::CCtx::default();
    let mut c_out = Vec::with_capacity(bound.max(1));
    cctx.compress(&mut c_out, &input, 3).expect("libzstd compress");

    eprintln!(
        "input={} MiB\n  ns_fastest={} bytes ratio={:.3}\n  ns_default={} bytes ratio={:.3}\n  c_zstd_l3={} bytes ratio={:.3}",
        mb,
        ns_fast.len(),
        ns_fast.len() as f64 / input.len() as f64,
        ns_def.len(),
        ns_def.len() as f64 / input.len() as f64,
        c_out.len(),
        c_out.len() as f64 / input.len() as f64,
    );

    // --- Compression throughput (ns default vs C level 3) ---
    let mut ns_best: f64 = 0.0;
    let mut ns_total = 0.0;
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = ns_zstd::encoding::compress_to_vec(
            input.as_slice(),
            ns_zstd::encoding::CompressionLevel::Default,
        );
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(out.len());
        let tput = mbps(input.len(), dt);
        ns_total += tput;
        ns_best = ns_best.max(tput);
    }

    let mut c_best: f64 = 0.0;
    let mut c_total = 0.0;
    let mut c_out = Vec::with_capacity(bound.max(1));
    for _ in 0..iters {
        c_out.clear();
        let t0 = Instant::now();
        cctx.compress(&mut c_out, &input, 3).expect("libzstd compress");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(c_out.len());
        let tput = mbps(input.len(), dt);
        c_total += tput;
        c_best = c_best.max(tput);
    }

    eprintln!(
        "compress throughput:\n  ns_default: best={:.1} MiB/s avg={:.1} MiB/s (iters={})\n  c_zstd_l3:  best={:.1} MiB/s avg={:.1} MiB/s (iters={})",
        ns_best,
        ns_total / iters as f64,
        iters,
        c_best,
        c_total / iters as f64,
        iters,
    );
}
