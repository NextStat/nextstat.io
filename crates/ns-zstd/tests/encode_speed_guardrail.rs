#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(default)
}

fn gen_rootish_bytes(len: usize) -> Vec<u8> {
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
fn bench_encode_speed_guardrail_default_rootish() {
    // Example:
    // NS_ZSTD_GUARD_MB=64 NS_ZSTD_GUARD_ITERS=15 NS_ZSTD_GUARD_MIN_MIBS=200 \
    //   cargo test -p ns-zstd --release --test encode_speed_guardrail -- --ignored --nocapture
    let mb = env_usize("NS_ZSTD_GUARD_MB", 64);
    let iters = env_usize("NS_ZSTD_GUARD_ITERS", 15);
    let min_mibs = env_f64("NS_ZSTD_GUARD_MIN_MIBS", 100.0);

    let input = gen_rootish_bytes(mb * 1024 * 1024);
    let bound = zstd_safe::compress_bound(input.len());
    let mut encoder: ns_zstd::encoding::FrameCompressor<
        std::io::Cursor<&[u8]>,
        Vec<u8>,
        ns_zstd::encoding::MatchGeneratorDriver,
    > = ns_zstd::encoding::FrameCompressor::new(ns_zstd::encoding::CompressionLevel::Default);
    encoder.set_drain(Vec::with_capacity(bound.max(1)));

    let mut best = 0.0f64;
    let mut total = 0.0f64;
    for _ in 0..iters {
        encoder.set_source(std::io::Cursor::new(&input));
        encoder.drain_mut().unwrap().clear();
        let t0 = Instant::now();
        encoder.compress();
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(encoder.drain().unwrap().len());
        let tput = mbps(input.len(), dt);
        total += tput;
        best = best.max(tput);
    }

    let avg = total / iters as f64;
    eprintln!(
        "encode_guardrail default rootish input={} MiB: best={:.1} MiB/s avg={:.1} MiB/s (iters={}) threshold={:.1}",
        mb, best, avg, iters, min_mibs
    );

    assert!(
        avg >= min_mibs,
        "encode avg throughput {:.1} MiB/s is below threshold {:.1} MiB/s",
        avg,
        min_mibs
    );
}
