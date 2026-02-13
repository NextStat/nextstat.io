#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
}

fn gen_rootish_bytes(len: usize) -> Vec<u8> {
    // Same shape as other benches: repeated integer structure + gently varying floats.
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
fn bench_decode_throughput_on_libzstd_level3_frames() {
    // Env vars:
    // - NS_ZSTD_PROFILE_MB=256
    // - NS_ZSTD_PROFILE_ITERS=100
    // - NS_ZSTD_PROFILE_LEVEL=3
    let mb = env_usize("NS_ZSTD_PROFILE_MB", 128);
    let iters = env_usize("NS_ZSTD_PROFILE_ITERS", 100);
    let level: i32 =
        std::env::var("NS_ZSTD_PROFILE_LEVEL").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    let input = gen_rootish_bytes(mb * 1024 * 1024);
    let bound = zstd_safe::compress_bound(input.len());

    let mut cctx = zstd_safe::CCtx::default();
    let mut compressed = Vec::with_capacity(bound.max(1));
    cctx.compress(&mut compressed, &input, level).expect("libzstd compress");

    eprintln!(
        "input={} MiB compressed={} bytes ratio={:.3}",
        mb,
        compressed.len(),
        (compressed.len() as f64) / (input.len() as f64)
    );

    // Correctness warmup + cache prime.
    {
        let mut out = Vec::with_capacity(input.len());
        let mut dec = ns_zstd::decoding::FrameDecoder::new();
        dec.decode_all_to_vec(&compressed, &mut out).expect("ns_zstd decode");
        assert_eq!(out, input);
    }

    // ns_zstd throughput
    let mut out = Vec::with_capacity(input.len());
    let mut frame = ns_zstd::decoding::FrameDecoder::new();
    let mut best = 0.0f64;
    let mut total = 0.0f64;
    for _ in 0..iters {
        out.clear();
        let t0 = Instant::now();
        frame.decode_all_to_vec(&compressed, &mut out).expect("ns_zstd decode");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(out.len());
        let tput = mbps(out.len(), dt);
        best = best.max(tput);
        total += tput;
    }
    eprintln!(
        "ns_zstd decode: best={:.1} MiB/s avg={:.1} MiB/s (iters={})",
        best,
        total / (iters as f64),
        iters
    );

    // libzstd throughput (reference)
    let mut ref_out = Vec::with_capacity(input.len().max(1));
    let mut dctx = zstd_safe::DCtx::default();
    let mut c_best = 0.0f64;
    let mut c_total = 0.0f64;
    for _ in 0..iters {
        ref_out.clear();
        let t0 = Instant::now();
        dctx.decompress(&mut ref_out, &compressed).expect("libzstd decode");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(ref_out.len());
        let tput = mbps(ref_out.len(), dt);
        c_best = c_best.max(tput);
        c_total += tput;
    }
    eprintln!(
        "libzstd decode: best={:.1} MiB/s avg={:.1} MiB/s (iters={})",
        c_best,
        c_total / (iters as f64),
        iters
    );
}
