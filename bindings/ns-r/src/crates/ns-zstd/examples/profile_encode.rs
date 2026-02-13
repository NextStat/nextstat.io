//! Profile encoder hot paths.
//!
//! Build:
//!   cargo build -p ns-zstd --example profile_encode --release
//! Run (defaults):
//!   NS_ZSTD_PROFILE_MB=64 NS_ZSTD_PROFILE_ITERS=200 target/release/examples/profile_encode
//!
//! Then sample the PID:
//!   sample <pid> 5 -file /tmp/ns_zstd.encode.sample.txt

use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
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

fn main() {
    let mb = env_usize("NS_ZSTD_PROFILE_MB", 64);
    let iters = env_usize("NS_ZSTD_PROFILE_ITERS", 200);
    let level_s = std::env::var("NS_ZSTD_PROFILE_LEVEL").unwrap_or_else(|_| "default".to_string());
    let level = match level_s.as_str() {
        "fastest" => ns_zstd::encoding::CompressionLevel::Fastest,
        "uncompressed" => ns_zstd::encoding::CompressionLevel::Uncompressed,
        _ => ns_zstd::encoding::CompressionLevel::Default,
    };

    let input = gen_rootish_bytes(mb * 1024 * 1024);
    eprintln!("input={} MiB level={} iters={}", mb, level_s, iters);

    // Warmup
    for _ in 0..5 {
        let out = ns_zstd::encoding::compress_to_vec(input.as_slice(), level);
        std::hint::black_box(out.len());
    }

    let start = Instant::now();
    let mut bytes = 0usize;
    for _ in 0..iters {
        let out = ns_zstd::encoding::compress_to_vec(input.as_slice(), level);
        bytes += out.len();
        std::hint::black_box(&out);
    }
    let dt = start.elapsed().as_secs_f64();
    eprintln!("elapsed={:.3}s total_out_bytes={} (~{:.1} it/s)", dt, bytes, iters as f64 / dt);
}
