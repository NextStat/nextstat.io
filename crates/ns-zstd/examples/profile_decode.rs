//! Simple long-running decode loop for `sample(1)` / Instruments profiling.
//!
//! Usage:
//!   NS_ZSTD_PROFILE_SECS=60 cargo run -p ns-zstd --example profile_decode --release
//!
//! Then in another shell:
//!   sample <pid> 5 -file /tmp/ns_zstd.sample.txt

#![cfg(feature = "std")]

use std::time::{Duration, Instant};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(default)
}

fn gen_deterministic_bytes(len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut x: u64 = 0x1234_5678_9abc_def0;
    for (i, b) in out.iter_mut().enumerate() {
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

fn mbps(bytes: usize, secs: f64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0) / secs
}

fn main() {
    let mb = env_usize("NS_ZSTD_PROFILE_MB", 64);
    let secs = env_f64("NS_ZSTD_PROFILE_SECS", 60.0);
    let level = std::env::var("NS_ZSTD_PROFILE_LEVEL").unwrap_or_else(|_| "default".to_string());
    let level = match level.as_str() {
        "fastest" => ns_zstd::encoding::CompressionLevel::Fastest,
        "uncompressed" => ns_zstd::encoding::CompressionLevel::Uncompressed,
        _ => ns_zstd::encoding::CompressionLevel::Default,
    };

    eprintln!("pid={}", std::process::id());
    let input = gen_deterministic_bytes(mb * 1024 * 1024);
    let compressed = ns_zstd::encoding::compress_to_vec(input.as_slice(), level);
    // Our encoder currently uses `single_segment=false`, so Window_Descriptor is present
    // at byte 5 (magic 4 bytes + descriptor 1 byte).
    let win_desc = compressed.get(5).copied().unwrap_or(0);
    let exp = win_desc >> 3;
    let mant = win_desc & 0x7;
    let window_log = 10u32 + exp as u32;
    let window_base = 1u64 << window_log;
    let window_add = (window_base / 8) * u64::from(mant);
    let win = window_base + window_add;
    eprintln!(
        "input={} MiB compressed={} bytes ratio={:.3} level={} window_desc=0x{:02x} window={} bytes ({:.3} MiB)",
        mb,
        compressed.len(),
        (compressed.len() as f64) / (input.len() as f64),
        std::env::var("NS_ZSTD_PROFILE_LEVEL").unwrap_or_else(|_| "default".to_string()),
        win_desc,
        win,
        (win as f64) / (1024.0 * 1024.0)
    );

    // Warmup + correctness.
    {
        let mut out = Vec::with_capacity(input.len());
        let mut frame = ns_zstd::decoding::FrameDecoder::new();
        frame.decode_all_to_vec(&compressed, &mut out).expect("decode");
        assert_eq!(out, input);
    }

    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs_f64(secs);
    let mut out = Vec::with_capacity(input.len());
    let mut frame = ns_zstd::decoding::FrameDecoder::new();
    let mut iters: u64 = 0;
    let mut total_bytes: u64 = 0;
    while Instant::now() < deadline {
        out.clear();
        frame.decode_all_to_vec(&compressed, &mut out).expect("decode");
        std::hint::black_box(out.as_slice());
        iters += 1;
        total_bytes += out.len() as u64;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "iters={} total_out={} MiB throughput={:.1} MiB/s",
        iters,
        total_bytes / (1024 * 1024),
        mbps(total_bytes as usize, elapsed)
    );
}
