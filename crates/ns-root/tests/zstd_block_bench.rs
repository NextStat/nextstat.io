use ns_root::decompress::decompress;
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(default)
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0) / secs
}

#[inline]
#[allow(clippy::manual_is_multiple_of)]
fn is_multiple_of_u32(value: u32, divisor: u32) -> bool {
    divisor != 0 && value % divisor == 0
}

fn gen_deterministic_bytes(len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    // Simple xorshift-ish fill; deterministic and cheap.
    let mut x = 0x243F_6A88_85A3_08D3u64;
    for b in out.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = (x & 0xFF) as u8;
    }
    out
}

fn gen_rootish_bytes(len: usize) -> Vec<u8> {
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
        let w = if is_multiple_of_u32(i, 7) { 1.05_f32 } else { 1.0_f32 };
        out.extend_from_slice(&w.to_le_bytes());

        i = i.wrapping_add(1);
    }
    out.resize(len, 0);
    out
}

fn push_le24(dst: &mut Vec<u8>, v: usize) {
    assert!(v <= 0xFF_FFFF);
    dst.push((v & 0xFF) as u8);
    dst.push(((v >> 8) & 0xFF) as u8);
    dst.push(((v >> 16) & 0xFF) as u8);
}

fn make_root_zs_blocks(uncompressed: &[u8], level: i32, block_u: usize) -> (Vec<u8>, usize) {
    assert!(block_u > 0);
    assert!(block_u <= 0xFF_FFFF);

    let mut out = Vec::new();
    let mut compressed_total = 0usize;
    let mut pos = 0;
    while pos < uncompressed.len() {
        let end = (pos + block_u).min(uncompressed.len());
        let chunk = &uncompressed[pos..end];

        let mut compressed = Vec::with_capacity(zstd_safe::compress_bound(chunk.len()).max(1));
        zstd_safe::compress(&mut compressed, chunk, level).expect("zstd_safe compress");
        compressed_total += compressed.len();

        out.extend_from_slice(b"ZS");
        out.push(0x04); // ROOT "method" byte for ZSTD blocks (ignored by decoder)
        push_le24(&mut out, compressed.len());
        push_le24(&mut out, chunk.len());
        out.extend_from_slice(&compressed);

        pos = end;
    }

    (out, compressed_total)
}

#[test]
#[ignore]
fn bench_root_style_zstd_blocks() {
    // Bench settings:
    // - NS_ROOT_ZSTD_MB=128
    // - NS_ROOT_ZSTD_ITERS=30
    // - NS_ROOT_ZSTD_LEVEL=3
    // - NS_ROOT_ZSTD_BLOCK_U=8388608  (<= 0xFF_FFFF)
    let mb = env_usize("NS_ROOT_ZSTD_MB", 64);
    let iters = env_usize("NS_ROOT_ZSTD_ITERS", 30);
    let level =
        std::env::var("NS_ROOT_ZSTD_LEVEL").ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(3);
    let block_u = env_usize("NS_ROOT_ZSTD_BLOCK_U", 8 * 1024 * 1024);
    let data = std::env::var("NS_ROOT_ZSTD_DATA").unwrap_or_else(|_| "rootish".to_string());

    let total = mb * 1024 * 1024;
    let input = match data.as_str() {
        "deterministic" => gen_deterministic_bytes(total),
        _ => gen_rootish_bytes(total),
    };
    let (src, compressed_total) = make_root_zs_blocks(&input, level, block_u);

    eprintln!(
        "root_zs_blocks: input={} MiB data={} blocks_u={} level={} compressed_bytes={} ratio={:.3} src_bytes={}",
        mb,
        data,
        block_u,
        level,
        compressed_total,
        (compressed_total as f64) / (input.len() as f64),
        src.len()
    );

    // Warmup correctness.
    let out = decompress(&src, input.len()).expect("decompress");
    assert_eq!(out, input);

    let mut best = 0.0f64;
    let mut total_tput = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = decompress(&src, input.len()).expect("decompress");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(out.len());
        let tput = mbps(out.len(), dt);
        total_tput += tput;
        if tput > best {
            best = tput;
        }
    }

    eprintln!(
        "ns-root decompress(ROOT ZS blocks): best={:.1} MiB/s avg={:.1} MiB/s (iters={})",
        best,
        total_tput / (iters as f64),
        iters
    );
}
