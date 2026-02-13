#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

fn make_pseudorandom(len: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut x = seed;
    for _ in 0..len {
        // Simple LCG-ish generator, deterministic and cheap.
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push((x ^ (x >> 11) ^ (x >> 19)) as u8);
    }
    out
}

#[test]
fn fastest_does_not_inflate_on_randomish_inputs() {
    for seed in [1u32, 2, 3, 12345, 0xDEADBEEF] {
        let input = make_pseudorandom(128 * 1024, seed);
        let unc = ns_zstd::encoding::compress_to_vec(
            input.as_slice(),
            ns_zstd::encoding::CompressionLevel::Uncompressed,
        );
        let fast = ns_zstd::encoding::compress_to_vec(
            input.as_slice(),
            ns_zstd::encoding::CompressionLevel::Fastest,
        );
        assert!(
            fast.len() <= unc.len(),
            "Fastest inflated vs Uncompressed: seed={} fast={} unc={}",
            seed,
            fast.len(),
            unc.len()
        );
    }
}

#[test]
fn default_does_not_inflate_on_randomish_inputs() {
    for seed in [1u32, 2, 3, 12345, 0xDEADBEEF] {
        let input = make_pseudorandom(128 * 1024, seed);
        let unc = ns_zstd::encoding::compress_to_vec(
            input.as_slice(),
            ns_zstd::encoding::CompressionLevel::Uncompressed,
        );
        let def = ns_zstd::encoding::compress_to_vec(
            input.as_slice(),
            ns_zstd::encoding::CompressionLevel::Default,
        );
        assert!(
            def.len() <= unc.len(),
            "Default inflated vs Uncompressed: seed={} def={} unc={}",
            seed,
            def.len(),
            unc.len()
        );
    }
}
