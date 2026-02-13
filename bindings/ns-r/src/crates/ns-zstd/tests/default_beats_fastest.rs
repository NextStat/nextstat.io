#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

#[test]
fn default_compresses_better_than_fastest_when_repeat_exceeds_128k_window() {
    // Construct 2x256KiB where the second half is an exact repeat of the first half.
    // With a 128KiB window, the first repeated 128KiB block cannot see the earlier block1.
    // With a 256KiB window (Default), both repeated blocks should compress well.
    const HALF: usize = 256 * 1024;

    let mut first = Vec::with_capacity(HALF);
    for i in 0..HALF {
        // Simple deterministic "random-looking" bytes.
        let x = (i as u32).wrapping_mul(0x9E37_79B1).rotate_left(13).wrapping_add(0x7F4A_7C15);
        first.push((x ^ (x >> 7) ^ (x >> 16)) as u8);
    }

    let mut input = Vec::with_capacity(2 * HALF);
    input.extend_from_slice(&first);
    input.extend_from_slice(&first);

    let fast = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Fastest,
    );
    let def = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Default,
    );

    // Require a meaningful improvement, not just a few bytes.
    assert!(
        def.len() + 4096 < fast.len(),
        "expected Default to beat Fastest by >=4KiB; fast={} default={}",
        fast.len(),
        def.len()
    );
}
