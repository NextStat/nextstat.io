#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

#[test]
fn default_is_not_worse_than_fastest_on_lazy_friendly_pattern() {
    // Construct a pattern where a short match exists at P ("baaaaa" seen before),
    // but a much longer match exists at P+1 ("aaaaa..." seen before).
    //
    // Greedy encoders tend to take the short match; lazy encoders should prefer the longer one.
    let mut input = Vec::new();
    input.extend(std::iter::repeat_n(b'a', 30_000));
    input.extend_from_slice(b"baaaaa");
    input.push(b'c'); // break the earlier "baaaaa..." match
    input.extend(std::iter::repeat_n(b'a', 30_000));

    // Now repeat a "b + long a-run" several times to amplify any lazy benefit.
    for _ in 0..64 {
        input.push(b'b');
        input.extend(std::iter::repeat_n(b'a', 2048));
    }

    let fast = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Fastest,
    );
    let def = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Default,
    );

    assert!(
        def.len() <= fast.len(),
        "expected Default <= Fastest; fast={} default={}",
        fast.len(),
        def.len()
    );
}
