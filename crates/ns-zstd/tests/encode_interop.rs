#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

fn decode_with_reference_zstd(compressed: &[u8], expected_len: usize) -> Vec<u8> {
    // Use C libzstd via zstd-safe as an oracle.
    // `zstd_safe::decompress` requires enough output capacity.
    let mut out = Vec::with_capacity(expected_len.max(1));
    zstd_safe::decompress(&mut out, compressed).expect("ref zstd-safe decompress");
    out
}

#[test]
fn encode_fastest_decodes_with_reference_zstd_small_inputs() {
    let cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![1, 2, 3, 4, 5],
        (0u8..=255u8).collect(),
        // Incompressible-ish but deterministic
        (0u32..10_000u32).flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>(),
    ];

    for input in cases {
        let compressed = ns_zstd::encoding::compress_to_vec(
            input.as_slice(),
            ns_zstd::encoding::CompressionLevel::Fastest,
        );
        let decoded = decode_with_reference_zstd(&compressed, input.len());
        assert_eq!(decoded, input);
    }
}

#[test]
fn encode_fastest_decodes_with_reference_zstd_long_match_len() {
    // Construct 2 full blocks with identical content so the second block should find a very long
    // match in the previous block (exercise large ML code paths).
    let block: Vec<u8> = (0u32..(128 * 1024 / 4) as u32).flat_map(|i| i.to_le_bytes()).collect();

    assert_eq!(block.len(), 128 * 1024);

    let mut input = Vec::with_capacity(2 * block.len());
    input.extend_from_slice(&block);
    input.extend_from_slice(&block);

    let compressed = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Fastest,
    );
    let decoded = decode_with_reference_zstd(&compressed, input.len());
    assert_eq!(decoded, input);
}

#[test]
fn encode_default_is_callable_and_decodes_with_reference_zstd() {
    // Until Level 3 is fully implemented, Default must at least be safe to call.
    let input: Vec<u8> = (0u32..50_000u32).flat_map(|i| i.to_le_bytes()).collect();

    let compressed = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Default,
    );
    let decoded = decode_with_reference_zstd(&compressed, input.len());
    assert_eq!(decoded, input);
}
