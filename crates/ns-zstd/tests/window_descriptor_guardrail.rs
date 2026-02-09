#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

fn window_desc_for_frame(encoded: &[u8]) -> u8 {
    // Our encoder emits:
    // - 4 bytes magic
    // - 1 byte Frame_Header_Descriptor
    // - 1 byte Window_Descriptor (since single_segment=false)
    assert!(encoded.len() >= 6, "encoded frame too small");
    encoded[5]
}

#[test]
fn window_descriptor_default_is_256k() {
    let input: Vec<u8> = (0u32..50_000u32).flat_map(|i| i.to_le_bytes()).collect();
    let encoded = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Default,
    );
    let desc = window_desc_for_frame(&encoded);
    // 256KiB = 1<<(10+8), mantissa=0 => (8<<3)|0 = 0x40.
    assert_eq!(desc, 0x40);
}

#[test]
fn window_descriptor_fastest_is_128k() {
    let input: Vec<u8> = (0u32..50_000u32).flat_map(|i| i.to_le_bytes()).collect();
    let encoded = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Fastest,
    );
    let desc = window_desc_for_frame(&encoded);
    // 128KiB = 1<<(10+7), mantissa=0 => (7<<3)|0 = 0x38.
    assert_eq!(desc, 0x38);
}
