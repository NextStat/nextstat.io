#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

/// Guardrail: Default (hash-chain) should meaningfully outperform Fastest (single-slot hash)
/// on adversarial collision-heavy inputs where Fastest locks onto an early, low-quality match.
#[test]
fn default_beats_fastest_on_collision_heavy_pattern() {
    // 5-byte repeating pattern: every 5-byte window is identical ("abcde"),
    // so the Fastest matcher stores exactly one position for the key and never updates it.
    //
    // We place an "abcdeX" early so the stored key points at a match that breaks at byte 6.
    // Later we include a large "abcdeY" + repeated pattern region that Default can match well,
    // but Fastest keeps taking tiny 5-byte matches (or falls back to Raw).
    let unit = b"abcde";
    let tail = unit.repeat(25_000); // ~125 KiB

    let mut input = Vec::with_capacity(2 * tail.len() + 16);
    input.extend_from_slice(b"abcdeX");
    input.extend_from_slice(&tail);
    input.extend_from_slice(b"abcdeY");
    input.extend_from_slice(&tail);

    let fast = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Fastest,
    );
    let def = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Default,
    );

    assert!(
        def.len() + 8192 < fast.len(),
        "expected Default to beat Fastest by >=8KiB; fast={} default={}",
        fast.len(),
        def.len()
    );
}
