#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::io::Read as _;
use std::io::Write as _;

#[test]
fn streaming_encoder_roundtrips_single_frame() {
    let input: Vec<u8> = (0u32..200_000u32).flat_map(|x| x.to_le_bytes()).collect();

    let mut out = Vec::new();
    {
        let mut enc = ns_zstd::encoding::StreamingEncoder::new(
            &mut out,
            ns_zstd::encoding::CompressionLevel::Default,
        );

        // Write in small chunks and force a few flushes to exercise partial-block handling.
        for chunk in input.chunks(777) {
            enc.write_all(chunk).unwrap();
        }
        enc.flush().unwrap();
        // Finish the frame.
        let _w = enc.finish().unwrap();
    }

    let mut dec = ns_zstd::decoding::StreamingDecoder::new(std::io::Cursor::new(&out)).unwrap();
    let mut decoded = Vec::with_capacity(input.len());
    dec.read_to_end(&mut decoded).unwrap();
    assert_eq!(decoded, input);
}
