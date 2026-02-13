#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::fs;
use std::io::Read as _;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_fixtures").join(name)
}

#[test]
fn decode_fixture_matches_libzstd() {
    let path = fixture_path("abc.txt.zst");
    let compressed = fs::read(&path).expect("read fixture");

    // ns-zstd streaming decode (does not require pre-sizing).
    let mut sdec = ns_zstd::decoding::StreamingDecoder::new(compressed.as_slice())
        .expect("ns_zstd streaming init");
    let mut stream_out = Vec::new();
    sdec.read_to_end(&mut stream_out).expect("ns_zstd streaming decode");

    // libzstd reference decode (via zstd-safe; requires enough spare capacity)
    let mut ref_out = Vec::with_capacity(stream_out.len().max(1));
    zstd_safe::decompress(&mut ref_out, &compressed).expect("libzstd decode");

    assert_eq!(stream_out, ref_out);

    // ns-zstd bulk decode (requires enough spare capacity)
    let mut bulk_out = Vec::with_capacity(ref_out.len().max(1));
    let mut dec = ns_zstd::decoding::FrameDecoder::new();
    dec.decode_all_to_vec(&compressed, &mut bulk_out).expect("ns_zstd bulk decode");
    assert_eq!(bulk_out, ref_out);
}
