#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    // Keep allocations bounded. The decompressor itself enforces a higher cap,
    // but the fuzzer should stay fast.
    let expected_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize % (1 << 20);
    let src = &data[4..];

    let _ = ns_root::decompress::decompress(src, expected_len);
});
