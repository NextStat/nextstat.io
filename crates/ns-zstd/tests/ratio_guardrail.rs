#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

fn gen_rootish_bytes(len: usize) -> Vec<u8> {
    // Deterministic, moderately compressible payload approximating typical ns-root IO patterns:
    // repeated integer structures + gently varying floats.
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
        let w = if i.is_multiple_of(7) { 1.05_f32 } else { 1.0_f32 };
        out.extend_from_slice(&w.to_le_bytes());

        i = i.wrapping_add(1);
    }
    out.resize(len, 0);
    out
}

#[test]
fn default_ratio_within_5pct_of_c_zstd_level3_on_rootish() {
    // Keep this small enough for CI, but large enough to smooth out tiny-input effects.
    let input = gen_rootish_bytes(4 * 1024 * 1024);
    let ns_def = ns_zstd::encoding::compress_to_vec(
        input.as_slice(),
        ns_zstd::encoding::CompressionLevel::Default,
    );

    let bound = zstd_safe::compress_bound(input.len());
    let mut cctx = zstd_safe::CCtx::default();
    let mut c_out = Vec::with_capacity(bound.max(1));
    cctx.compress(&mut c_out, &input, 3).expect("libzstd compress");

    // Guardrail: Default output should be within 5% of C zstd level 3 (smaller is fine).
    let max_ns = (c_out.len() * 105).div_ceil(100); // ceil(c_out * 1.05)
    assert!(
        ns_def.len() <= max_ns,
        "ns_default={} c_zstd_l3={} max_allowed={}",
        ns_def.len(),
        c_out.len(),
        max_ns
    );
}
