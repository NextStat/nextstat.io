# ruzstd — NextStat optimized fork

Based on [ruzstd 0.8.2](https://github.com/KillingSpark/zstd-rs) by Moritz Borcherding (MIT license).

## Why fork?

ruzstd is 1.4-3.5x slower than C libzstd. For ROOT files up to 100 GB (TTree ntuples),
decompression is the bottleneck. These optimizations target the hottest decoder paths.

## Applied optimizations

1. **`BitReaderReversed::refill`** — removed `#[cold]`, added `#[inline(always)]`.
   This is the hottest function in the decoder; `#[cold]` was causing LLVM to not inline it.

2. **Static lookup tables** for literal-length and match-length codes.
   Replaced 36/53-arm `match` with `static` arrays + `get_unchecked`.

3. **`assert!` → `debug_assert!`** in release hot paths (FSE table build, Huffman table,
   sequence execution).

4. **Removed heap allocations** for predefined FSE distribution tables.
   `Vec::from(&STATIC_SLICE[..])` → `&STATIC_SLICE`.

5. **Pre-sized Huffman decode buffer** — `target.resize()` + `get_unchecked_mut` instead of
   `target.push()` per decoded byte. Eliminates per-symbol capacity check.

6. **`#[inline(always)]`** on `HuffmanDecoder::decode_symbol/next_state` and
   `FSEDecoder::decode_symbol/update_state`.

## Removed from upstream

- `tests/` directory (depends on dev-deps `zstd`, `rand`, `criterion` and test corpora)
- `dictionary/` module (unused)
- `fuzz_exports` feature flag
- `dict_builder` feature flag
- `rustc-dep-of-std` feature flag
- Benchmarks (`benches/`)

## Future optimizations (not yet implemented)

- Short-offset match copy (8-byte load/store trick for offset < 8)
- Flat buffer instead of RingBuffer for non-streaming decode
- Two-symbol Huffman decode unrolling
- Interleaved FSE state updates
- Fused sequence decode + execute (single pass)
