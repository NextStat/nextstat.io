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

7. **Flat output buffer** (`FlatDecodeBuffer`) — replaced `RingBuffer`+`DecodeBuffer` with
   a flat `Vec<u8>`. Eliminates modulo arithmetic, wraparound branches, and two-slice
   splits on every write. Match copies become single `memcpy` or pointer arithmetic.

8. **Fast match copy** (`fast_copy_match`) — 3-tier copy strategy:
   - offset ≥ 16: `copy_nonoverlapping` (compiler auto-vectorizes)
   - offset 8-15: unaligned 8-byte load/store chunks
   - offset 1: RLE fast path (8-byte pattern fill)
   - offset 2-7: copy-doubling trick (log₂ memcpy calls)

9. **Fused decode + execute** (`fused.rs`) — single-pass sequence processing.
   Decodes each FSE sequence and immediately executes it (copy literals + match copy),
   eliminating the intermediate `Vec<Sequence>` and the second pass over all sequences.

10. **Interleaved FSE state updates** — reads bits for all 3 FSE decoders (LL, ML, OF)
    in a single `get_bits_triple` call (one refill), then does 3 independent table lookups
    that the CPU can pipeline via out-of-order execution.

11. **2-stream interleaved Huffman decode** — for 4-stream literals, processes streams
    in pairs (0,1) then (2,3). Two independent Huffman state machines let the CPU
    pipeline table lookups from one stream while the other's memory load completes.

## Performance

Benchmark: 4 MB structured data, compression ratio 3.02x (bench_zstd.rs, Apple M-series)

| Version                   | Bulk (median) | Streaming (median) |
|---------------------------|---------------|--------------------|
| Original ruzstd 0.8.2    | ~420 MB/s     | ~420 MB/s          |
| Fork (optimizations 1-6)  | ~530 MB/s     | ~440 MB/s          |
| Fork (optimizations 1-11) | **824 MB/s**  | **808 MB/s**       |
| C libzstd (reference)     | ~1500-2000 MB/s | —                |

Gap with C libzstd: reduced from **3.4x → ~2x**.

Note: benchmark uses ruzstd's `Fastest` encoder which produces mostly raw/RLE literals.
With C libzstd-compressed ROOT files (level 1-4, Huffman 4-stream), the Huffman
interleaving (opt 11) provides additional benefit beyond what's shown here.

## Removed from upstream

- `tests/` directory (depends on dev-deps `zstd`, `rand`, `criterion` and test corpora)
- `dictionary/` module (unused)
- `fuzz_exports` feature flag
- `dict_builder` feature flag
- `rustc-dep-of-std` feature flag
- Benchmarks (`benches/`)

## Future optimizations (not yet implemented)

- Explicit SIMD (SSE2/NEON) for Huffman decode and match copy
- 4-stream Huffman interleave (all 4 simultaneously, vs current 2-at-a-time)
- Profile-guided optimization with real ROOT file payloads
