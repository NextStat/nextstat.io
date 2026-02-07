# ruzstd → C libzstd Performance Gap: Deep Architecture Analysis

**Date**: 2026-02-08
**Current**: 440 MB/s (fork) vs ~1500-2000 MB/s (C libzstd) — **3.4x gap**
**Goal**: Close to ≤1.5-2x gap → **~800-1200 MB/s**

---

## 1. Time Budget at Ratio 3x (4 MB structured data)

At ratio 3x, each 3 output bytes ≈ 1 literal (Huffman) + 2 match-copy (sequences).

```
Pipeline per compressed block:
  decode_literals (Huffman 4-stream)  →  decode_sequences (FSE 3-state)  →  execute_sequences (match copy)
         ~25% CPU                              ~10% CPU                           ~55% CPU
                                                                             ← THE bottleneck
  Overhead (frame parse, alloc, buffer mgmt): ~10%
```

---

## 2. Five Architectural Bottlenecks (ordered by impact)

### 2.1 Two-Pass Decode → Execute (~30-40% of gap)

**Current** (`sequence_section_decoder.rs` + `sequence_execution.rs`):
```
Pass 1: decode ALL sequences → Vec<Sequence>   // heap alloc, write to memory
Pass 2: execute sequences (copy literals + matches)  // read from memory
```

**C libzstd** (`zstd_decompress_block.c`, `ZSTD_decompressSequences_body`):
```
Single pass: decode sequence → execute IMMEDIATELY → next sequence
// FSE states, offset history, literals pointer — ALL in registers
```

**Why it matters**:
- 100K sequences × 12 bytes = 1.2 MB intermediate `Vec<Sequence>` — blows L1 cache
- Double memory traffic: write sequences, then read them back
- FSE state → Sequence struct → offset history requires extra loads

**C reference** — the fused loop (simplified from `ZSTD_decompressSequences_body`):
```c
for ( ; ; ) {
    // 1. Decode one sequence from FSE (all in registers)
    size_t const llBase = seqState.prevOffset[...];
    size_t const mlBase = ...;
    size_t const ofBase = ...;
    // Read extra bits
    BIT_reloadDStream(&seqState.DStream);

    // 2. IMMEDIATELY execute — copy literals
    ZSTD_wildcopy(op, litPtr, ll, ZSTD_no_overlap);
    op += ll; litPtr += ll;

    // 3. IMMEDIATELY execute — match copy
    ZSTD_wildcopy(op, match, ml, ovtype);
    op += ml;

    // 4. Update FSE states (interleaved)
    ZSTD_updateFseStateWithDInfo(&seqState.stateLL, ...);
    ZSTD_updateFseStateWithDInfo(&seqState.stateML, ...);
    ZSTD_updateFseStateWithDInfo(&seqState.stateOffb, ...);
}
```

**Rust implementation design** — new `fused_decode_execute()`:
```rust
/// Fused single-pass: decode FSE sequence → execute immediately.
/// Eliminates Vec<Sequence> and second pass entirely.
pub fn fused_decode_execute(
    section: &SequencesHeader,
    source: &[u8],
    fse: &mut FSEScratch,
    literals: &[u8],
    buffer: &mut FlatDecodeBuffer,  // Phase 2.2 flat buffer
    offset_hist: &mut [u32; 3],
) -> Result<(), DecodeBlockError> {
    let bytes_read = maybe_update_fse_tables(section, source, fse)?;
    let bit_stream = &source[bytes_read..];
    let mut br = BitReaderReversed::new(bit_stream);

    // Skip padding
    skip_padding(&mut br)?;

    let mut ll_dec = FSEDecoder::new(&fse.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&fse.match_lengths);
    let mut of_dec = FSEDecoder::new(&fse.offsets);

    ll_dec.init_state(&mut br)?;
    of_dec.init_state(&mut br)?;
    ml_dec.init_state(&mut br)?;

    let mut lit_idx: usize = 0;

    for seq_idx in 0..section.num_sequences {
        // ── Decode (registers only, no memory write) ──
        let ll_code = ll_dec.decode_symbol();
        let ml_code = ml_dec.decode_symbol();
        let of_code = of_dec.decode_symbol();

        let (ll_value, ll_num_bits) = lookup_ll_code(ll_code);
        let (ml_value, ml_num_bits) = lookup_ml_code(ml_code);

        let (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_num_bits, ll_num_bits);
        let offset_raw = obits as u32 + (1u32 << of_code);
        let ll = (ll_value + ll_add as u32) as usize;
        let ml = (ml_value + ml_add as u32) as usize;

        // ── Execute IMMEDIATELY ──

        // 1) Copy literals
        if ll > 0 {
            let lit_end = lit_idx + ll;
            buffer.push_slice(&literals[lit_idx..lit_end]);
            lit_idx = lit_end;
        }

        // 2) Resolve offset & update history
        let actual_offset = do_offset_history(offset_raw, ll as u32, offset_hist);

        // 3) Match copy (uses fast_copy_match from Phase 2.3)
        if ml > 0 {
            buffer.repeat_fast(actual_offset as usize, ml)?;
        }

        // ── Update FSE states ──
        if seq_idx + 1 < section.num_sequences as usize {
            ll_dec.update_state(&mut br);
            ml_dec.update_state(&mut br);
            of_dec.update_state(&mut br);
        }
    }

    // Copy remaining literals after last sequence
    if lit_idx < literals.len() {
        buffer.push_slice(&literals[lit_idx..]);
    }

    Ok(())
}
```

**Estimated gain**: +30-40% total throughput (single biggest win)
**Complexity**: High — requires threading `FlatDecodeBuffer` through block_decoder.rs
**Risk**: Medium — logic is well-understood, just needs careful porting

---

### 2.2 RingBuffer → Flat Vec (~15-20% of gap)

**Current** (`ringbuffer.rs`, 888 lines):
```rust
// EVERY write operation:
self.tail = (self.tail + len) % self.cap;  // modulo on every op
// extend() splits into two free slices → two memcpy calls
// extend_from_within_unchecked() → 4 major branch cases (170 lines!)
```

**C libzstd**: Flat `BYTE* op` pointer. Output writes to `*op++`. Match copies = single `memcpy` or `ZSTD_wildcopy`. Zero wraparound.

**Rust implementation** — `FlatDecodeBuffer`:
```rust
pub struct FlatDecodeBuffer {
    buf: Vec<u8>,
    pub dict_content: Vec<u8>,
    pub window_size: usize,
    total_output_counter: u64,
    #[cfg(feature = "hash")]
    pub hash: twox_hash::XxHash64,
}

impl FlatDecodeBuffer {
    pub fn new(window_size: usize) -> Self {
        Self {
            buf: Vec::with_capacity(window_size),
            dict_content: Vec::new(),
            window_size,
            total_output_counter: 0,
            #[cfg(feature = "hash")]
            hash: twox_hash::XxHash64::with_seed(0),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize { self.buf.len() }

    #[inline(always)]
    pub fn push_slice(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        self.total_output_counter += data.len() as u64;
    }

    /// Fast match copy: offset bytes back, copy match_length bytes.
    /// Uses 8-byte overlap trick for short offsets (see §2.3).
    #[inline(always)]
    pub fn repeat_fast(&mut self, offset: usize, match_length: usize) -> Result<(), DecodeBufferError> {
        if offset > self.buf.len() {
            return self.repeat_from_dict(offset, match_length);
        }
        if offset == 0 {
            return Err(DecodeBufferError::OffsetTooBig { offset, buf_len: self.buf.len() });
        }

        self.buf.reserve(match_length);
        let old_len = self.buf.len();
        unsafe {
            self.buf.set_len(old_len + match_length);
            let base = self.buf.as_mut_ptr();
            fast_copy_match(base.add(old_len - offset), base.add(old_len), offset, match_length);
        }
        self.total_output_counter += match_length as u64;
        Ok(())
    }

    /// Drain all decoded output.
    pub fn drain_all(&mut self) -> Vec<u8> {
        #[cfg(feature = "hash")]
        self.hash.write(&self.buf);
        let result = core::mem::take(&mut self.buf);
        result
    }

    /// Drain keeping window_size bytes for backreferences.
    pub fn drain_to_window_size(&mut self) -> Option<Vec<u8>> {
        if self.buf.len() <= self.window_size {
            return None;
        }
        let drain_amount = self.buf.len() - self.window_size;
        #[cfg(feature = "hash")]
        self.hash.write(&self.buf[..drain_amount]);
        let drained: Vec<u8> = self.buf[..drain_amount].to_vec();
        // Shift remaining to front
        self.buf.copy_within(drain_amount.., 0);
        self.buf.truncate(self.window_size);
        Some(drained)
    }
}
```

**Integration strategy**: Use `FlatDecodeBuffer` when the full frame is decoded at once (the `StreamingDecoder::read_to_end` path — our ROOT file use case). Keep `RingBuffer`-based `DecodeBuffer` for true streaming with memory limits. A simple enum dispatch:

```rust
pub enum OutputBuffer {
    Ring(DecodeBuffer),    // streaming, memory-constrained
    Flat(FlatDecodeBuffer), // bulk decode, our hot path
}
```

**Estimated gain**: +15-20%
**Complexity**: Medium — new struct + trait/enum dispatch in block_decoder
**Risk**: Low — well-contained, no algorithmic changes

---

### 2.3 8-Byte Match Copy Trick (~15-20% of gap)

**Current** (`decode_buffer.rs:101-129`):
```rust
fn repeat_in_chunks(&mut self, offset: usize, match_length: usize, start_idx: usize) {
    while copied_counter_left > 0 {
        let chunksize = usize::min(offset, copied_counter_left);
        unsafe { self.buffer.extend_from_within_unchecked(start_idx, chunksize) };
        // For offset=1, length=1000: THIS IS 1000 ITERATIONS
        // Each through RingBuffer::extend_from_within_unchecked (4+ branches!)
    }
}
```

**C libzstd** (`ZSTD_wildcopy` + `ZSTD_execSequence`):
```c
// wildcopy for offset >= 16: 16-byte chunk copies (SIMD on x86/ARM)
ZSTD_copy16(op, ip);
if (16 >= length) return;
op += 16; ip += 16;
do { COPY16(op, ip); COPY16(op, ip); } while (op < oend);

// For offset < 16 but >= 8: 8-byte copies (overlap-safe)
do { COPY8(op, ip); } while (op < oend);

// The key: even for short matches, always copy at least 8 bytes.
// Overshoot is safe because output buffer has WILDCOPY_OVERLENGTH (32 bytes) padding.
```

**Rust implementation** — `fast_copy_match`:
```rust
/// SAFETY:
/// - `src` and `dst` within same allocation
/// - `dst` has `length` bytes of writable space (+ 32 bytes overshoot OK)
/// - `src + offset ≤ dst` (source is behind destination)
#[inline(always)]
unsafe fn fast_copy_match(src: *const u8, dst: *mut u8, offset: usize, length: usize) {
    if offset >= 16 {
        // Non-overlapping: bulk copy. Compiler can auto-vectorize.
        // Use copy_nonoverlapping which maps to memcpy.
        core::ptr::copy_nonoverlapping(src, dst, length);
    } else if offset >= 8 {
        // Mildly overlapping but offset >= 8: 8-byte copies are safe
        // because we always read behind write position by ≥8 bytes.
        let mut i = 0usize;
        while i + 8 <= length {
            (dst.add(i) as *mut u64).write_unaligned(
                (src.add(i) as *const u64).read_unaligned()
            );
            i += 8;
        }
        // Tail bytes
        if i < length {
            core::ptr::copy_nonoverlapping(src.add(i), dst.add(i), length - i);
        }
    } else if offset == 0 {
        // Error case — should not happen, caller validates
        return;
    } else {
        // Short offset (1-7): The 8-byte overlap trick.
        //
        // Key insight: reading 8 bytes from `src` when offset < 8 creates
        // a naturally repeating pattern due to overlapping with `dst`.
        //
        // Example offset=3, src=[A,B,C]:
        //   Read 8 bytes from src: [A,B,C,?,?,?,?,?] (? = whatever follows)
        //   Write 8 bytes to dst:  [A,B,C,A,B,C,A,B] (overlapping read!)
        //   Actually — we must copy correctly. The trick:
        //   dst[0..8] ← src[0..8] (unaligned read from src which overlaps)
        //
        // After first 8 bytes, the pattern is established in dst.
        // Then continue copying from dst itself with 8-byte chunks.

        // Step 1: First 8 bytes — read from src, creates repeat pattern
        (dst as *mut u64).write_unaligned(
            (src as *const u64).read_unaligned()
        );

        if length <= 8 { return; }

        // Step 2: Now we have ≥8 bytes of pattern at dst.
        // For offsets 1,2,4: the pattern repeats every offset bytes.
        // Copy from the established pattern using 8-byte chunks.
        let mut pos = 8usize;
        // Use the destination itself as source (the pattern is there)
        // Since pos >= 8 and offset < 8, we can always read 8 bytes
        // from (dst + pos - 8) which is in the already-written zone.
        while pos + 8 <= length {
            let v = (dst.add(pos - offset) as *const u64).read_unaligned();
            (dst.add(pos) as *mut u64).write_unaligned(v);
            pos += 8;
        }
        // Tail
        while pos < length {
            *dst.add(pos) = *dst.add(pos - offset);
            pos += 1;
        }
    }
}
```

**Why offset < 8 matters for ROOT data**:
- offset=1: RLE of zeros/padding — extremely common in structured arrays
- offset=4: repeating f32 patterns (jet pT = 25.0 repeated)
- offset=8: repeating f64 patterns

Current ruzstd: 1000 iterations for offset=1, length=1000.
With this: ~125 iterations (8 bytes each) + 1 initial 8-byte copy = **8x faster for this case**.

**Overshoot padding**: Like C libzstd's `WILDCOPY_OVERLENGTH = 32`, we reserve 32 extra bytes in `FlatDecodeBuffer` to allow safe overshooting:
```rust
self.buf.reserve(match_length + 32);  // 32 bytes overshoot padding
```

**Estimated gain**: +15-20%
**Complexity**: Medium (unsafe pointer arithmetic, needs careful testing)
**Risk**: Medium — off-by-one in overlap logic = silent corruption

---

### 2.4 Huffman 2-Symbol Unroll (~10-15% of gap)

**Current** (`literals_section_decoder.rs:122-130`):
```rust
while br.bits_remaining() > threshold {
    unsafe { *out.get_unchecked_mut(write_idx) = decoder.decode_symbol() };
    write_idx += 1;
    decoder.next_state(&mut br);  // data dependency: state₁ → bits → state₂
}
// Each symbol: table[state] → read bits → new state → table[state] → ...
// SERIAL dependency chain — CPU pipeline stalls waiting for table lookup
```

**C libzstd** (`HUF_decompress4X1_usingDTable_internal_body`):
```c
// Two interleaved decoders per stream, hiding table lookup latency:
HUF_DECODE_SYMBOLX1_2(op1, &bitD1);  // decoder A: lookup + bits
HUF_DECODE_SYMBOLX1_2(op2, &bitD2);  // decoder B: lookup + bits (INDEPENDENT!)
HUF_DECODE_SYMBOLX1_2(op3, &bitD3);
HUF_DECODE_SYMBOLX1_2(op4, &bitD4);
// CPU sees 4 independent instruction streams → fills pipeline
```

**Rust implementation** — 2-symbol unroll for single stream:
```rust
// threshold = -(max_num_bits as isize)
// After first next_state: bits drop by at most max_num_bits
// For 2 symbols: need bits_remaining > 0 (= threshold + max_num_bits)

let threshold_2x = 0isize;  // = -(max_num_bits) + max_num_bits

// Unrolled loop: 2 symbols per iteration
while br.bits_remaining() > threshold_2x {
    // Symbol 1
    unsafe { *out.get_unchecked_mut(write_idx) = decoder.decode_symbol() };
    write_idx += 1;
    decoder.next_state(&mut br);

    // Check: did we exhaust the stream on the first symbol?
    if br.bits_remaining() <= threshold {
        break;
    }

    // Symbol 2 (independent from symbol 1's OUTPUT, shares bit stream)
    unsafe { *out.get_unchecked_mut(write_idx) = decoder.decode_symbol() };
    write_idx += 1;
    decoder.next_state(&mut br);
}

// Tail: 0 or 1 remaining symbols
while br.bits_remaining() > threshold {
    unsafe { *out.get_unchecked_mut(write_idx) = decoder.decode_symbol() };
    write_idx += 1;
    decoder.next_state(&mut br);
}
```

**Better approach** — 4-stream interleave (what C does):

Instead of decoding stream1 fully → stream2 fully → ..., interleave two streams:

```rust
// Process streams in pairs for ILP
let streams = [stream1, stream2, stream3, stream4];
let mut decoders = streams.map(|s| {
    let mut br = BitReaderReversed::new(s);
    skip_padding(&mut br);
    let mut dec = HuffmanDecoder::new(&scratch.table);
    dec.init_state(&mut br);
    (dec, br)
});

// Interleaved: decode from stream A, then stream B
// CPU can pipeline the table lookups
let threshold = -(scratch.table.max_num_bits as isize);
for pair in decoders.chunks_mut(2) {
    let (dec_a, br_a) = &mut pair[0];
    let (dec_b, br_b) = &mut pair[1];

    while br_a.bits_remaining() > threshold && br_b.bits_remaining() > threshold {
        // Stream A symbol
        unsafe { *out.get_unchecked_mut(write_idx) = dec_a.decode_symbol() };
        write_idx += 1;
        dec_a.next_state(br_a);

        // Stream B symbol (INDEPENDENT lookup — fills pipeline)
        unsafe { *out.get_unchecked_mut(write_idx) = dec_b.decode_symbol() };
        write_idx += 1;
        dec_b.next_state(br_b);
    }
    // Drain remaining from each stream individually
    // ...
}
```

**Caveat**: The 4-stream interleave changes output ORDER. In C libzstd, each stream decodes a known fraction of symbols (regenerated_size / 4). We need to track per-stream write positions:
```rust
let quarter = regen / 4;
let mut pos = [0, quarter, 2*quarter, 3*quarter]; // write positions per stream
```

**Estimated gain**: +10-15%
**Complexity**: Medium for 2-symbol unroll, High for 4-stream interleave
**Risk**: Low (2-symbol), Medium (4-stream — output ordering)

---

### 2.5 Interleaved FSE State Updates (~5-10% of gap)

**Current** (`sequence_section_decoder.rs:204-206`):
```rust
ll_dec.update_state(br);  // read bits → table lookup (serial)
ml_dec.update_state(br);  // read bits → table lookup (serial)
of_dec.update_state(br);  // read bits → table lookup (serial)
```

Each `update_state` does:
1. Read `num_bits` from `state.num_bits`
2. Call `br.get_bits(num_bits)` — may refill
3. Compute `new_state = baseline + add`
4. Table lookup `self.table.decode[new_state]`

Steps 1→2→3→4 are dependent. Three sequential calls = 12 dependent steps.

**C libzstd** (`ZSTD_updateFseStateWithDInfo`):
```c
// Read all bits first (single refill), then do independent lookups:
BIT_reloadDStream(&seqState->DStream);  // ONE refill for all three

// Three independent state updates — CPU can overlap table lookups
ZSTD_updateFseStateWithDInfo(&seqState->stateLL, ...);
ZSTD_updateFseStateWithDInfo(&seqState->stateML, ...);
ZSTD_updateFseStateWithDInfo(&seqState->stateOffb, ...);
```

**Rust implementation** — batch bit reading:
```rust
// In the fused loop, after executing sequence:
br.refill();  // ONE refill guarantees ≥56 bits available

// Read all three bit values in one go (no refill in between)
let ll_bits_needed = ll_dec.state.num_bits;
let ml_bits_needed = ml_dec.state.num_bits;
let of_bits_needed = of_dec.state.num_bits;

let ll_add = br.peek_bits(ll_bits_needed);
br.consume(ll_bits_needed);
let ml_add = br.peek_bits(ml_bits_needed);
br.consume(ml_bits_needed);
let of_add = br.peek_bits(of_bits_needed);
br.consume(of_bits_needed);

// Three INDEPENDENT table lookups (CPU can parallelize)
let ll_new = ll_dec.state.base_line + ll_add as u32;
ll_dec.state = ll_dec.table.decode[ll_new as usize];

let ml_new = ml_dec.state.base_line + ml_add as u32;
ml_dec.state = ml_dec.table.decode[ml_new as usize];

let of_new = of_dec.state.base_line + of_add as u32;
of_dec.state = of_dec.table.decode[of_new as usize];
```

**Bit budget**: FSE accuracy logs are 9+9+8 = 26 bits max per sequence. After refill we have ≥56 bits. Safe.

**Estimated gain**: +5-10%
**Complexity**: Low (straightforward refactor inside the fused loop)
**Risk**: Low

---

## 3. Implementation Order & Dependencies

```
Phase 1: Foundation (can be done in parallel)
├── P1a: FlatDecodeBuffer struct (§2.2)         ~200 lines new code
└── P1b: fast_copy_match function (§2.3)         ~80 lines unsafe

Phase 2: Core Rewrite (depends on Phase 1)
└── P2: fused_decode_execute (§2.1)              ~150 lines
        Uses FlatDecodeBuffer + fast_copy_match
        Integrates interleaved FSE (§2.5)

Phase 3: Huffman (independent of Phase 2)
└── P3: 2-symbol Huffman unroll (§2.4)           ~40 lines changed
        Optional: 4-stream interleave             ~100 lines
```

**Critical path**: P1a + P1b → P2 (includes §2.5)
**Independent**: P3 can be done any time

---

## 4. Projected Throughput

```
                          Median MB/s    Gain
Current fork              440            baseline
+ Flat buffer (P1a)       ~510           +16%
+ fast_copy_match (P1b)   ~610           +20%
+ Fused decode (P2+§2.5)  ~830           +35%
+ Huffman unroll (P3)      ~950           +15%
                                         ─────
Total projected            ~950 MB/s     ~2.2x improvement

vs C libzstd (~1700)       ~1.8x gap     (down from 3.4x)
```

**To close remaining 1.8x gap** (future, requires platform-specific code):
- Explicit SIMD Huffman decode (`std::arch::x86_64` / NEON): +30-50%
- SIMD match copy (16-byte `_mm_storeu_si128`): +10-20%
- BMI2 `pext` for multi-symbol extract: +10-15%
- `portable_simd` (nightly) or `std::simd` when stabilized

**Hard floor**: Pure scalar Rust will likely plateau at ~900-1100 MB/s.
To reach C libzstd parity (~1700) requires SIMD, which is the same conclusion
as every other pure-Rust decompressor (brotli-rs, flate2, lz4_flex).

---

## 5. Testing Strategy

Each phase needs:

1. **Correctness**: Decode test corpus, compare byte-for-byte with `zstd` crate (C binding)
2. **Fuzz**: Reuse upstream ruzstd fuzz harnesses with new paths
3. **Benchmark**: `bench_zstd.rs` (existing) + add per-phase A/B comparison

**Test data profiles**:
- Ratio 3x structured (current bench) — exercises match copy
- Ratio 8x+ sparse (many zeros) — exercises Huffman + short offset
- Ratio 1.5x random-ish — exercises literals copy
- Real ROOT files from test fixtures

---

## 6. Risk Assessment

| Phase | Lines Changed | Unsafe Code | Correctness Risk | Rollback Ease |
|-------|--------------|-------------|-----------------|---------------|
| P1a   | ~200 new     | Minimal     | Low             | Easy (enum)   |
| P1b   | ~80 new      | **Heavy**   | **Medium**      | Easy (fn)     |
| P2    | ~150 new + ~50 modified | Light | Medium       | Medium        |
| P3    | ~40 modified | None new    | Low             | Easy          |

**Highest risk**: P1b (`fast_copy_match`) — incorrect overlap handling = silent data corruption. Mitigate with exhaustive fuzz testing for all offset/length combinations 1..32.

---

## 7. Decision Required

Before proceeding with implementation, choose strategy:

**Option A**: Full rewrite (P1a + P1b + P2 + P3) — ~500 lines, reaches ~950 MB/s
- Pro: Maximum performance, clean architecture
- Con: ~2-3 sessions of work, higher risk

**Option B**: Incremental (P1a + P1b only) — ~280 lines, reaches ~610 MB/s
- Pro: Fast to implement, lower risk
- Con: Leaves 30-40% on the table (two-pass remains)

**Option C**: P1b + P2 (skip flat buffer, fuse decode+execute with existing RingBuffer)
- Pro: Gets the biggest single win (fused decode)
- Con: RingBuffer tax remains, more complex integration

**Recommendation**: Option A, phased. P1a+P1b first (validate with benchmarks), then P2, then P3.
