use alloc::vec::Vec;

use crate::{
    bit_io::BitWriter,
    encoding::frame_compressor::{CompressState, EncodedSequence},
    encoding::{Matcher, Sequence},
    fse::fse_encoder::{build_table_from_data, FSETable, State},
    huff0::huff0_encoder,
};

#[cfg(feature = "std")]
use std::time::Instant;

/// A block of [`crate::common::BlockType::Compressed`]
pub fn compress_block<M: Matcher>(state: &mut CompressState<M>, output: &mut Vec<u8>) {
    let out_start = output.len();
    let in_len = state.matcher.get_last_space().len();
    state.tmp_literals.clear();
    state.tmp_sequences.clear();
    #[cfg(feature = "std")]
    let t_match0 = state.perf.enabled().then(Instant::now);

    {
        // Split borrows: matcher drives sequence generation, while offset history evolves as we emit OF values.
        let literals_vec = &mut state.tmp_literals;
        let sequences = &mut state.tmp_sequences;
        let offset_hist = &mut state.offset_hist;
        let matcher = &mut state.matcher;

        matcher.start_matching(|seq| match seq {
            Sequence::Literals { literals } => literals_vec.extend_from_slice(literals),
            Sequence::Triple { literals, offset, match_len } => {
                literals_vec.extend_from_slice(literals);
                let ll = literals.len() as u32;
                let actual_offset = offset as u32;
                let of = encode_offset_with_history(actual_offset, ll, offset_hist);
                let ml = match_len as u32;

                // Pre-encode LL/ML/OF for table selection + sequence encoding hot paths.
                let (ll_code, ll_add_bits, ll_num_bits) = encode_literal_length(ll);
                let (ml_code, ml_add_bits, ml_num_bits) = encode_match_len(ml);
                let (of_code, of_add_bits, of_num_bits) = encode_offset(of);
                debug_assert!(ll_add_bits <= u16::MAX as u32);
                debug_assert!(ml_add_bits <= u16::MAX as u32);
                debug_assert!(ll_num_bits <= u8::MAX as usize);
                debug_assert!(ml_num_bits <= u8::MAX as usize);
                debug_assert!(of_num_bits <= u8::MAX as usize);

                sequences.push(EncodedSequence {
                    ll_add_bits: ll_add_bits as u16,
                    ml_add_bits: ml_add_bits as u16,
                    of_add_bits,
                    ll_code,
                    ml_code,
                    of_code,
                    ll_num_bits: ll_num_bits as u8,
                    ml_num_bits: ml_num_bits as u8,
                    of_num_bits: of_num_bits as u8,
                });
            }
        });
    }

    #[cfg(feature = "std")]
    if let Some(t0) = t_match0 {
        state.perf.add_match(t0.elapsed());
    }

    // literals section

    let mut writer = BitWriter::from(&mut *output);
    #[cfg(feature = "std")]
    let t_lit0 = state.perf.enabled().then(Instant::now);

    // Throughput-first: for small literal regions, Huffman setup dominates.
    // Keep a simple threshold (same spirit as libzstd) and fall back to raw literals.
    if state.tmp_literals.len() > 1024 {
        if let Some(table) =
            compress_literals(&state.tmp_literals, state.last_huff_table.as_ref(), &mut writer)
        {
            state.last_huff_table.replace(table);
        }
    } else {
        raw_literals(&state.tmp_literals, &mut writer);
    }

    #[cfg(feature = "std")]
    if let Some(t0) = t_lit0 {
        state.perf.add_literals(t0.elapsed());
    }

    // sequences section

    #[cfg(feature = "std")]
    let t_seq0 = state.perf.enabled().then(Instant::now);

    if state.tmp_sequences.is_empty() {
        writer.write_bits(0u8, 8);
    } else {
        encode_seqnum(state.tmp_sequences.len(), &mut writer);

        // Choose the tables
        // TODO store previously used tables
        let ll_mode = choose_table(
            state.fse_tables.ll_previous.as_ref(),
            &state.fse_tables.ll_default,
            state.tmp_sequences.iter().map(|seq| seq.ll_code),
            9,
        );
        let ml_mode = choose_table(
            state.fse_tables.ml_previous.as_ref(),
            &state.fse_tables.ml_default,
            state.tmp_sequences.iter().map(|seq| seq.ml_code),
            9,
        );
        let of_mode = choose_table(
            state.fse_tables.of_previous.as_ref(),
            &state.fse_tables.of_default,
            state.tmp_sequences.iter().map(|seq| seq.of_code),
            8,
        );

        writer.write_bits(encode_fse_table_modes(&ll_mode, &ml_mode, &of_mode), 8);

        encode_table(&ll_mode, &mut writer);
        encode_table(&of_mode, &mut writer);
        encode_table(&ml_mode, &mut writer);

        encode_sequences(
            &state.tmp_sequences,
            &mut writer,
            ll_mode.as_ref(),
            ml_mode.as_ref(),
            of_mode.as_ref(),
        );

        if let FseTableMode::Encoded(table) = ll_mode {
            state.fse_tables.ll_previous = Some(table)
        }
        if let FseTableMode::Encoded(table) = ml_mode {
            state.fse_tables.ml_previous = Some(table)
        }
        if let FseTableMode::Encoded(table) = of_mode {
            state.fse_tables.of_previous = Some(table)
        }
    }

    #[cfg(feature = "std")]
    if let Some(t0) = t_seq0 {
        state.perf.add_sequences(t0.elapsed());
    }

    writer.flush();

    #[cfg(feature = "std")]
    state.perf.on_block_end(
        in_len,
        output.len().saturating_sub(out_start),
        state.tmp_literals.len(),
        state.tmp_sequences.len(),
    );
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum FseTableMode<'a> {
    Predefined(&'a FSETable),
    Encoded(FSETable),
    RepeatLast(&'a FSETable),
}

impl FseTableMode<'_> {
    pub fn as_ref(&self) -> &FSETable {
        match self {
            Self::Predefined(t) => t,
            Self::RepeatLast(t) => t,
            Self::Encoded(t) => t,
        }
    }
}

fn choose_table<'a>(
    previous: Option<&'a FSETable>,
    default_table: &'a FSETable,
    codes: impl DoubleEndedIterator<Item = u8> + ExactSizeIterator + Clone,
    max_log: u8,
) -> FseTableMode<'a> {
    debug_assert!(codes.len() > 0);

    // Costs measured in bits for the FSE-coded symbol stream only.
    // (Extra bits for LL/ML/OF values are identical regardless of table choice.)
    let default_cost = estimate_fse_bits(default_table, codes.clone()).unwrap_or(usize::MAX);
    let repeat_cost =
        previous.and_then(|t| estimate_fse_bits(t, codes.clone())).unwrap_or(usize::MAX);

    // First pick the best of {repeat, predefined} without building a new table.
    let (mut best_mode, mut best_cost) = if let Some(prev) = previous {
        if repeat_cost <= default_cost {
            (FseTableMode::RepeatLast(prev), repeat_cost)
        } else {
            (FseTableMode::Predefined(default_table), default_cost)
        }
    } else {
        (FseTableMode::Predefined(default_table), default_cost)
    };

    // Heuristic: building a new table is expensive. Only try when:
    // - enough symbols (so there's something to gain)
    // - current coding is not already very tight
    //
    // This keeps Default fast while still allowing ratio improvements on "hard" blocks.
    let n = codes.len();
    if n < 256 {
        return best_mode;
    }
    let avg_bits = best_cost / n;
    if avg_bits <= 2 {
        return best_mode;
    }

    let new_table = build_table_from_data(codes.clone(), max_log, true);
    let new_cost = estimate_fse_bits(&new_table, codes.clone())
        .unwrap_or(usize::MAX)
        .saturating_add(fse_table_header_bits(&new_table));

    // Require a meaningful win to justify the header + CPU.
    // (Win threshold: at least 1/64th of the symbol stream bits.)
    let min_win = best_cost / 64;
    if new_cost.saturating_add(min_win) < best_cost {
        best_mode = FseTableMode::Encoded(new_table);
        best_cost = new_cost;
    }
    let _ = best_cost;
    best_mode
}

fn estimate_fse_bits(
    table: &FSETable,
    mut codes: impl DoubleEndedIterator<Item = u8>,
) -> Option<usize> {
    let last_code = codes.next_back()?;
    let mut state_idx = table.try_start_state(last_code)?.index;
    let mut bits = 0usize;

    for code in codes.rev() {
        let next = table.try_next_state(code, state_idx)?;
        bits += next.num_bits as usize;
        state_idx = next.index;
    }

    // Final state is written as a raw index using log2(table_size) bits.
    bits += table.table_size.ilog2() as usize;
    Some(bits)
}

fn fse_table_header_bits(table: &FSETable) -> usize {
    let mut w = BitWriter::new();
    table.write_table(&mut w);
    debug_assert!(w.index().is_multiple_of(8));
    w.index()
}

fn encode_table(mode: &FseTableMode<'_>, writer: &mut BitWriter<&mut Vec<u8>>) {
    match mode {
        FseTableMode::Predefined(_) => {}
        FseTableMode::RepeatLast(_) => {}
        FseTableMode::Encoded(table) => table.write_table(writer),
    }
}

fn encode_fse_table_modes(
    ll_mode: &FseTableMode<'_>,
    ml_mode: &FseTableMode<'_>,
    of_mode: &FseTableMode<'_>,
) -> u8 {
    fn mode_to_bits(mode: &FseTableMode<'_>) -> u8 {
        match mode {
            FseTableMode::Predefined(_) => 0,
            FseTableMode::RepeatLast(_) => 3,
            FseTableMode::Encoded(_) => 2,
        }
    }
    mode_to_bits(ll_mode) << 6 | mode_to_bits(of_mode) << 4 | mode_to_bits(ml_mode) << 2
}

fn encode_sequences(
    sequences: &[EncodedSequence],
    writer: &mut BitWriter<&mut Vec<u8>>,
    ll_table: &FSETable,
    ml_table: &FSETable,
    of_table: &FSETable,
) {
    let sequence = sequences[sequences.len() - 1];
    let ll_code = sequence.ll_code;
    let of_code = sequence.of_code;
    let ml_code = sequence.ml_code;
    let mut ll_state: &State = ll_table.start_state(ll_code);
    let mut ml_state: &State = ml_table.start_state(ml_code);
    let mut of_state: &State = of_table.start_state(of_code);

    writer.write_bits(sequence.ll_add_bits, sequence.ll_num_bits as usize);
    writer.write_bits(sequence.ml_add_bits, sequence.ml_num_bits as usize);
    writer.write_bits(sequence.of_add_bits, sequence.of_num_bits as usize);

    // encode backwards so the decoder reads the first sequence first
    if sequences.len() > 1 {
        for sequence in (0..=sequences.len() - 2).rev() {
            let sequence = sequences[sequence];
            let ll_code = sequence.ll_code;
            let of_code = sequence.of_code;
            let ml_code = sequence.ml_code;

            {
                let next = of_table.next_state(of_code, of_state.index);
                let diff = of_state.index - next.baseline;
                writer.write_bits(diff as u64, next.num_bits as usize);
                of_state = next;
            }
            {
                let next = ml_table.next_state(ml_code, ml_state.index);
                let diff = ml_state.index - next.baseline;
                writer.write_bits(diff as u64, next.num_bits as usize);
                ml_state = next;
            }
            {
                let next = ll_table.next_state(ll_code, ll_state.index);
                let diff = ll_state.index - next.baseline;
                writer.write_bits(diff as u64, next.num_bits as usize);
                ll_state = next;
            }

            writer.write_bits(sequence.ll_add_bits, sequence.ll_num_bits as usize);
            writer.write_bits(sequence.ml_add_bits, sequence.ml_num_bits as usize);
            writer.write_bits(sequence.of_add_bits, sequence.of_num_bits as usize);
        }
    }
    writer.write_bits(ml_state.index as u64, ml_table.table_size.ilog2() as usize);
    writer.write_bits(of_state.index as u64, of_table.table_size.ilog2() as usize);
    writer.write_bits(ll_state.index as u64, ll_table.table_size.ilog2() as usize);

    let bits_to_fill = writer.misaligned();
    if bits_to_fill == 0 {
        writer.write_bits(1u32, 8);
    } else {
        writer.write_bits(1u32, bits_to_fill);
    }
}

fn encode_seqnum(seqnum: usize, writer: &mut BitWriter<impl AsMut<Vec<u8>>>) {
    const UPPER_LIMIT: usize = 0xFFFF + 0x7F00;
    match seqnum {
        1..=127 => writer.write_bits(seqnum as u32, 8),
        128..=0x7FFF => {
            let upper = ((seqnum >> 8) | 0x80) as u8;
            let lower = seqnum as u8;
            writer.write_bits(upper, 8);
            writer.write_bits(lower, 8);
        }
        0x8000..=UPPER_LIMIT => {
            let encode = seqnum - 0x7F00;
            let upper = (encode >> 8) as u8;
            let lower = encode as u8;
            writer.write_bits(255u8, 8);
            writer.write_bits(upper, 8);
            writer.write_bits(lower, 8);
        }
        _ => unreachable!(),
    }
}

fn encode_literal_length(len: u32) -> (u8, u32, usize) {
    match len {
        0..=15 => (len as u8, 0, 0),
        16..=17 => (16, len - 16, 1),
        18..=19 => (17, len - 18, 1),
        20..=21 => (18, len - 20, 1),
        22..=23 => (19, len - 22, 1),
        24..=27 => (20, len - 24, 2),
        28..=31 => (21, len - 28, 2),
        32..=39 => (22, len - 32, 3),
        40..=47 => (23, len - 40, 3),
        48..=63 => (24, len - 48, 4),
        64..=127 => (25, len - 64, 6),
        128..=255 => (26, len - 128, 7),
        256..=511 => (27, len - 256, 8),
        512..=1023 => (28, len - 512, 9),
        1024..=2047 => (29, len - 1024, 10),
        2048..=4095 => (30, len - 2048, 11),
        4096..=8191 => (31, len - 4096, 12),
        8192..=16383 => (32, len - 8192, 13),
        16384..=32767 => (33, len - 16384, 14),
        32768..=65535 => (34, len - 32768, 15),
        65536..=131071 => (35, len - 65536, 16),
        131072.. => unreachable!(),
    }
}

fn encode_match_len(len: u32) -> (u8, u32, usize) {
    // Keep this in sync with decoder's ML_CODE_TABLE.
    const ML_BASE: [u32; 53] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515,
        1027, 2051, 4099, 8195, 16387, 32771, 65539,
    ];
    const ML_BITS: [u8; 53] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    ];

    const fn build_ml_max() -> [u32; 53] {
        let mut out = [0u32; 53];
        let mut i = 0usize;
        while i < 53 {
            let base = ML_BASE[i];
            let eb = ML_BITS[i] as u32;
            out[i] = if eb == 0 { base } else { base + ((1u32 << eb) - 1) };
            i += 1;
        }
        out
    }

    const ML_MAX: [u32; 53] = build_ml_max();

    debug_assert!((3..=131_074).contains(&len));
    // Manual binary search is consistently faster than `partition_point` here (tiny arrays, hot path).
    let mut lo = 0usize;
    let mut hi = ML_MAX.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if ML_MAX[mid] < len {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let code = lo;
    debug_assert!(code < ML_BASE.len());
    let base = ML_BASE[code];
    let eb = ML_BITS[code] as usize;
    (code as u8, len - base, eb)
}

fn encode_offset_with_history(actual_offset: u32, lit_len: u32, hist: &mut [u32; 3]) -> u32 {
    // Use zstd repeat-offset encoding (rep-codes 1..=3) when possible.
    // History update rules are subtle (ll==0 has special handling), so we reuse the decoder logic
    // as the source of truth for history evolution.
    let prev = *hist;
    let of_value = if lit_len > 0 {
        if actual_offset == prev[0] {
            1
        } else if actual_offset == prev[1] {
            2
        } else if actual_offset == prev[2] {
            3
        } else {
            actual_offset + 3
        }
    } else if actual_offset == prev[1] {
        1
    } else if actual_offset == prev[2] {
        2
    } else if prev[0] > 1 && actual_offset == prev[0] - 1 {
        3
    } else {
        actual_offset + 3
    };

    #[inline(always)]
    fn update_offset_history(of_value: u32, lit_len: u32, actual_offset: u32, hist: &mut [u32; 3]) {
        if lit_len > 0 {
            match of_value {
                1 => {}
                2 => {
                    hist[1] = hist[0];
                    hist[0] = actual_offset;
                }
                _ => {
                    hist[2] = hist[1];
                    hist[1] = hist[0];
                    hist[0] = actual_offset;
                }
            }
        } else {
            match of_value {
                1 => {
                    hist[1] = hist[0];
                    hist[0] = actual_offset;
                }
                2 => {
                    hist[2] = hist[1];
                    hist[1] = hist[0];
                    hist[0] = actual_offset;
                }
                _ => {
                    hist[2] = hist[1];
                    hist[1] = hist[0];
                    hist[0] = actual_offset;
                }
            }
        }
    }

    update_offset_history(of_value, lit_len, actual_offset, hist);

    #[cfg(debug_assertions)]
    {
        let mut check = prev;
        let decoded_actual =
            crate::decoding::sequence_execution::do_offset_history(of_value, lit_len, &mut check);
        debug_assert_eq!(
            decoded_actual, actual_offset,
            "offset history mismatch: ll={lit_len} wanted={actual_offset} got={decoded_actual} prev={prev:?} of_value={of_value}"
        );
        debug_assert_eq!(
            check, *hist,
            "offset history update mismatch: ll={lit_len} prev={prev:?} of_value={of_value}"
        );
    }
    of_value
}

fn encode_offset(len: u32) -> (u8, u32, usize) {
    let log = len.ilog2();
    let lower = len & ((1 << log) - 1);
    (log as u8, lower, log as usize)
}

fn raw_literals(literals: &[u8], writer: &mut BitWriter<&mut Vec<u8>>) {
    writer.write_bits(0u8, 2);
    writer.write_bits(0b11u8, 2);
    writer.write_bits(literals.len() as u32, 20);
    writer.append_bytes(literals);
}

fn compress_literals(
    literals: &[u8],
    last_table: Option<&huff0_encoder::HuffmanTable>,
    writer: &mut BitWriter<&mut Vec<u8>>,
) -> Option<huff0_encoder::HuffmanTable> {
    fn try_compress_with_table(
        literals: &[u8],
        encoder_table: &huff0_encoder::HuffmanTable,
        new_table: bool,
        writer: &mut BitWriter<&mut Vec<u8>>,
    ) -> bool {
        let reset_idx = writer.index();
        if new_table {
            writer.write_bits(2u8, 2); // compressed literals type
        } else {
            writer.write_bits(3u8, 2); // treeless compressed literals type
        }

        let (size_format, size_bits) = match literals.len() {
            0..6 => (0b00u8, 10),
            6..1024 => (0b01, 10),
            1024..16384 => (0b10, 14),
            16384..262144 => (0b11, 18),
            _ => unimplemented!("too many literals"),
        };

        writer.write_bits(size_format, 2);
        writer.write_bits(literals.len() as u32, size_bits);
        let size_index = writer.index();
        writer.write_bits(0u32, size_bits);
        let index_before = writer.index();
        let mut encoder = huff0_encoder::HuffmanEncoder::new(encoder_table, writer);
        if size_format == 0 {
            encoder.encode(literals, new_table)
        } else {
            encoder.encode4x(literals, new_table)
        };
        let encoded_len = (writer.index() - index_before) / 8;
        writer.change_bits(size_index, encoded_len as u64, size_bits);
        let total_len = (writer.index() - reset_idx) / 8;

        // If encoded len is bigger than the raw literals we are better off just writing raw literals.
        if total_len >= literals.len() {
            writer.reset_to(reset_idx);
            false
        } else {
            true
        }
    }

    // Fast path: if we have a previous Huffman table and it can encode this block's literals,
    // try treeless literals first (no table build / encode). If that isn't beneficial, fall back
    // to building a new table.
    if let Some(prev) = last_table {
        if prev.can_encode_data(literals) && try_compress_with_table(literals, prev, false, writer)
        {
            return None;
        }
    }

    let new_encoder_table = huff0_encoder::HuffmanTable::build_from_data(literals);
    if try_compress_with_table(literals, &new_encoder_table, true, writer) {
        Some(new_encoder_table)
    } else {
        raw_literals(literals, writer);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::encode_match_len;
    use super::{choose_table, FseTableMode};

    #[test]
    fn match_len_roundtrips_via_table() {
        // Validate that encode_match_len produces a code + add-bits that reconstructs the input len.
        // This specifically protects the large-length ranges where off-by-base errors are easy to make.
        const ML_CODE_TABLE: [(u32, u8); 53] = [
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (15, 0),
            (16, 0),
            (17, 0),
            (18, 0),
            (19, 0),
            (20, 0),
            (21, 0),
            (22, 0),
            (23, 0),
            (24, 0),
            (25, 0),
            (26, 0),
            (27, 0),
            (28, 0),
            (29, 0),
            (30, 0),
            (31, 0),
            (32, 0),
            (33, 0),
            (34, 0),
            (35, 1),
            (37, 1),
            (39, 1),
            (41, 1),
            (43, 2),
            (47, 2),
            (51, 3),
            (59, 3),
            (67, 4),
            (83, 4),
            (99, 5),
            (131, 7),
            (259, 8),
            (515, 9),
            (1027, 10),
            (2051, 11),
            (4099, 12),
            (8195, 13),
            (16387, 14),
            (32771, 15),
            (65539, 16),
        ];

        for len in 3u32..=131_074u32 {
            let (code, add, bits) = encode_match_len(len);
            let (base, extra_bits) = ML_CODE_TABLE[code as usize];
            assert_eq!(bits as u8, extra_bits, "len={len} code={code}");
            assert_eq!(base + add, len, "len={len} code={code} base={base} add={add}");
            if extra_bits == 0 {
                assert_eq!(add, 0, "len={len} code={code} should have no addbits");
            } else {
                assert!(add < (1u32 << extra_bits), "len={} code={} add too large", len, code);
            }
        }
    }

    #[test]
    fn choose_table_prefers_predefined_for_tiny_inputs() {
        let default = crate::fse::fse_encoder::default_ml_table();
        let mode = choose_table(None, &default, [0u8].iter().copied(), 9);
        assert!(matches!(mode, FseTableMode::Predefined(_)));
    }

    #[test]
    fn choose_table_prefers_repeat_on_ties() {
        let default = crate::fse::fse_encoder::default_of_table();
        let mode = choose_table(Some(&default), &default, [0u8, 0u8, 0u8].iter().copied(), 8);
        assert!(matches!(mode, FseTableMode::RepeatLast(_)));
    }
}
