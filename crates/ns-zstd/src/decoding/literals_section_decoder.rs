//! This module contains the decompress_literals function, used to take a
//! parsed literals header and a source and decompress it.

use super::super::blocks::literals_section::{LiteralsSection, LiteralsSectionType};
use super::scratch::HuffmanScratch;
use crate::bit_io::BitReaderReversed;
use crate::decoding::errors::DecompressLiteralsError;
use crate::huff0::{Entry, HuffmanDecoder};
use alloc::vec::Vec;

/// Peek+skip Huffman decode (matches libzstd's HUF_decodeSymbolX1):
/// peek max_bits WITHOUT consuming → table lookup → consume only actual num_bits.
/// Eliminates 3 ops/symbol (shift, mask, or) vs the state-based approach.
#[inline(always)]
fn peek_skip_decode(
    br: &mut BitReaderReversed<'_>,
    max_bits: u8,
    table_ptr: *const Entry,
    table_mask: usize,
) -> u8 {
    let val = br.peek_bits_refill(max_bits) as usize & table_mask;
    let entry = unsafe { *table_ptr.add(val) };
    br.consume(entry.num_bits);
    entry.symbol
}

/// Like peek_skip_decode but WITHOUT refill check.
/// Caller must guarantee at least max_bits are available in the container.
/// Used inside batched-refill blocks (5 symbols per refill for max_bits≤11).
#[inline(always)]
fn peek_skip_decode_norefill(
    br: &mut BitReaderReversed<'_>,
    max_bits: u8,
    table_ptr: *const Entry,
    table_mask: usize,
) -> u8 {
    let val = br.peek_bits(max_bits) as usize & table_mask;
    let entry = unsafe { *table_ptr.add(val) };
    br.consume(entry.num_bits);
    entry.symbol
}

/// Decode and decompress the provided literals section into `target`, returning the number of bytes read.
pub fn decode_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, DecompressLiteralsError> {
    match section.ls_type {
        LiteralsSectionType::Raw => {
            target.extend(&source[0..section.regenerated_size as usize]);
            Ok(section.regenerated_size)
        }
        LiteralsSectionType::RLE => {
            target.resize(target.len() + section.regenerated_size as usize, source[0]);
            Ok(1)
        }
        LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => {
            let bytes_read = decompress_literals(section, scratch, source, target)?;

            //return sum of used bytes
            Ok(bytes_read)
        }
    }
}

/// Decompress the provided literals section and source into the provided `target`.
/// This function is used when the literals section is `Compressed` or `Treeless`
///
/// Returns the number of bytes read.
fn decompress_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, DecompressLiteralsError> {
    use DecompressLiteralsError as err;

    let compressed_size = section.compressed_size.ok_or(err::MissingCompressedSize)? as usize;
    let num_streams = section.num_streams.ok_or(err::MissingNumStreams)?;

    target.reserve(section.regenerated_size as usize);
    let source = &source[0..compressed_size];
    let mut bytes_read = 0;

    match section.ls_type {
        LiteralsSectionType::Compressed => {
            //read Huffman tree description
            bytes_read += scratch.table.build_decoder(source)?;
            vprintln!("Built huffman table using {} bytes", bytes_read);
        }
        LiteralsSectionType::Treeless => {
            if scratch.table.max_num_bits == 0 {
                return Err(err::UninitializedHuffmanTable);
            }
        }
        _ => { /* nothing to do, huffman tree has been provided by previous block */ }
    }

    let source = &source[bytes_read as usize..];

    // Pre-size target buffer to avoid per-byte push overhead.
    // Use reserve + set_len instead of resize to skip the zero-fill;
    // all bytes up to write_idx will be overwritten, and we truncate at the end.
    let regen = section.regenerated_size as usize;
    let start_len = target.len();
    target.reserve(regen);
    unsafe { target.set_len(start_len + regen) };
    // Use raw pointer to avoid holding a mutable borrow on `target`,
    // which would conflict with `target.truncate()` in error paths.
    let out_ptr = target[start_len..].as_mut_ptr();
    let mut write_idx = 0;

    if num_streams == 4 {
        if source.len() < 6 {
            target.truncate(start_len);
            return Err(err::MissingBytesForJumpHeader { got: source.len() });
        }
        let jump1 = source[0] as usize + ((source[1] as usize) << 8);
        let jump2 = jump1 + source[2] as usize + ((source[3] as usize) << 8);
        let jump3 = jump2 + source[4] as usize + ((source[5] as usize) << 8);
        bytes_read += 6;
        let source = &source[6..];

        if source.len() < jump3 {
            target.truncate(start_len);
            return Err(err::MissingBytesForLiterals {
                got: source.len(),
                needed: jump3,
            });
        }

        let streams: [&[u8]; 4] = [
            &source[..jump1],
            &source[jump1..jump2],
            &source[jump2..jump3],
            &source[jump3..],
        ];

        // Per-stream output sizes (zstd spec: first 3 = (regen+3)/4, 4th = remainder)
        let seg = (regen + 3) / 4;
        let sizes = [seg, seg, seg, regen - 3 * seg];
        let starts = [0usize, seg, 2 * seg, 3 * seg];

        // Peek+skip Huffman decode (matches libzstd's approach):
        // - NO init_state (peek implicitly sees the initial bits)
        // - Each symbol: refill if needed → peek max_num_bits → table lookup → consume num_bits
        // - This eliminates 3 ops/symbol (shift, mask, or) from the state-based approach
        let max_bits = scratch.table.max_num_bits;
        let table_ptr = scratch.table.decode.as_ptr();
        let table_mask = scratch.table.decode.len() - 1;

        let mut br = [
            BitReaderReversed::new(streams[0]),
            BitReaderReversed::new(streams[1]),
            BitReaderReversed::new(streams[2]),
            BitReaderReversed::new(streams[3]),
        ];

        for i in 0..4 {
            if let Err(e) = skip_padding(&mut br[i]) {
                target.truncate(start_len);
                return Err(e);
            }
        }

        // Batched-refill decode: after one refill() we have ≥56 bits (bits_consumed < 8).
        // Safe batch size = floor(57 / max_bits): worst-case K symbols consume
        // 7 + K*max_bits bits, need 7 + K*max_bits ≤ 64 → K ≤ 57/max_bits.
        // max_bits=11→5, =9→6, =8→7, =7→8, =6→9
        let batch = if max_bits > 0 { 57 / max_bits as usize } else { 1 };
        let min_count = sizes[3]; // 4th stream is smallest or equal

        let mut w0 = starts[0];
        let mut w1 = starts[1];
        let mut w2 = starts[2];
        let mut w3 = starts[3];

        // Phase 1: all 4 streams active, batched refill.
        let full_batches = min_count / batch;
        let remainder4 = min_count % batch;

        for _ in 0..full_batches {
            // Single refill per stream per batch
            br[0].refill();
            br[1].refill();
            br[2].refill();
            br[3].refill();

            for _ in 0..batch {
                let sym0 = peek_skip_decode_norefill(&mut br[0], max_bits, table_ptr, table_mask);
                let sym1 = peek_skip_decode_norefill(&mut br[1], max_bits, table_ptr, table_mask);
                let sym2 = peek_skip_decode_norefill(&mut br[2], max_bits, table_ptr, table_mask);
                let sym3 = peek_skip_decode_norefill(&mut br[3], max_bits, table_ptr, table_mask);
                unsafe {
                    *out_ptr.add(w0) = sym0;
                    *out_ptr.add(w1) = sym1;
                    *out_ptr.add(w2) = sym2;
                    *out_ptr.add(w3) = sym3;
                }
                w0 += 1;
                w1 += 1;
                w2 += 1;
                w3 += 1;
            }
        }

        // Remainder for all 4 streams (with per-symbol refill)
        for _ in 0..remainder4 {
            let sym0 = peek_skip_decode(&mut br[0], max_bits, table_ptr, table_mask);
            let sym1 = peek_skip_decode(&mut br[1], max_bits, table_ptr, table_mask);
            let sym2 = peek_skip_decode(&mut br[2], max_bits, table_ptr, table_mask);
            let sym3 = peek_skip_decode(&mut br[3], max_bits, table_ptr, table_mask);
            unsafe {
                *out_ptr.add(w0) = sym0;
                *out_ptr.add(w1) = sym1;
                *out_ptr.add(w2) = sym2;
                *out_ptr.add(w3) = sym3;
            }
            w0 += 1;
            w1 += 1;
            w2 += 1;
            w3 += 1;
        }

        // Phase 2: drain remaining symbols for streams 0-2 (they may have more).
        let extra = sizes[0] - min_count;
        let full_batches3 = extra / batch;
        let remainder3 = extra % batch;

        for _ in 0..full_batches3 {
            br[0].refill();
            br[1].refill();
            br[2].refill();

            for _ in 0..batch {
                let sym0 = peek_skip_decode_norefill(&mut br[0], max_bits, table_ptr, table_mask);
                let sym1 = peek_skip_decode_norefill(&mut br[1], max_bits, table_ptr, table_mask);
                let sym2 = peek_skip_decode_norefill(&mut br[2], max_bits, table_ptr, table_mask);
                unsafe {
                    *out_ptr.add(w0) = sym0;
                    *out_ptr.add(w1) = sym1;
                    *out_ptr.add(w2) = sym2;
                }
                w0 += 1;
                w1 += 1;
                w2 += 1;
            }
        }

        for _ in 0..remainder3 {
            let sym0 = peek_skip_decode(&mut br[0], max_bits, table_ptr, table_mask);
            let sym1 = peek_skip_decode(&mut br[1], max_bits, table_ptr, table_mask);
            let sym2 = peek_skip_decode(&mut br[2], max_bits, table_ptr, table_mask);
            unsafe {
                *out_ptr.add(w0) = sym0;
                *out_ptr.add(w1) = sym1;
                *out_ptr.add(w2) = sym2;
            }
            w0 += 1;
            w1 += 1;
            w2 += 1;
        }

        // Phase 3: verify bitstream alignment — with peek+skip (no init_state),
        // all bits should be exactly consumed at the end (bits_remaining == 0).
        for i in 0..4 {
            if br[i].bits_remaining() != 0 {
                target.truncate(start_len);
                return Err(DecompressLiteralsError::BitstreamReadMismatch {
                    read_til: br[i].bits_remaining(),
                    expected: 0,
                });
            }
        }

        write_idx = regen;
        bytes_read += source.len() as u32;
    } else {
        //just decode the one stream
        debug_assert!(num_streams == 1);
        let mut decoder = HuffmanDecoder::new(&scratch.table);
        let mut br = BitReaderReversed::new(source);
        let mut skipped_bits = 0;
        loop {
            let val = br.get_bits(1);
            skipped_bits += 1;
            if val == 1 || skipped_bits > 8 {
                break;
            }
        }
        if skipped_bits > 8 {
            target.truncate(start_len);
            return Err(DecompressLiteralsError::ExtraPadding { skipped_bits });
        }
        decoder.init_state(&mut br);
        let threshold = -(scratch.table.max_num_bits as isize);
        while br.bits_remaining() > threshold {
            if write_idx < regen {
                let sym = decoder.decode_and_advance(&mut br);
                unsafe { *out_ptr.add(write_idx) = sym };
            } else {
                let _ = decoder.decode_and_advance(&mut br);
            }
            write_idx += 1;
        }
        bytes_read += source.len() as u32;
    }

    // Truncate to actual decoded length (may differ from pre-allocated)
    target.truncate(start_len + write_idx);

    if target.len() != section.regenerated_size as usize {
        return Err(DecompressLiteralsError::DecodedLiteralCountMismatch {
            decoded: target.len(),
            expected: section.regenerated_size as usize,
        });
    }

    Ok(bytes_read)
}

/// Skip the zero-padding and sentinel 1-bit at the end of a Huffman bitstream.
#[inline(always)]
fn skip_padding(br: &mut BitReaderReversed<'_>) -> Result<(), DecompressLiteralsError> {
    let mut skipped_bits = 0;
    loop {
        let val = br.get_bits(1);
        skipped_bits += 1;
        if val == 1 || skipped_bits > 8 {
            break;
        }
    }
    if skipped_bits > 8 {
        return Err(DecompressLiteralsError::ExtraPadding { skipped_bits });
    }
    Ok(())
}
