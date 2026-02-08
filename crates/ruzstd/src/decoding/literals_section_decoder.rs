//! This module contains the decompress_literals function, used to take a
//! parsed literals header and a source and decompress it.

use super::super::blocks::literals_section::{LiteralsSection, LiteralsSectionType};
use super::scratch::HuffmanScratch;
use crate::bit_io::BitReaderReversed;
use crate::decoding::errors::DecompressLiteralsError;
use crate::huff0::HuffmanDecoder;
use alloc::vec::Vec;

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

    // Pre-size target buffer to avoid per-byte push overhead
    let regen = section.regenerated_size as usize;
    let start_len = target.len();
    target.resize(start_len + regen, 0);
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

        let threshold = -(scratch.table.max_num_bits as isize);

        // Process streams in pairs for ILP: (0,1) then (2,3).
        // Two independent Huffman state machines let the CPU pipeline
        // table lookups from one stream while the other's load completes.
        for pair in 0..2 {
            let si_a = pair * 2;
            let si_b = pair * 2 + 1;

            let mut dec_a = HuffmanDecoder::new(&scratch.table);
            let mut dec_b = HuffmanDecoder::new(&scratch.table);
            let mut br_a = BitReaderReversed::new(streams[si_a]);
            let mut br_b = BitReaderReversed::new(streams[si_b]);

            if let Err(e) = skip_padding(&mut br_a) {
                target.truncate(start_len);
                return Err(e);
            }
            if let Err(e) = skip_padding(&mut br_b) {
                target.truncate(start_len);
                return Err(e);
            }

            dec_a.init_state(&mut br_a);
            dec_b.init_state(&mut br_b);

            let mut w_a = starts[si_a];
            let mut w_b = starts[si_b];
            let end_a = starts[si_a] + sizes[si_a];
            let end_b = starts[si_b] + sizes[si_b];

            // Interleaved main loop: decode one symbol from each stream per iteration.
            // The two decode_symbol + next_state chains are independent — CPU can
            // execute both in parallel via out-of-order execution.
            while br_a.bits_remaining() > threshold && br_b.bits_remaining() > threshold {
                if w_a < end_a {
                    let sym = dec_a.decode_and_advance(&mut br_a);
                    unsafe { *out_ptr.add(w_a) = sym };
                } else {
                    let _ = dec_a.decode_and_advance(&mut br_a);
                }
                w_a += 1;

                if w_b < end_b {
                    let sym = dec_b.decode_and_advance(&mut br_b);
                    unsafe { *out_ptr.add(w_b) = sym };
                } else {
                    let _ = dec_b.decode_and_advance(&mut br_b);
                }
                w_b += 1;
            }

            // Drain whichever stream is longer
            while br_a.bits_remaining() > threshold {
                if w_a < end_a {
                    let sym = dec_a.decode_and_advance(&mut br_a);
                    unsafe { *out_ptr.add(w_a) = sym };
                } else {
                    let _ = dec_a.decode_and_advance(&mut br_a);
                }
                w_a += 1;
            }
            while br_b.bits_remaining() > threshold {
                if w_b < end_b {
                    let sym = dec_b.decode_and_advance(&mut br_b);
                    unsafe { *out_ptr.add(w_b) = sym };
                } else {
                    let _ = dec_b.decode_and_advance(&mut br_b);
                }
                w_b += 1;
            }

            if br_a.bits_remaining() != threshold {
                target.truncate(start_len);
                return Err(DecompressLiteralsError::BitstreamReadMismatch {
                    read_til: br_a.bits_remaining(),
                    expected: threshold,
                });
            }
            if br_b.bits_remaining() != threshold {
                target.truncate(start_len);
                return Err(DecompressLiteralsError::BitstreamReadMismatch {
                    read_til: br_b.bits_remaining(),
                    expected: threshold,
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
