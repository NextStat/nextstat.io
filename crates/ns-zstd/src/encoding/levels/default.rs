use crate::{
    common::MAX_BLOCK_SIZE,
    encoding::{
        block_header::BlockHeader, blocks::compress_block, frame_compressor::CompressState, Matcher,
    },
};
use alloc::vec::Vec;

/// Placeholder for [`crate::encoding::CompressionLevel::Default`].
///
/// Uses a more expensive matcher (hash-chain + lazy matching) via `MatchGeneratorDriver::reset()`.
/// This function also avoids output inflation by falling back to raw blocks when compression
/// doesn't beat the uncompressed size.
#[inline]
pub fn compress_default<M: Matcher>(
    state: &mut CompressState<M>,
    last_block: bool,
    uncompressed_data: Vec<u8>,
    output: &mut Vec<u8>,
) {
    let block_size = uncompressed_data.len() as u32;

    // Same RLE fast-path as Fastest.
    if uncompressed_data.iter().all(|x| uncompressed_data[0].eq(x)) {
        let rle_byte = uncompressed_data[0];
        state.matcher.commit_space(uncompressed_data);
        state.matcher.skip_matching();
        let header = BlockHeader {
            last_block,
            block_type: crate::blocks::block::BlockType::RLE,
            block_size,
        };
        header.serialize(output);
        output.push(rle_byte);
        return;
    }

    // Compress as a standard compressed block.
    let mut compressed = core::mem::take(&mut state.tmp_block);
    compressed.clear();
    state.matcher.commit_space(uncompressed_data);
    compress_block(state, &mut compressed);

    let raw = state.matcher.get_last_space();
    // Avoid output inflation: if compressed is not smaller than raw, emit raw.
    // Also enforce the max allowable block size for compressed blocks.
    if compressed.len() >= raw.len() || compressed.len() >= MAX_BLOCK_SIZE as usize {
        let header = BlockHeader {
            last_block,
            block_type: crate::blocks::block::BlockType::Raw,
            block_size,
        };
        header.serialize(output);
        output.extend_from_slice(raw);
    } else {
        let header = BlockHeader {
            last_block,
            block_type: crate::blocks::block::BlockType::Compressed,
            block_size: compressed.len() as u32,
        };
        header.serialize(output);
        output.extend_from_slice(compressed.as_slice());
    }
    state.tmp_block = compressed;
}
