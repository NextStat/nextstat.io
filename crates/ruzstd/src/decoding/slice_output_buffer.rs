use crate::decoding::errors::{DecodeBufferError, ExecuteSequencesError};
use crate::decoding::flat_decode_buffer::fast_copy_match;
use crate::decoding::fused::FusedOutputBuffer;

pub(crate) struct SliceOutputBuffer<'a> {
    out: &'a mut [u8],
    pos: usize,
    dict: &'a [u8],
}

impl<'a> SliceOutputBuffer<'a> {
    pub(crate) fn new(out: &'a mut [u8], dict: &'a [u8]) -> Self {
        Self { out, pos: 0, dict }
    }

    #[inline(always)]
    pub(crate) fn bytes_written(&self) -> usize {
        self.pos
    }

    #[inline(always)]
    pub(crate) fn remaining_mut(&mut self) -> &mut [u8] {
        &mut self.out[self.pos..]
    }

    #[inline(always)]
    pub(crate) fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    #[inline(always)]
    fn ensure_capacity(&self, extra: usize) -> Result<(), ExecuteSequencesError> {
        let need = self.pos.saturating_add(extra);
        if need <= self.out.len() {
            Ok(())
        } else {
            Err(ExecuteSequencesError::NotEnoughBytesForSequence {
                wanted: need,
                have: self.out.len(),
            })
        }
    }

    fn repeat_from_dict_slow(
        &mut self,
        offset: usize,
        match_length: usize,
    ) -> Result<(), ExecuteSequencesError> {
        let dict_len = self.dict.len();
        if dict_len == 0 {
            return Err(ExecuteSequencesError::DecodebufferError(
                DecodeBufferError::NotEnoughBytesInDictionary {
                    got: 0,
                    need: offset.saturating_sub(self.pos),
                },
            ));
        }

        self.ensure_capacity(match_length)?;

        for i in 0..match_length {
            let d = self.pos + i;
            let src = d as isize - offset as isize;
            let byte = if src >= 0 {
                self.out[src as usize]
            } else {
                let di = dict_len as isize + src;
                if di < 0 {
                    return Err(ExecuteSequencesError::DecodebufferError(
                        DecodeBufferError::NotEnoughBytesInDictionary {
                            got: dict_len,
                            need: (-src) as usize,
                        },
                    ));
                }
                self.dict[di as usize]
            };
            self.out[d] = byte;
        }

        self.pos += match_length;
        Ok(())
    }
}

impl FusedOutputBuffer for SliceOutputBuffer<'_> {
    #[inline(always)]
    fn push(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError> {
        self.ensure_capacity(data.len())?;
        let end = self.pos + data.len();
        self.out[self.pos..end].copy_from_slice(data);
        self.pos = end;
        Ok(())
    }

    #[inline(always)]
    fn repeat(&mut self, offset: usize, match_length: usize) -> Result<(), ExecuteSequencesError> {
        if offset == 0 {
            return Err(ExecuteSequencesError::ZeroOffset);
        }
        if match_length == 0 {
            return Ok(());
        }
        self.ensure_capacity(match_length)?;

        if offset > self.pos {
            return self.repeat_from_dict_slow(offset, match_length);
        }

        unsafe {
            let base = self.out.as_mut_ptr();
            fast_copy_match(
                base.add(self.pos - offset),
                base.add(self.pos),
                offset,
                match_length,
            );
        }
        self.pos += match_length;
        Ok(())
    }
}
