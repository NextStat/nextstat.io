use crate::decoding::errors::{DecodeBufferError, ExecuteSequencesError};
use crate::decoding::flat_decode_buffer::fast_copy_match;
use crate::decoding::fused::FusedOutputBuffer;

pub(crate) struct SliceOutputBuffer<'a> {
    out: &'a mut [u8],
    pos: usize,
    dict_ptr: *const u8,
    dict_len: usize,
}

impl<'a> SliceOutputBuffer<'a> {
    pub(crate) fn new(out: &'a mut [u8], dict_ptr: *const u8, dict_len: usize) -> Self {
        Self { out, pos: 0, dict_ptr, dict_len }
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
    fn ensure_capacity_inner(&self, extra: usize) -> Result<(), ExecuteSequencesError> {
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

    #[allow(dead_code)]
    fn repeat_from_dict_slow(
        &mut self,
        offset: usize,
        match_length: usize,
    ) -> Result<(), ExecuteSequencesError> {
        let dict_len = self.dict_len;
        if dict_len == 0 {
            return Err(ExecuteSequencesError::DecodebufferError(
                DecodeBufferError::NotEnoughBytesInDictionary {
                    got: 0,
                    need: offset.saturating_sub(self.pos),
                },
            ));
        }

        self.ensure_capacity_inner(match_length)?;

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
                unsafe { *self.dict_ptr.add(di as usize) }
            };
            self.out[d] = byte;
        }

        self.pos += match_length;
        Ok(())
    }

    #[inline(always)]
    unsafe fn push_unchecked_inner(&mut self, data: &[u8]) {
        debug_assert!(self.pos + data.len() <= self.out.len());
        core::ptr::copy_nonoverlapping(
            data.as_ptr(),
            self.out.as_mut_ptr().add(self.pos),
            data.len(),
        );
        self.pos += data.len();
    }

    #[inline(always)]
    unsafe fn repeat_unchecked_inner(&mut self, offset: usize, match_length: usize) {
        debug_assert!(offset > 0);
        debug_assert!(offset <= self.pos);
        debug_assert!(self.pos + match_length <= self.out.len());

        let base = self.out.as_mut_ptr();
        fast_copy_match(base.add(self.pos - offset), base.add(self.pos), offset, match_length);
        self.pos += match_length;
    }

    #[inline(always)]
    fn repeat_from_dict_checked(
        &mut self,
        offset: usize,
        match_length: usize,
    ) -> Result<(), ExecuteSequencesError> {
        let dict_len = self.dict_len;
        if dict_len == 0 {
            return Err(ExecuteSequencesError::DecodebufferError(
                DecodeBufferError::NotEnoughBytesInDictionary {
                    got: 0,
                    need: offset.saturating_sub(self.pos),
                },
            ));
        }

        self.ensure_capacity_inner(match_length)?;

        let bytes_from_dict = offset - self.pos;
        if bytes_from_dict > dict_len {
            return Err(ExecuteSequencesError::DecodebufferError(
                DecodeBufferError::NotEnoughBytesInDictionary {
                    got: dict_len,
                    need: bytes_from_dict,
                },
            ));
        }

        unsafe {
            let dst = self.out.as_mut_ptr().add(self.pos);
            let dict_src = self.dict_ptr.add(dict_len - bytes_from_dict);

            if bytes_from_dict >= match_length {
                core::ptr::copy_nonoverlapping(dict_src, dst, match_length);
                self.pos += match_length;
                return Ok(());
            }

            core::ptr::copy_nonoverlapping(dict_src, dst, bytes_from_dict);
            self.pos += bytes_from_dict;

            let remaining = match_length - bytes_from_dict;
            self.repeat_unchecked_inner(offset, remaining);
        }

        Ok(())
    }
}

impl FusedOutputBuffer for SliceOutputBuffer<'_> {
    #[inline(always)]
    fn ensure_capacity(&mut self, extra: usize) -> Result<(), ExecuteSequencesError> {
        self.ensure_capacity_inner(extra)
    }

    #[inline(always)]
    fn push(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError> {
        self.ensure_capacity_inner(data.len())?;
        unsafe { self.push_unchecked_inner(data) };
        Ok(())
    }

    #[inline(always)]
    fn push_unchecked(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError> {
        unsafe { self.push_unchecked_inner(data) };
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
        if offset > self.pos {
            return self.repeat_from_dict_checked(offset, match_length);
        }

        self.ensure_capacity_inner(match_length)?;
        unsafe { self.repeat_unchecked_inner(offset, match_length) };
        Ok(())
    }

    #[inline(always)]
    fn repeat_unchecked(
        &mut self,
        offset: usize,
        match_length: usize,
    ) -> Result<(), ExecuteSequencesError> {
        if offset > self.pos {
            return self.repeat_from_dict_checked(offset, match_length);
        }
        unsafe { self.repeat_unchecked_inner(offset, match_length) };
        Ok(())
    }
}

/// Output buffer backed by a raw pointer + length.
///
/// Used to decode directly into a `Vec<u8>`'s spare capacity without first
/// initializing it (avoids `resize(..., 0)` memset).
pub(crate) struct PtrOutputBuffer {
    out_ptr: *mut u8,
    out_len: usize,
    pos: usize,
    dict_ptr: *const u8,
    dict_len: usize,
}

impl PtrOutputBuffer {
    pub(crate) fn new(
        out_ptr: *mut u8,
        out_len: usize,
        dict_ptr: *const u8,
        dict_len: usize,
    ) -> Self {
        Self { out_ptr, out_len, pos: 0, dict_ptr, dict_len }
    }

    #[inline(always)]
    pub(crate) fn bytes_written(&self) -> usize {
        self.pos
    }

    #[inline(always)]
    fn ensure_capacity_inner(&self, extra: usize) -> Result<(), ExecuteSequencesError> {
        let need = self.pos.saturating_add(extra);
        if need <= self.out_len {
            Ok(())
        } else {
            Err(ExecuteSequencesError::NotEnoughBytesForSequence {
                wanted: need,
                have: self.out_len,
            })
        }
    }

    #[inline(always)]
    unsafe fn push_unchecked_inner(&mut self, data: &[u8]) {
        debug_assert!(self.pos + data.len() <= self.out_len);
        core::ptr::copy_nonoverlapping(data.as_ptr(), self.out_ptr.add(self.pos), data.len());
        self.pos += data.len();
    }

    #[inline(always)]
    unsafe fn repeat_unchecked_inner(&mut self, offset: usize, match_length: usize) {
        debug_assert!(offset > 0);
        debug_assert!(offset <= self.pos);
        debug_assert!(self.pos + match_length <= self.out_len);

        let base = self.out_ptr;
        fast_copy_match(base.add(self.pos - offset), base.add(self.pos), offset, match_length);
        self.pos += match_length;
    }

    #[inline(always)]
    fn repeat_from_dict_checked(
        &mut self,
        offset: usize,
        match_length: usize,
    ) -> Result<(), ExecuteSequencesError> {
        let dict_len = self.dict_len;
        if dict_len == 0 {
            return Err(ExecuteSequencesError::DecodebufferError(
                DecodeBufferError::NotEnoughBytesInDictionary {
                    got: 0,
                    need: offset.saturating_sub(self.pos),
                },
            ));
        }

        self.ensure_capacity_inner(match_length)?;

        let bytes_from_dict = offset - self.pos;
        if bytes_from_dict > dict_len {
            return Err(ExecuteSequencesError::DecodebufferError(
                DecodeBufferError::NotEnoughBytesInDictionary {
                    got: dict_len,
                    need: bytes_from_dict,
                },
            ));
        }

        unsafe {
            let dst = self.out_ptr.add(self.pos);
            let dict_src = self.dict_ptr.add(dict_len - bytes_from_dict);

            if bytes_from_dict >= match_length {
                core::ptr::copy_nonoverlapping(dict_src, dst, match_length);
                self.pos += match_length;
                return Ok(());
            }

            core::ptr::copy_nonoverlapping(dict_src, dst, bytes_from_dict);
            self.pos += bytes_from_dict;

            let remaining = match_length - bytes_from_dict;
            self.repeat_unchecked_inner(offset, remaining);
        }

        Ok(())
    }
}

impl FusedOutputBuffer for PtrOutputBuffer {
    #[inline(always)]
    fn ensure_capacity(&mut self, extra: usize) -> Result<(), ExecuteSequencesError> {
        self.ensure_capacity_inner(extra)
    }

    #[inline(always)]
    fn push(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError> {
        self.ensure_capacity_inner(data.len())?;
        unsafe { self.push_unchecked_inner(data) };
        Ok(())
    }

    #[inline(always)]
    fn push_unchecked(&mut self, data: &[u8]) -> Result<(), ExecuteSequencesError> {
        unsafe { self.push_unchecked_inner(data) };
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
        if offset > self.pos {
            return self.repeat_from_dict_checked(offset, match_length);
        }

        self.ensure_capacity_inner(match_length)?;
        unsafe { self.repeat_unchecked_inner(offset, match_length) };
        Ok(())
    }

    #[inline(always)]
    fn repeat_unchecked(
        &mut self,
        offset: usize,
        match_length: usize,
    ) -> Result<(), ExecuteSequencesError> {
        if offset > self.pos {
            return self.repeat_from_dict_checked(offset, match_length);
        }
        unsafe { self.repeat_unchecked_inner(offset, match_length) };
        Ok(())
    }
}
