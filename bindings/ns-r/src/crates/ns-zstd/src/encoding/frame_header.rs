//! Utilities and representations for a frame header.
use crate::bit_io::BitWriter;
use crate::common::{MAGIC_NUM, MAX_WINDOW_SIZE, MIN_WINDOW_SIZE};
use crate::encoding::util::{find_min_size, minify_val};
use alloc::vec::Vec;

/// A header for a single Zstandard frame.
///
/// <https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frame_header>
#[derive(Debug)]
pub struct FrameHeader {
    /// Optionally, the original (uncompressed) size of the data within the frame in bytes.
    /// If not present, `window_size` must be set.
    pub frame_content_size: Option<u64>,
    /// If set to true, data must be regenerated within a single
    /// continuous memory segment.
    pub single_segment: bool,
    /// If set to true, a 32 bit content checksum will be present
    /// at the end of the frame.
    pub content_checksum: bool,
    /// If a dictionary ID is provided, the ID of that dictionary.
    pub dictionary_id: Option<u64>,
    /// The minimum memory buffer required to compress a frame. If not present,
    /// `single_segment` will be set to true. If present, this value must be greater than 1KB
    /// and less than 3.75TB. Encoders should not generate a frame that requires a window size larger than
    /// 8mb.
    pub window_size: Option<u64>,
}

impl FrameHeader {
    /// Writes the serialized frame header into the provided buffer.
    ///
    /// The returned header *does include* a frame header descriptor.
    pub fn serialize(self, output: &mut Vec<u8>) {
        vprintln!("Serializing frame with header: {self:?}");
        // https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frame_header
        // Magic Number:
        output.extend_from_slice(&MAGIC_NUM.to_le_bytes());

        // `Frame_Header_Descriptor`:
        output.push(self.descriptor());

        // `Window_Descriptor
        if !self.single_segment {
            if let Some(window_size) = self.window_size {
                output.push(encode_window_descriptor(window_size));
            }
        }

        if let Some(id) = self.dictionary_id {
            output.extend(minify_val(id));
        }

        if let Some(frame_content_size) = self.frame_content_size {
            output.extend(minify_val_fcs(frame_content_size));
        }
    }

    /// Generate a serialized frame header descriptor for the frame header.
    ///
    /// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frame_header_descriptor
    fn descriptor(&self) -> u8 {
        let mut bw = BitWriter::new();
        // A frame header starts with a frame header descriptor.
        // It describes what other fields are present
        // https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frame_header_descriptor
        // Writing the frame header descriptor:
        // `Frame_Content_Size_flag`:
        // The Frame_Content_Size_flag specifies if
        // the Frame_Content_Size field is provided within the header.
        // TODO: The Frame_Content_Size field isn't set at all, we should prefer to include it always.
        // If the `Single_Segment_flag` is set and this value is zero,
        // the size of the FCS field is 1 byte.
        // Otherwise, the FCS field is omitted.
        // | Value | Size of field (Bytes)
        // | 0     | 0 or 1
        // | 1     | 2
        // | 2     | 4
        // | 3     | 8

        // `Dictionary_ID_flag`:
        if let Some(id) = self.dictionary_id {
            let flag_value: u8 = match find_min_size(id) {
                0 => 0,
                1 => 1,
                2 => 2,
                4 => 3,
                _ => panic!(),
            };
            bw.write_bits(flag_value, 2);
        } else {
            // A `Dictionary_ID` was not provided
            bw.write_bits(0u8, 2);
        }

        // `Content_Checksum_flag`:
        if self.content_checksum {
            bw.write_bits(1u8, 1);
        } else {
            bw.write_bits(0u8, 1);
        }

        // `Reserved_bit`:
        // This value must be zero
        bw.write_bits(0u8, 1);

        // `Unused_bit`:
        // An encoder compliant with this spec must set this bit to zero
        bw.write_bits(0u8, 1);

        // `Single_Segment_flag`:
        // If this flag is set, data must be regenerated within a single continuous memory segment,
        // and the `Frame_Content_Size` field must be present in the header.
        // If this flag is not set, the `Window_Descriptor` field must be present in the frame header.
        if self.single_segment {
            assert!(self.frame_content_size.is_some(), "if the `single_segment` flag is set to true, then a frame content size must be provided");
            bw.write_bits(1u8, 1);
        } else {
            assert!(
                self.window_size.is_some(),
                "if the `single_segment` flag is set to false, then a window size must be provided"
            );
            bw.write_bits(0u8, 1);
        }

        if let Some(frame_content_size) = self.frame_content_size {
            let field_size = find_min_size(frame_content_size);
            let flag_value: u8 = match field_size {
                1 => 0,
                2 => 1,
                4 => 2,
                8 => 3,
                _ => panic!("invalid fcs field size {}", field_size),
            };

            bw.write_bits(flag_value, 2);
        } else {
            // `Frame_Content_Size` was not provided
            bw.write_bits(0u8, 2);
        }

        bw.dump()[0]
    }
}

/// Encode a `Window_Descriptor` byte (Exponent+Mantissa) for a requested window size.
///
/// Decoder formula (see `decoding/frame.rs`):
/// - `windowLog = 10 + Exponent`
/// - `windowBase = 1 << windowLog`
/// - `windowAdd = (windowBase / 8) * Mantissa`
/// - `Window_Size = windowBase + windowAdd`
fn encode_window_descriptor(window_size: u64) -> u8 {
    assert!(
        (MIN_WINDOW_SIZE..=MAX_WINDOW_SIZE).contains(&window_size),
        "window_size out of spec bounds: {}",
        window_size
    );

    // Pick the smallest representable window >= requested.
    // There are only 32*8 possible descriptors, so brute-force is fine and avoids off-by-one math.
    let mut best: Option<(u64, u8)> = None; // (decoded window_size, descriptor)
    for exp in 0u8..=31u8 {
        let window_log = 10u32 + exp as u32;
        let window_base = 1u64 << window_log;
        let step = window_base / 8;
        for mantissa in 0u8..=7u8 {
            let decoded = window_base + step * u64::from(mantissa);
            if decoded < window_size {
                continue;
            }
            if decoded > MAX_WINDOW_SIZE {
                continue;
            }
            let desc = (exp << 3) | mantissa;
            match best {
                None => best = Some((decoded, desc)),
                Some((best_decoded, _)) if decoded < best_decoded => best = Some((decoded, desc)),
                _ => {}
            }
        }
    }

    best.expect("window_size must be representable").1
}

/// Identical to [`minify_val`], but it implements the following edge case:
///
/// > When FCS_Field_Size is 1, 4 or 8 bytes, the value is read directly. When FCS_Field_Size is 2, the offset of 256 is added.
///
/// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#frame_content_size
fn minify_val_fcs(val: u64) -> Vec<u8> {
    let new_size = find_min_size(val);
    let mut val = val;
    if new_size == 2 {
        val -= 256;
    }
    val.to_le_bytes()[0..new_size].to_vec()
}

#[cfg(test)]
mod tests {
    use super::FrameHeader;
    use crate::decoding::frame::{read_frame_header, FrameDescriptor};
    use alloc::vec::Vec;

    #[test]
    fn frame_header_descriptor_decode() {
        let header = FrameHeader {
            frame_content_size: Some(1),
            single_segment: true,
            content_checksum: false,
            dictionary_id: None,
            window_size: None,
        };
        let descriptor = header.descriptor();
        let decoded_descriptor = FrameDescriptor(descriptor);
        assert_eq!(decoded_descriptor.frame_content_size_bytes().unwrap(), 1);
        assert!(!decoded_descriptor.content_checksum_flag());
        assert_eq!(decoded_descriptor.dictionary_id_bytes().unwrap(), 0);
    }

    #[test]
    fn frame_header_decode() {
        let header = FrameHeader {
            frame_content_size: Some(1),
            single_segment: true,
            content_checksum: false,
            dictionary_id: None,
            window_size: None,
        };

        let mut serialized_header = Vec::new();
        header.serialize(&mut serialized_header);
        let parsed_header = read_frame_header(serialized_header.as_slice()).unwrap().0;
        assert!(parsed_header.dictionary_id().is_none());
        assert_eq!(parsed_header.frame_content_size(), 1);
    }

    #[test]
    fn frame_header_decode_large_fcs() {
        let header = FrameHeader {
            frame_content_size: Some(0x01ff_ffff_ffff_ffff),
            single_segment: true,
            content_checksum: false,
            dictionary_id: None,
            window_size: None,
        };

        let mut serialized_header = Vec::new();
        header.serialize(&mut serialized_header);
        let parsed_header = read_frame_header(serialized_header.as_slice()).unwrap().0;
        assert_eq!(parsed_header.frame_content_size(), 0x01ff_ffff_ffff_ffff);
    }

    #[test]
    #[should_panic]
    fn catches_single_segment_no_fcs() {
        let header = FrameHeader {
            frame_content_size: None,
            single_segment: true,
            content_checksum: false,
            dictionary_id: None,
            window_size: Some(1),
        };

        let mut serialized_header = Vec::new();
        header.serialize(&mut serialized_header);
    }

    #[test]
    #[should_panic]
    fn catches_single_segment_no_winsize() {
        let header = FrameHeader {
            frame_content_size: Some(7),
            single_segment: false,
            content_checksum: false,
            dictionary_id: None,
            window_size: None,
        };

        let mut serialized_header = Vec::new();
        header.serialize(&mut serialized_header);
    }

    #[test]
    fn window_descriptor_encodes_exponent_and_mantissa() {
        // Validate that the encoder writes a descriptor which the decoder interprets
        // as a window >= requested, and that it's minimal among all representable values.
        fn decode_descriptor(desc: u8) -> u64 {
            let exp = (desc >> 3) as u64;
            let mantissa = (desc & 0x7) as u64;
            let window_log = 10 + exp;
            let window_base = 1u64 << window_log;
            let window_add = (window_base / 8) * mantissa;
            window_base + window_add
        }

        fn minimal_descriptor_for(requested: u64) -> u8 {
            let mut best: Option<(u64, u8)> = None;
            for exp in 0u8..=31u8 {
                for mantissa in 0u8..=7u8 {
                    let desc = (exp << 3) | mantissa;
                    let decoded = decode_descriptor(desc);
                    if decoded < requested {
                        continue;
                    }
                    if decoded > crate::common::MAX_WINDOW_SIZE {
                        continue;
                    }
                    match best {
                        None => best = Some((decoded, desc)),
                        Some((best_decoded, _)) if decoded < best_decoded => {
                            best = Some((decoded, desc))
                        }
                        _ => {}
                    }
                }
            }
            best.unwrap().1
        }

        for requested in [
            1024u64,
            1025,
            4096,
            65_536,
            128 * 1024,
            256 * 1024,
            384 * 1024, // Default currently uses 256KiB history + 128KiB current resident
            1024 * 1024,
            7 * 1024 * 1024,
        ] {
            let encoded = super::encode_window_descriptor(requested);
            let decoded = decode_descriptor(encoded);
            assert!(
                decoded >= requested,
                "requested={} desc=0x{:02x} decoded={}",
                requested,
                encoded,
                decoded
            );

            let minimal = minimal_descriptor_for(requested);
            assert_eq!(
                encoded, minimal,
                "descriptor not minimal for requested={requested}: got=0x{encoded:02x} expected=0x{minimal:02x}"
            );
        }
    }

    #[test]
    fn window_descriptor_accepts_spec_max_window_size() {
        let header = FrameHeader {
            frame_content_size: Some(1),
            single_segment: false,
            content_checksum: false,
            dictionary_id: None,
            window_size: Some(crate::common::MAX_WINDOW_SIZE),
        };

        let mut serialized_header = Vec::new();
        header.serialize(&mut serialized_header);
        let parsed_header = read_frame_header(serialized_header.as_slice()).unwrap().0;
        assert_eq!(parsed_header.window_size().unwrap(), crate::common::MAX_WINDOW_SIZE);
    }
}
