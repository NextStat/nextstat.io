//! Minimal RNTuple anchor metadata parsing.
//!
//! This module parses the serialized `ROOT::RNTuple` anchor object payload
//! stored in a top-level TKey with class name similar to `ROOT::RNTuple`.

use crate::error::{Result, RootError};
use crate::rbuffer::RBuffer;

/// Envelope type id for a header envelope.
pub const RNTUPLE_ENVELOPE_TYPE_HEADER: u16 = 0x01;
/// Envelope type id for a footer envelope.
pub const RNTUPLE_ENVELOPE_TYPE_FOOTER: u16 = 0x02;

/// Parsed metadata from a serialized `ROOT::RNTuple` anchor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleAnchor {
    /// ROOT class version for `ROOT::RNTuple` object streamer.
    pub class_version: u16,
    /// RNTuple format epoch.
    pub version_epoch: u16,
    /// RNTuple format major version.
    pub version_major: u16,
    /// RNTuple format minor version.
    pub version_minor: u16,
    /// RNTuple format patch version.
    pub version_patch: u16,
    /// File offset of header payload (excluding TKey).
    pub seek_header: u64,
    /// Compressed header byte size.
    pub nbytes_header: u64,
    /// Uncompressed header byte size.
    pub len_header: u64,
    /// File offset of footer payload (excluding TKey).
    pub seek_footer: u64,
    /// Compressed footer byte size.
    pub nbytes_footer: u64,
    /// Uncompressed footer byte size.
    pub len_footer: u64,
    /// Maximum payload size for a single TKey blob.
    pub max_key_size: u64,
}

/// Parse a `ROOT::RNTuple` anchor object payload into [`RNTupleAnchor`].
pub fn parse_rntuple_anchor_payload(payload: &[u8]) -> Result<RNTupleAnchor> {
    let mut r = RBuffer::new(payload);
    let (class_version, end_pos) = r.read_version()?;

    if class_version == 0 {
        return Err(RootError::Deserialization("invalid ROOT::RNTuple class version: 0".into()));
    }

    let anchor = RNTupleAnchor {
        class_version,
        version_epoch: r.read_u16()?,
        version_major: r.read_u16()?,
        version_minor: r.read_u16()?,
        version_patch: r.read_u16()?,
        seek_header: r.read_u64()?,
        nbytes_header: r.read_u64()?,
        len_header: r.read_u64()?,
        seek_footer: r.read_u64()?,
        nbytes_footer: r.read_u64()?,
        len_footer: r.read_u64()?,
        max_key_size: r.read_u64()?,
    };

    if let Some(end) = end_pos {
        // ROOT streamers may append versioned padding/refs; keep cursor consistent.
        r.set_pos(end);
    }

    Ok(anchor)
}

/// Parsed RNTuple envelope framing metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleEnvelopeInfo {
    /// Envelope type id (e.g. header=1, footer=2).
    pub envelope_type: u16,
    /// Declared full envelope length in bytes.
    pub envelope_len: u16,
    /// Trailing xxhash3 checksum bytes interpreted as little-endian `u64`.
    pub xxhash3_le: u64,
    /// Length of envelope payload bytes excluding preamble/postscript.
    pub payload_len: usize,
}

/// Human-readable summary extracted from a header envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleHeaderSummary {
    /// Ntuple name if a recognizable string block was found.
    pub ntuple_name: Option<String>,
    /// Writer string (e.g. `ROOT v6.38.00`) if recognizable.
    pub writer: Option<String>,
    /// Distinct length-prefixed ASCII strings discovered in payload order.
    pub strings: Vec<String>,
    /// Best-effort field tokens extracted from `strings` as `(name, type)` pairs.
    pub field_tokens: Vec<RNTupleFieldToken>,
}

/// Best-effort field token pair extracted from header strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleFieldToken {
    /// Field/column name candidate.
    pub name: String,
    /// Type name candidate.
    pub type_name: String,
}

/// Scalar type normalized from an RNTuple field type token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNTupleScalarType {
    /// Boolean value.
    Bool,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 64-bit integer.
    U64,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
}

/// Structural kind of an RNTuple field type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNTupleFieldKind {
    /// Single scalar value per entry.
    Primitive,
    /// Fixed-size array per entry (e.g. `std::array<T, N>`, `T[N]`).
    FixedArray,
    /// Variable-size array per entry (e.g. `std::vector<T>`).
    VariableArray,
    /// Nested record-like/object type.
    Nested,
    /// Not recognized by current type mapper.
    Unknown,
}

/// Best-effort schema field summary extracted from RNTuple metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleSchemaField {
    /// Field/column name.
    pub name: String,
    /// Raw type token as seen in metadata.
    pub type_name: String,
    /// Normalized scalar type when recognized.
    pub scalar_type: Option<RNTupleScalarType>,
    /// Structural type category.
    pub kind: RNTupleFieldKind,
    /// Normalized scalar type for array element when recognized.
    pub element_scalar_type: Option<RNTupleScalarType>,
    /// Fixed array length when available.
    pub fixed_len: Option<usize>,
}

/// Best-effort schema summary derived from header metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleSchemaSummary {
    /// Ntuple name if available.
    pub ntuple_name: Option<String>,
    /// Parsed field list.
    pub fields: Vec<RNTupleSchemaField>,
}

/// High-level metadata summary from anchor + header/footer envelope framing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleMetadataSummary {
    /// Parsed RNTuple anchor fields.
    pub anchor: RNTupleAnchor,
    /// Parsed header envelope framing.
    pub header_envelope: RNTupleEnvelopeInfo,
    /// Parsed footer envelope framing.
    pub footer_envelope: RNTupleEnvelopeInfo,
    /// Header string summary.
    pub header_summary: RNTupleHeaderSummary,
}

/// Parse RNTuple envelope framing.
///
/// The current ROOT payloads use a 4-byte preamble (`u16 type`, `u16 length`, LE)
/// and an 8-byte trailing checksum.
pub fn parse_rntuple_envelope(
    envelope_bytes: &[u8],
    expected_type: Option<u16>,
) -> Result<RNTupleEnvelopeInfo> {
    if envelope_bytes.len() < 12 {
        return Err(RootError::Deserialization(format!(
            "RNTuple envelope too short: {}",
            envelope_bytes.len()
        )));
    }

    let envelope_type = u16::from_le_bytes([envelope_bytes[0], envelope_bytes[1]]);
    let envelope_len = u16::from_le_bytes([envelope_bytes[2], envelope_bytes[3]]);
    let declared_len = envelope_len as usize;
    if declared_len != envelope_bytes.len() {
        return Err(RootError::Deserialization(format!(
            "RNTuple envelope length mismatch: declared={} actual={}",
            declared_len,
            envelope_bytes.len()
        )));
    }
    if let Some(t) = expected_type {
        if envelope_type != t {
            return Err(RootError::Deserialization(format!(
                "RNTuple envelope type mismatch: expected={} actual={}",
                t, envelope_type
            )));
        }
    }

    let c0 = envelope_bytes[declared_len - 8];
    let c1 = envelope_bytes[declared_len - 7];
    let c2 = envelope_bytes[declared_len - 6];
    let c3 = envelope_bytes[declared_len - 5];
    let c4 = envelope_bytes[declared_len - 4];
    let c5 = envelope_bytes[declared_len - 3];
    let c6 = envelope_bytes[declared_len - 2];
    let c7 = envelope_bytes[declared_len - 1];
    let xxhash3_le = u64::from_le_bytes([c0, c1, c2, c3, c4, c5, c6, c7]);

    Ok(RNTupleEnvelopeInfo {
        envelope_type,
        envelope_len,
        xxhash3_le,
        payload_len: declared_len - 12,
    })
}

/// Best-effort extraction of name/writer strings from a header envelope payload.
pub fn parse_rntuple_header_summary(header_envelope_bytes: &[u8]) -> Result<RNTupleHeaderSummary> {
    let info = parse_rntuple_envelope(header_envelope_bytes, Some(RNTUPLE_ENVELOPE_TYPE_HEADER))?;
    let payload = &header_envelope_bytes[4..(info.envelope_len as usize - 8)];
    let strings = collect_len_prefixed_ascii_strings(payload);
    let field_tokens = infer_field_tokens(&strings);
    let ntuple_name = strings.first().cloned();
    let writer = strings.get(1).cloned();
    Ok(RNTupleHeaderSummary { ntuple_name, writer, strings, field_tokens })
}

/// Build best-effort schema summary from parsed header metadata.
pub fn parse_rntuple_schema_summary(header: &RNTupleHeaderSummary) -> RNTupleSchemaSummary {
    let mut fields = Vec::new();
    for tok in &header.field_tokens {
        if fields.iter().any(|f: &RNTupleSchemaField| f.name == tok.name) {
            continue;
        }
        let mapped = map_field_type(&tok.type_name);
        fields.push(RNTupleSchemaField {
            name: tok.name.clone(),
            type_name: tok.type_name.clone(),
            scalar_type: mapped.scalar_type,
            kind: mapped.kind,
            element_scalar_type: mapped.element_scalar_type,
            fixed_len: mapped.fixed_len,
        });
    }
    RNTupleSchemaSummary { ntuple_name: header.ntuple_name.clone(), fields }
}

#[derive(Debug, Clone, Copy)]
struct FieldTypeMap {
    kind: RNTupleFieldKind,
    scalar_type: Option<RNTupleScalarType>,
    element_scalar_type: Option<RNTupleScalarType>,
    fixed_len: Option<usize>,
}

fn collect_len_prefixed_ascii_strings(payload: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < payload.len() {
        if let Some((s, next)) = read_len_prefixed_ascii(payload, pos) {
            if !out.iter().any(|existing| existing == &s) {
                out.push(s);
            }
            pos = next;
            continue;
        }
        pos += 1;
    }
    out
}

fn read_len_prefixed_ascii(payload: &[u8], start: usize) -> Option<(String, usize)> {
    if start + 4 > payload.len() {
        return None;
    }
    let len = u32::from_le_bytes([
        payload[start],
        payload[start + 1],
        payload[start + 2],
        payload[start + 3],
    ]) as usize;
    if len == 0 {
        return None;
    }
    let str_start = start + 4;
    let str_end = str_start.checked_add(len)?;
    if str_end > payload.len() {
        return None;
    }
    let sbytes = &payload[str_start..str_end];
    if !sbytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        return None;
    }
    let s = String::from_utf8(sbytes.to_vec()).ok()?;
    Some((s, str_end))
}

fn infer_field_tokens(strings: &[String]) -> Vec<RNTupleFieldToken> {
    let mut out = Vec::new();
    for pair in strings.windows(2) {
        let name = &pair[0];
        let ty = &pair[1];
        if !looks_like_field_name(name) {
            continue;
        }
        if !looks_like_type_name(ty) {
            continue;
        }
        out.push(RNTupleFieldToken { name: name.clone(), type_name: ty.clone() });
    }
    out
}

fn looks_like_field_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        && !s.contains("ROOT")
        && !s.contains("std::")
}

fn looks_like_type_name(s: &str) -> bool {
    let ls = s.to_ascii_lowercase();
    ls.contains("int")
        || ls.contains("float")
        || ls.contains("double")
        || ls.contains("bool")
        || ls.contains("std::")
}

fn normalize_scalar_type(type_name: &str) -> Option<RNTupleScalarType> {
    let t = type_name.to_ascii_lowercase();
    if t.contains('<') || t.contains('>') || t.contains('[') || t.contains(']') || t.contains(',') {
        return None;
    }
    if t == "bool" || t == "std::bool_t" {
        return Some(RNTupleScalarType::Bool);
    }
    if t == "uint64_t" || t == "std::uint64_t" || t == "u64" || t == "unsignedlonglong" {
        return Some(RNTupleScalarType::U64);
    }
    if t == "int64_t" || t == "std::int64_t" || t == "i64" || t == "longlong" {
        return Some(RNTupleScalarType::I64);
    }
    if t == "uint32_t" || t == "std::uint32_t" || t == "u32" || t == "unsignedint" {
        return Some(RNTupleScalarType::U32);
    }
    if t == "int32_t" || t == "std::int32_t" || t == "i32" || t == "int" {
        return Some(RNTupleScalarType::I32);
    }
    if t == "uint16_t" || t == "std::uint16_t" || t == "u16" || t == "unsignedshort" {
        return Some(RNTupleScalarType::U16);
    }
    if t == "int16_t" || t == "std::int16_t" || t == "i16" || t == "short" {
        return Some(RNTupleScalarType::I16);
    }
    if t == "uint8_t" || t == "std::uint8_t" || t == "u8" || t == "unsignedchar" {
        return Some(RNTupleScalarType::U8);
    }
    if t == "int8_t" || t == "std::int8_t" || t == "i8" || t == "char" || t == "signedchar" {
        return Some(RNTupleScalarType::I8);
    }
    if t == "double" || t == "f64" {
        return Some(RNTupleScalarType::F64);
    }
    if t == "float" || t == "f32" {
        return Some(RNTupleScalarType::F32);
    }
    None
}

fn map_field_type(type_name: &str) -> FieldTypeMap {
    let compact = type_name.replace(' ', "");
    if let Some((elem_ty, len)) = parse_std_array(&compact).or_else(|| parse_c_array(&compact)) {
        return FieldTypeMap {
            kind: RNTupleFieldKind::FixedArray,
            scalar_type: None,
            element_scalar_type: normalize_scalar_type(elem_ty),
            fixed_len: Some(len),
        };
    }

    if let Some(elem_ty) = parse_std_vector_like(&compact) {
        return FieldTypeMap {
            kind: RNTupleFieldKind::VariableArray,
            scalar_type: None,
            element_scalar_type: normalize_scalar_type(elem_ty),
            fixed_len: None,
        };
    }

    if let Some(scalar) = normalize_scalar_type(&compact) {
        return FieldTypeMap {
            kind: RNTupleFieldKind::Primitive,
            scalar_type: Some(scalar),
            element_scalar_type: None,
            fixed_len: None,
        };
    }

    if looks_like_nested_type(&compact) {
        return FieldTypeMap {
            kind: RNTupleFieldKind::Nested,
            scalar_type: None,
            element_scalar_type: None,
            fixed_len: None,
        };
    }

    FieldTypeMap {
        kind: RNTupleFieldKind::Unknown,
        scalar_type: None,
        element_scalar_type: None,
        fixed_len: None,
    }
}

fn parse_std_array(type_name: &str) -> Option<(&str, usize)> {
    let inner = template_payload(type_name, "std::array<")
        .or_else(|| template_payload(type_name, "array<"))?;
    let (elem_ty, len_str) = split_top_level_once(inner, ',')?;
    let len = len_str.parse::<usize>().ok()?;
    Some((elem_ty, len))
}

fn parse_std_vector_like(type_name: &str) -> Option<&str> {
    let inner = template_payload(type_name, "std::vector<")
        .or_else(|| template_payload(type_name, "vector<"))
        .or_else(|| template_payload(type_name, "ROOT::VecOps::RVec<"))
        .or_else(|| template_payload(type_name, "RVec<"))?;
    let (elem_ty, _) = split_top_level_once(inner, ',').unwrap_or((inner, ""));
    Some(elem_ty)
}

fn parse_c_array(type_name: &str) -> Option<(&str, usize)> {
    let end = type_name.len();
    if !type_name.ends_with(']') {
        return None;
    }
    let lb = type_name.rfind('[')?;
    if lb + 1 >= end.saturating_sub(1) {
        return None;
    }
    let len = type_name[lb + 1..end - 1].parse::<usize>().ok()?;
    let elem_ty = &type_name[..lb];
    if elem_ty.is_empty() {
        return None;
    }
    Some((elem_ty, len))
}

fn template_payload<'a>(type_name: &'a str, prefix: &str) -> Option<&'a str> {
    if !type_name.starts_with(prefix) || !type_name.ends_with('>') {
        return None;
    }
    Some(&type_name[prefix.len()..type_name.len() - 1])
}

fn split_top_level_once(s: &str, delim: char) -> Option<(&str, &str)> {
    let mut depth = 0usize;
    for (idx, ch) in s.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if ch == delim && depth == 0 => {
                let left = &s[..idx];
                let right = &s[idx + ch.len_utf8()..];
                return Some((left, right));
            }
            _ => {}
        }
    }
    None
}

fn looks_like_nested_type(type_name: &str) -> bool {
    if type_name.is_empty() {
        return false;
    }
    if type_name.starts_with("std::") {
        return type_name.contains("tuple<")
            || type_name.contains("pair<")
            || type_name.contains("optional<");
    }
    type_name.contains("::")
        || type_name.contains("struct")
        || type_name.contains("class")
        || type_name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_zero_class_version() {
        let mut payload = Vec::new();
        // Non-bytecount mode: version encoded in the high 16 bits of the first u32.
        // version=0, then no body.
        payload.extend_from_slice(&0u32.to_be_bytes());
        let err = parse_rntuple_anchor_payload(&payload).expect_err("expected parse failure");
        assert!(matches!(err, RootError::Deserialization(_)));
    }

    #[test]
    fn parse_roundtrip_minimal_bytecount_payload() {
        let mut payload = Vec::new();
        // Bytecount mode: byte count covers from right after this u32 to object end.
        // Payload body is: u16 class version + 4*u16 + 7*u64 = 2 + 8 + 56 = 66 bytes.
        let byte_count = 66u32;
        payload.extend_from_slice(&(0x4000_0000u32 | byte_count).to_be_bytes());
        payload.extend_from_slice(&2u16.to_be_bytes()); // class_version

        // 4x u16 version parts
        payload.extend_from_slice(&1u16.to_be_bytes()); // epoch
        payload.extend_from_slice(&0u16.to_be_bytes()); // major
        payload.extend_from_slice(&1u16.to_be_bytes()); // minor
        payload.extend_from_slice(&0u16.to_be_bytes()); // patch

        // 7x u64 anchor fields
        for v in [10u64, 11, 12, 20, 21, 22, 1024] {
            payload.extend_from_slice(&v.to_be_bytes());
        }

        let anchor = parse_rntuple_anchor_payload(&payload).expect("parse failed");
        assert_eq!(anchor.class_version, 2);
        assert_eq!(anchor.version_epoch, 1);
        assert_eq!(anchor.version_major, 0);
        assert_eq!(anchor.version_minor, 1);
        assert_eq!(anchor.version_patch, 0);
        assert_eq!(anchor.seek_header, 10);
        assert_eq!(anchor.nbytes_header, 11);
        assert_eq!(anchor.len_header, 12);
        assert_eq!(anchor.seek_footer, 20);
        assert_eq!(anchor.nbytes_footer, 21);
        assert_eq!(anchor.len_footer, 22);
        assert_eq!(anchor.max_key_size, 1024);
    }

    #[test]
    fn parse_envelope_rejects_length_mismatch() {
        let data = [1u8, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = parse_rntuple_envelope(&data, Some(RNTUPLE_ENVELOPE_TYPE_HEADER))
            .expect_err("expected mismatch");
        assert!(matches!(err, RootError::Deserialization(_)));
    }

    #[test]
    fn parse_envelope_smoke() {
        let mut data = Vec::new();
        data.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_HEADER.to_le_bytes());
        data.extend_from_slice(&(16u16).to_le_bytes());
        data.extend_from_slice(&[1, 2, 3, 4]); // payload
        data.extend_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes()); // checksum
        let info = parse_rntuple_envelope(&data, Some(RNTUPLE_ENVELOPE_TYPE_HEADER))
            .expect("parse failed");
        assert_eq!(info.envelope_type, RNTUPLE_ENVELOPE_TYPE_HEADER);
        assert_eq!(info.envelope_len, 16);
        assert_eq!(info.payload_len, 4);
        assert_eq!(info.xxhash3_le, 0x1122_3344_5566_7788u64);
    }

    #[test]
    fn header_summary_string_scan_smoke() {
        // Build synthetic envelope payload with two aligned strings.
        // envelope = 4-byte preamble + payload + 8-byte checksum
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0u8; 12]); // prefix area
        payload.extend_from_slice(&(6u32).to_le_bytes());
        payload.extend_from_slice(b"Events");
        payload.extend_from_slice(&[0u8; 0]); // already aligned
        payload.extend_from_slice(&(13u32).to_le_bytes());
        payload.extend_from_slice(b"ROOT v6.38.00");

        let total_len = 4 + payload.len() + 8;
        let mut env = Vec::new();
        env.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_HEADER.to_le_bytes());
        env.extend_from_slice(&(total_len as u16).to_le_bytes());
        env.extend_from_slice(&payload);
        env.extend_from_slice(&0u64.to_le_bytes());

        let summary = parse_rntuple_header_summary(&env).expect("summary parse failed");
        assert_eq!(summary.ntuple_name.as_deref(), Some("Events"));
        assert_eq!(summary.writer.as_deref(), Some("ROOT v6.38.00"));
        assert!(summary.strings.contains(&"Events".to_string()));
        assert!(summary.strings.contains(&"ROOT v6.38.00".to_string()));
        assert!(summary.field_tokens.is_empty());
    }

    #[test]
    fn infer_field_tokens_smoke() {
        let strings = vec![
            "Events".to_string(),
            "ROOT v6.38.00".to_string(),
            "pt".to_string(),
            "float".to_string(),
            "n".to_string(),
            "std::int32_t".to_string(),
        ];
        let fields = infer_field_tokens(&strings);
        assert!(fields.iter().any(|f| f.name == "pt" && f.type_name == "float"));
        assert!(fields.iter().any(|f| f.name == "n" && f.type_name == "std::int32_t"));
    }

    #[test]
    fn schema_summary_smoke() {
        let header = RNTupleHeaderSummary {
            ntuple_name: Some("Events".to_string()),
            writer: Some("ROOT v6.38.00".to_string()),
            strings: vec![],
            field_tokens: vec![
                RNTupleFieldToken { name: "pt".to_string(), type_name: "float".to_string() },
                RNTupleFieldToken { name: "n".to_string(), type_name: "std::int32_t".to_string() },
            ],
        };
        let schema = parse_rntuple_schema_summary(&header);
        assert_eq!(schema.ntuple_name.as_deref(), Some("Events"));
        assert!(schema.fields.iter().any(|f| {
            f.name == "pt"
                && f.scalar_type == Some(RNTupleScalarType::F32)
                && f.kind == RNTupleFieldKind::Primitive
        }));
        assert!(schema.fields.iter().any(|f| {
            f.name == "n"
                && f.scalar_type == Some(RNTupleScalarType::I32)
                && f.kind == RNTupleFieldKind::Primitive
        }));
    }

    #[test]
    fn schema_summary_maps_arrays_and_nested_types() {
        let header = RNTupleHeaderSummary {
            ntuple_name: Some("Events".to_string()),
            writer: Some("ROOT v6.38.00".to_string()),
            strings: vec![],
            field_tokens: vec![
                RNTupleFieldToken {
                    name: "arr_fixed".to_string(),
                    type_name: "std::array<float,3>".to_string(),
                },
                RNTupleFieldToken {
                    name: "arr_var".to_string(),
                    type_name: "std::vector<std::int32_t>".to_string(),
                },
                RNTupleFieldToken {
                    name: "nested".to_string(),
                    type_name: "MyNamespace::Track".to_string(),
                },
            ],
        };
        let schema = parse_rntuple_schema_summary(&header);
        assert!(schema.fields.iter().any(|f| {
            f.name == "arr_fixed"
                && f.kind == RNTupleFieldKind::FixedArray
                && f.element_scalar_type == Some(RNTupleScalarType::F32)
                && f.fixed_len == Some(3)
        }));
        assert!(schema.fields.iter().any(|f| {
            f.name == "arr_var"
                && f.kind == RNTupleFieldKind::VariableArray
                && f.element_scalar_type == Some(RNTupleScalarType::I32)
        }));
        assert!(schema.fields.iter().any(|f| {
            f.name == "nested"
                && f.kind == RNTupleFieldKind::Nested
                && f.scalar_type.is_none()
                && f.element_scalar_type.is_none()
                && f.fixed_len.is_none()
        }));
    }

    #[test]
    fn map_field_type_supports_c_array_and_unknown() {
        let c_arr = map_field_type("double[4]");
        assert_eq!(c_arr.kind, RNTupleFieldKind::FixedArray);
        assert_eq!(c_arr.element_scalar_type, Some(RNTupleScalarType::F64));
        assert_eq!(c_arr.fixed_len, Some(4));

        let unknown = map_field_type("opaque_blob_t");
        assert_eq!(unknown.kind, RNTupleFieldKind::Unknown);
        assert_eq!(unknown.scalar_type, None);
        assert_eq!(unknown.element_scalar_type, None);
        assert_eq!(unknown.fixed_len, None);
    }

    #[test]
    fn split_top_level_once_handles_nested_templates() {
        let (left, right) = split_top_level_once("std::vector<std::int32_t>,allocator", ',')
            .expect("expected split");
        assert_eq!(left, "std::vector<std::int32_t>");
        assert_eq!(right, "allocator");
    }

    #[test]
    fn map_field_type_supports_rvec_alias() {
        let mapped = map_field_type("ROOT::VecOps::RVec<float>");
        assert_eq!(mapped.kind, RNTupleFieldKind::VariableArray);
        assert_eq!(mapped.element_scalar_type, Some(RNTupleScalarType::F32));
    }

    #[test]
    fn parse_std_array_rejects_non_numeric_len() {
        assert!(parse_std_array("std::array<float,N>").is_none());
    }

    #[test]
    fn parse_c_array_rejects_missing_len() {
        assert!(parse_c_array("float[]").is_none());
    }

    #[test]
    fn looks_like_nested_type_std_scalar_is_not_nested() {
        assert!(!looks_like_nested_type("std::int32_t"));
        assert!(looks_like_nested_type("std::pair<float,float>"));
    }

    #[test]
    fn split_top_level_once_none_without_delim() {
        assert!(split_top_level_once("std::vector<float>", ',').is_none());
    }

    #[test]
    fn template_payload_requires_full_wrapper() {
        assert!(template_payload("xstd::array<float,3>", "std::array<").is_none());
        assert!(template_payload("std::array<float,3>tail", "std::array<").is_none());
    }

    #[test]
    fn map_field_type_for_std_optional_is_nested() {
        let mapped = map_field_type("std::optional<MyType>");
        assert_eq!(mapped.kind, RNTupleFieldKind::Nested);
    }

    #[test]
    fn map_field_type_for_vector_with_allocator() {
        let mapped = map_field_type("std::vector<std::uint64_t,std::allocator<std::uint64_t>>");
        assert_eq!(mapped.kind, RNTupleFieldKind::VariableArray);
        assert_eq!(mapped.element_scalar_type, Some(RNTupleScalarType::U64));
    }

    #[test]
    fn map_field_type_for_array_with_nested_element() {
        let mapped = map_field_type("std::array<MyNs::Hit,8>");
        assert_eq!(mapped.kind, RNTupleFieldKind::FixedArray);
        assert_eq!(mapped.element_scalar_type, None);
        assert_eq!(mapped.fixed_len, Some(8));
    }

    #[test]
    fn map_field_type_for_prefixed_custom_nested() {
        let mapped = map_field_type("custom::RecoTrack");
        assert_eq!(mapped.kind, RNTupleFieldKind::Nested);
    }

    #[test]
    fn map_field_type_primitive_keeps_scalar_only() {
        let mapped = map_field_type("std::int16_t");
        assert_eq!(mapped.kind, RNTupleFieldKind::Primitive);
        assert_eq!(mapped.scalar_type, Some(RNTupleScalarType::I16));
        assert_eq!(mapped.element_scalar_type, None);
        assert_eq!(mapped.fixed_len, None);
    }

    #[test]
    fn map_field_type_whitespace_normalization() {
        let mapped = map_field_type("std::array< float , 5 >");
        assert_eq!(mapped.kind, RNTupleFieldKind::FixedArray);
        assert_eq!(mapped.element_scalar_type, Some(RNTupleScalarType::F32));
        assert_eq!(mapped.fixed_len, Some(5));
    }

    #[test]
    fn parse_std_vector_like_alias_forms() {
        assert_eq!(parse_std_vector_like("vector<float>"), Some("float"));
        assert_eq!(parse_std_vector_like("RVec<std::int32_t>"), Some("std::int32_t"));
    }

    #[test]
    fn parse_std_vector_like_rejects_non_wrapped() {
        assert!(parse_std_vector_like("xvector<float>").is_none());
    }

    #[test]
    fn parse_c_array_rejects_non_terminal_brackets() {
        assert!(parse_c_array("float[4]x").is_none());
    }

    #[test]
    fn looks_like_nested_type_uppercase_token() {
        assert!(looks_like_nested_type("Track"));
    }

    #[test]
    fn map_field_type_unknown_lowercase_token() {
        let mapped = map_field_type("blob");
        assert_eq!(mapped.kind, RNTupleFieldKind::Unknown);
    }

    #[test]
    fn split_top_level_once_handles_angle_depth() {
        let (left, right) = split_top_level_once("A<B<C,D>>,E", ',').expect("split");
        assert_eq!(left, "A<B<C,D>>");
        assert_eq!(right, "E");
    }

    #[test]
    fn parse_std_array_simple() {
        let (elem, len) = parse_std_array("std::array<std::uint16_t,12>").expect("array");
        assert_eq!(elem, "std::uint16_t");
        assert_eq!(len, 12);
    }

    #[test]
    fn parse_c_array_simple() {
        let (elem, len) = parse_c_array("std::uint8_t[64]").expect("c array");
        assert_eq!(elem, "std::uint8_t");
        assert_eq!(len, 64);
    }

    #[test]
    fn map_field_type_nested_std_pair() {
        let mapped = map_field_type("std::pair<float,float>");
        assert_eq!(mapped.kind, RNTupleFieldKind::Nested);
    }

    #[test]
    fn template_payload_extracts_inner() {
        assert_eq!(template_payload("std::vector<float>", "std::vector<"), Some("float"));
    }
}
