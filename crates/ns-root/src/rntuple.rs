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
/// Envelope type id for a page-list envelope.
pub const RNTUPLE_ENVELOPE_TYPE_PAGELIST: u16 = 0x03;

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

/// Best-effort file locator for an envelope/page-list blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleLocatorSummary {
    /// Number of bytes on storage (usually compressed size).
    pub nbytes_on_storage: u64,
    /// Byte position in file.
    pub position: u64,
    /// Locator type id (`0` for file locators in current fixtures).
    pub locator_type: u8,
    /// Backend-specific reserved byte.
    pub reserved: u8,
}

/// Cluster-group summary extracted from footer metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleClusterGroupSummary {
    /// First global entry index covered by this cluster group.
    pub min_entry: u64,
    /// Number of entries covered by this cluster group.
    pub entry_span: u64,
    /// Number of clusters in this cluster group.
    pub n_clusters: u32,
    /// Uncompressed page-list envelope length.
    pub page_list_envelope_len: u64,
    /// Locator of the page-list envelope blob.
    pub page_list_locator: RNTupleLocatorSummary,
}

/// Footer metadata summary needed for page/cluster navigation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTupleFooterSummary {
    /// Header xxhash3 referenced by the footer payload.
    pub header_xxhash3_le: u64,
    /// Parsed cluster groups.
    pub cluster_groups: Vec<RNTupleClusterGroupSummary>,
}

/// Raw page descriptor summary extracted from a page-list envelope.
///
/// Field semantics beyond locator/size are still being reverse-engineered for
/// full native decoding; the `*_raw` fields preserve observed record values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTuplePageSummary {
    /// Raw record tag (`i64`) observed in fixtures (`-40` for page entries).
    pub record_tag_raw: i64,
    /// Raw repetition/count field (`u32`), observed as `1` in fixtures.
    pub repetition_raw: u32,
    /// Raw signed element-count hint (`i32`) from page records.
    pub element_count_raw: i32,
    /// Bytes on storage for this page blob.
    pub nbytes_on_storage: u32,
    /// File byte position for this page blob.
    pub position: u64,
    /// Raw locator type marker.
    pub locator_type_raw: u32,
    /// Raw locator reserved marker.
    pub locator_reserved_raw: u32,
    /// Raw trailing cluster-anchor marker from the page record.
    pub cluster_anchor_raw: u32,
}

/// Page-list summary needed to load raw page blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RNTuplePageListSummary {
    /// Footer xxhash3 referenced by the page-list payload.
    pub footer_xxhash3_le: u64,
    /// Parsed page descriptors.
    pub pages: Vec<RNTuplePageSummary>,
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
        if tok.name.starts_with('_') {
            continue;
        }
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

/// Build schema summary from header tokens plus footer-discovered `(name,type)` tokens.
///
/// Footer tokens are appended only for field names that are absent in the header token set.
pub(crate) fn parse_rntuple_schema_summary_with_footer(
    header: &RNTupleHeaderSummary,
    footer_envelope_bytes: &[u8],
) -> Result<RNTupleSchemaSummary> {
    let footer_tokens = parse_rntuple_footer_field_tokens(footer_envelope_bytes)?;
    let mut merged_tokens = header.field_tokens.clone();
    for tok in footer_tokens {
        if tok.name.starts_with('_') {
            continue;
        }
        if merged_tokens.iter().any(|existing| existing.name == tok.name) {
            continue;
        }
        merged_tokens.push(tok);
    }
    let merged_header = RNTupleHeaderSummary {
        ntuple_name: header.ntuple_name.clone(),
        writer: header.writer.clone(),
        strings: header.strings.clone(),
        field_tokens: merged_tokens,
    };
    Ok(parse_rntuple_schema_summary(&merged_header))
}

/// Parse footer payload into cluster-group/page-list summary.
///
/// The parser currently targets the envelope/layout emitted by ROOT 6.38 fixtures and
/// extracts the cluster-group list required to load page-list envelopes.
pub fn parse_rntuple_footer_summary(footer_envelope_bytes: &[u8]) -> Result<RNTupleFooterSummary> {
    let info = parse_rntuple_envelope(footer_envelope_bytes, Some(RNTUPLE_ENVELOPE_TYPE_FOOTER))?;
    let payload = &footer_envelope_bytes[4..(info.envelope_len as usize - 8)];
    if payload.len() < 20 {
        return Err(RootError::Deserialization(format!(
            "RNTuple footer payload too short: {}",
            payload.len()
        )));
    }

    // Observed in ROOT 6.38 fixtures: 12-byte prefix followed by header xxhash3.
    let header_xxhash3_le = le_u64(payload, 12)?;
    let cluster_groups = parse_cluster_groups(payload)?;

    Ok(RNTupleFooterSummary { header_xxhash3_le, cluster_groups })
}

/// Best-effort extraction of field tokens from footer payload strings.
pub(crate) fn parse_rntuple_footer_field_tokens(
    footer_envelope_bytes: &[u8],
) -> Result<Vec<RNTupleFieldToken>> {
    let info = parse_rntuple_envelope(footer_envelope_bytes, Some(RNTUPLE_ENVELOPE_TYPE_FOOTER))?;
    let payload = &footer_envelope_bytes[4..(info.envelope_len as usize - 8)];
    let strings = collect_len_prefixed_ascii_strings(payload);
    Ok(infer_field_tokens(&strings))
}

/// Parse page-list payload into page locator summary.
///
/// The parser currently targets ROOT 6.38 fixture layout and extracts page
/// descriptors required to load raw page blobs for downstream decode stages.
pub fn parse_rntuple_pagelist_summary(
    page_list_envelope_bytes: &[u8],
) -> Result<RNTuplePageListSummary> {
    let info =
        parse_rntuple_envelope(page_list_envelope_bytes, Some(RNTUPLE_ENVELOPE_TYPE_PAGELIST))?;
    let payload = &page_list_envelope_bytes[4..(info.envelope_len as usize - 8)];

    // Observed ROOT 6.38 layout:
    // - variable-size preamble (includes footer hash and frame metadata)
    // - repeated 40-byte page records
    const PAGELIST_RECORD_LEN: usize = 40;
    if payload.len() < PAGELIST_RECORD_LEN {
        return Err(RootError::Deserialization(format!(
            "RNTuple page-list payload too short: {}",
            payload.len()
        )));
    }
    let footer_xxhash3_le = le_u64(payload, 4)?;

    // In larger layouts, page-list payload may contain multiple record sections
    // separated by additional frame metadata. Scan aligned offsets and collect
    // plausible 40-byte records without assuming one contiguous block.
    let mut pages = Vec::new();
    let mut last_selected_off = None::<usize>;
    for off in (0..=payload.len().saturating_sub(PAGELIST_RECORD_LEN)).step_by(4) {
        let record_tag_raw = match le_i64(payload, off) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let repetition_raw = match le_u32(payload, off + 8) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let element_count_raw = match le_i32(payload, off + 12) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let nbytes_on_storage = match le_u32(payload, off + 16) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let pos_lo = match le_u32(payload, off + 20) {
            Ok(v) => v as u64,
            Err(_) => continue,
        };
        let pos_hi = match le_u32(payload, off + 24) {
            Ok(v) => v as u64,
            Err(_) => continue,
        };
        let position = pos_lo | (pos_hi << 32);
        let locator_type_raw = match le_u32(payload, off + 28) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let locator_reserved_raw = match le_u32(payload, off + 32) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let cluster_anchor_raw = match le_u32(payload, off + 36) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let plausible = (-256..=-1).contains(&record_tag_raw)
            && (1..=4096).contains(&repetition_raw)
            && element_count_raw < 0
            && (1..=536_870_912).contains(&nbytes_on_storage)
            && position > 0;
        if !plausible {
            continue;
        }
        if let Some(last_off) = last_selected_off {
            if off < last_off.saturating_add(PAGELIST_RECORD_LEN) {
                continue;
            }
        }
        last_selected_off = Some(off);
        pages.push(RNTuplePageSummary {
            record_tag_raw,
            repetition_raw,
            element_count_raw,
            nbytes_on_storage,
            position,
            locator_type_raw,
            locator_reserved_raw,
            cluster_anchor_raw,
        });
    }

    if pages.is_empty() {
        return Err(RootError::Deserialization(format!(
            "RNTuple page-list record framing mismatch: payload={} record={}",
            payload.len(),
            PAGELIST_RECORD_LEN
        )));
    }

    Ok(RNTuplePageListSummary { footer_xxhash3_le, pages })
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

fn parse_cluster_groups(payload: &[u8]) -> Result<Vec<RNTupleClusterGroupSummary>> {
    // Locate a list-frame preamble `[n_items: u32][frame_len: u32]`.
    for i in 0..payload.len().saturating_sub(8) {
        let n_items = le_u32(payload, i)? as usize;
        if n_items == 0 || n_items > 4096 {
            continue;
        }
        let frame_len = le_u32(payload, i + 4)? as usize;
        if frame_len == 0 || frame_len % n_items != 0 {
            continue;
        }

        let body_start = i + 8;
        let body_end = body_start + frame_len;
        if body_end > payload.len() {
            continue;
        }

        let item_size = frame_len / n_items;
        if item_size < 40 {
            continue;
        }

        // ROOT producer variants encode list items either as direct records
        // or as 4-byte-record-length-prefixed payloads.
        for shift in [0usize, 2usize, 4usize, 6usize] {
            if item_size >= shift + 40 {
                if let Some(out) =
                    try_parse_cluster_groups_frame(payload, body_start, n_items, item_size, shift)
                {
                    return Ok(out);
                }
            }
        }
    }

    if let Some(out) = scan_cluster_group_records(payload) {
        return Ok(out);
    }

    Err(RootError::Deserialization("RNTuple footer cluster-group frame not found".into()))
}

fn cluster_group_sequence_score(groups: &[RNTupleClusterGroupSummary]) -> u64 {
    groups.iter().fold(0u64, |acc, cg| {
        acc.saturating_add(cg.entry_span)
            .saturating_add(cg.n_clusters as u64)
            .saturating_add(cg.page_list_locator.position)
    })
}

fn try_parse_cluster_groups_frame(
    payload: &[u8],
    body_start: usize,
    n_items: usize,
    item_size: usize,
    item_base_shift: usize,
) -> Option<Vec<RNTupleClusterGroupSummary>> {
    const MAX_REASONABLE_LOCATOR_BYTES: u64 = 1 << 31; // 2 GiB
    const MAX_REASONABLE_ENVELOPE_LEN: u64 = 1 << 31; // 2 GiB

    if item_size < item_base_shift + 40 {
        return None;
    }

    let mut out = Vec::with_capacity(n_items);
    let mut prev_min_entry = 0u64;
    let mut prev_entry_span = 0u64;
    for idx in 0..n_items {
        let off = body_start + idx * item_size;
        let base = off + item_base_shift;
        let min_entry = le_u64(payload, base).ok()?;
        let entry_span = le_u64(payload, base + 8).ok()?;
        let n_clusters = le_u32(payload, base + 16).ok()?;
        let page_list_envelope_len = le_u64(payload, base + 20).ok()?;
        let nbytes_on_storage = le_u32(payload, base + 28).ok()? as u64;
        let pos_lo = le_u32(payload, base + 32).ok()? as u64;
        let pos_hi = le_u32(payload, base + 36).ok()? as u64;
        let position = pos_lo | (pos_hi << 32);

        if idx == 0 && min_entry != 0 {
            return None;
        }
        if idx > 0
            && (min_entry < prev_min_entry
                || min_entry < prev_min_entry.saturating_add(prev_entry_span))
        {
            return None;
        }
        if entry_span == 0 || n_clusters == 0 || n_clusters > 4096 {
            return None;
        }
        if page_list_envelope_len == 0
            || page_list_envelope_len > MAX_REASONABLE_ENVELOPE_LEN
            || nbytes_on_storage == 0
            || nbytes_on_storage > page_list_envelope_len
            || nbytes_on_storage > MAX_REASONABLE_LOCATOR_BYTES
            || position == 0
        {
            return None;
        }

        prev_min_entry = min_entry;
        prev_entry_span = entry_span;
        out.push(RNTupleClusterGroupSummary {
            min_entry,
            entry_span,
            n_clusters,
            page_list_envelope_len,
            page_list_locator: RNTupleLocatorSummary {
                nbytes_on_storage,
                position,
                locator_type: 0,
                reserved: 0,
            },
        });
    }

    Some(out)
}

fn scan_cluster_group_records(payload: &[u8]) -> Option<Vec<RNTupleClusterGroupSummary>> {
    fn parse_cluster_group_record(
        payload: &[u8],
        base: usize,
    ) -> Option<RNTupleClusterGroupSummary> {
        const MAX_REASONABLE_LOCATOR_BYTES: u64 = 1 << 31; // 2 GiB
        const MAX_REASONABLE_ENVELOPE_LEN: u64 = 1 << 31; // 2 GiB

        let min_entry = le_u64(payload, base).ok()?;
        let entry_span = le_u64(payload, base + 8).ok()?;
        let n_clusters = le_u32(payload, base + 16).ok()?;
        let page_list_envelope_len = le_u64(payload, base + 20).ok()?;
        let nbytes_on_storage = le_u32(payload, base + 28).ok()? as u64;
        let pos_lo = le_u32(payload, base + 32).ok()? as u64;
        let pos_hi = le_u32(payload, base + 36).ok()? as u64;
        let position = pos_lo | (pos_hi << 32);

        if entry_span == 0 || n_clusters == 0 || n_clusters > 4096 {
            return None;
        }
        if page_list_envelope_len == 0
            || page_list_envelope_len > MAX_REASONABLE_ENVELOPE_LEN
            || nbytes_on_storage == 0
            || nbytes_on_storage > page_list_envelope_len
            || nbytes_on_storage > MAX_REASONABLE_LOCATOR_BYTES
            || position == 0
        {
            return None;
        }

        Some(RNTupleClusterGroupSummary {
            min_entry,
            entry_span,
            n_clusters,
            page_list_envelope_len,
            page_list_locator: RNTupleLocatorSummary {
                nbytes_on_storage,
                position,
                locator_type: 0,
                reserved: 0,
            },
        })
    }

    let mut best: Option<Vec<RNTupleClusterGroupSummary>> = None;

    for item_size in (40usize..=96usize).step_by(4) {
        for item_base_shift in [0usize, 2usize, 4usize, 6usize] {
            if item_size < item_base_shift + 40 {
                continue;
            }
            for body_start in 0..payload.len().saturating_sub(item_base_shift + 40) {
                let base0 = body_start + item_base_shift;
                let Some(first) = parse_cluster_group_record(payload, base0) else {
                    continue;
                };
                if first.min_entry != 0 {
                    continue;
                }

                let mut out = vec![first];
                let mut prev_min = out[0].min_entry;
                let mut prev_span = out[0].entry_span;
                let mut prev_pos = out[0].page_list_locator.position;

                loop {
                    let idx = out.len();
                    let next_body = body_start + idx * item_size;
                    if next_body + item_base_shift + 40 > payload.len() {
                        break;
                    }
                    let next_base = next_body + item_base_shift;
                    let Some(next) = parse_cluster_group_record(payload, next_base) else {
                        break;
                    };
                    if next.min_entry < prev_min.saturating_add(prev_span) {
                        break;
                    }
                    if next.page_list_locator.position <= prev_pos {
                        break;
                    }
                    prev_min = next.min_entry;
                    prev_span = next.entry_span;
                    prev_pos = next.page_list_locator.position;
                    out.push(next);
                }

                let replace = match &best {
                    Some(cur) if out.len() < cur.len() => false,
                    Some(cur) if out.len() == cur.len() => {
                        cluster_group_sequence_score(&out) < cluster_group_sequence_score(cur)
                    }
                    Some(_) => true,
                    None => true,
                };
                if replace {
                    best = Some(out);
                }
            }
        }
    }

    best
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

fn le_u32(data: &[u8], off: usize) -> Result<u32> {
    let b0 = *data.get(off).ok_or_else(|| {
        RootError::Deserialization(format!("u32 read out of bounds at offset {}", off))
    })?;
    let b1 = *data.get(off + 1).ok_or_else(|| {
        RootError::Deserialization(format!("u32 read out of bounds at offset {}", off))
    })?;
    let b2 = *data.get(off + 2).ok_or_else(|| {
        RootError::Deserialization(format!("u32 read out of bounds at offset {}", off))
    })?;
    let b3 = *data.get(off + 3).ok_or_else(|| {
        RootError::Deserialization(format!("u32 read out of bounds at offset {}", off))
    })?;
    Ok(u32::from_le_bytes([b0, b1, b2, b3]))
}

fn le_u64(data: &[u8], off: usize) -> Result<u64> {
    let b0 = *data.get(off).ok_or_else(|| {
        RootError::Deserialization(format!("u64 read out of bounds at offset {}", off))
    })?;
    let b1 = *data.get(off + 1).ok_or_else(|| {
        RootError::Deserialization(format!("u64 read out of bounds at offset {}", off))
    })?;
    let b2 = *data.get(off + 2).ok_or_else(|| {
        RootError::Deserialization(format!("u64 read out of bounds at offset {}", off))
    })?;
    let b3 = *data.get(off + 3).ok_or_else(|| {
        RootError::Deserialization(format!("u64 read out of bounds at offset {}", off))
    })?;
    let b4 = *data.get(off + 4).ok_or_else(|| {
        RootError::Deserialization(format!("u64 read out of bounds at offset {}", off))
    })?;
    let b5 = *data.get(off + 5).ok_or_else(|| {
        RootError::Deserialization(format!("u64 read out of bounds at offset {}", off))
    })?;
    let b6 = *data.get(off + 6).ok_or_else(|| {
        RootError::Deserialization(format!("u64 read out of bounds at offset {}", off))
    })?;
    let b7 = *data.get(off + 7).ok_or_else(|| {
        RootError::Deserialization(format!("u64 read out of bounds at offset {}", off))
    })?;
    Ok(u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]))
}

fn le_i32(data: &[u8], off: usize) -> Result<i32> {
    Ok(le_u32(data, off)? as i32)
}

fn le_i64(data: &[u8], off: usize) -> Result<i64> {
    Ok(le_u64(data, off)? as i64)
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
    fn parse_footer_field_tokens_smoke() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0u8; 4]);
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.extend_from_slice(b"n");
        payload.extend_from_slice(&(12u32).to_le_bytes());
        payload.extend_from_slice(b"std::int32_t");
        payload.extend_from_slice(&0u32.to_le_bytes());
        let total_len = 4 + payload.len() + 8;
        let mut env = Vec::new();
        env.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_FOOTER.to_le_bytes());
        env.extend_from_slice(&(total_len as u16).to_le_bytes());
        env.extend_from_slice(&payload);
        env.extend_from_slice(&0u64.to_le_bytes());

        let tokens = parse_rntuple_footer_field_tokens(&env).expect("footer tokens parse failed");
        assert!(tokens.iter().any(|t| t.name == "n" && t.type_name == "std::int32_t"));
    }

    #[test]
    fn schema_summary_with_footer_merges_new_field_tokens() {
        let header = RNTupleHeaderSummary {
            ntuple_name: Some("Events".to_string()),
            writer: Some("ROOT v6.38.00".to_string()),
            strings: vec![],
            field_tokens: vec![RNTupleFieldToken {
                name: "pt".to_string(),
                type_name: "float".to_string(),
            }],
        };

        let mut payload = Vec::new();
        payload.extend_from_slice(&[0u8; 4]);
        payload.extend_from_slice(&(1u32).to_le_bytes());
        payload.extend_from_slice(b"n");
        payload.extend_from_slice(&(12u32).to_le_bytes());
        payload.extend_from_slice(b"std::int32_t");
        payload.extend_from_slice(&0u32.to_le_bytes());
        let total_len = 4 + payload.len() + 8;
        let mut env = Vec::new();
        env.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_FOOTER.to_le_bytes());
        env.extend_from_slice(&(total_len as u16).to_le_bytes());
        env.extend_from_slice(&payload);
        env.extend_from_slice(&0u64.to_le_bytes());

        let schema = parse_rntuple_schema_summary_with_footer(&header, &env)
            .expect("schema merge from footer failed");
        assert!(schema.fields.iter().any(|f| f.name == "pt"));
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

    #[test]
    fn parse_footer_summary_smoke() {
        let mut env = Vec::new();
        env.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_FOOTER.to_le_bytes());
        env.extend_from_slice(&(160u16).to_le_bytes());

        let mut payload = vec![0u8; 148];
        let header_hash = 0x1122_3344_5566_7788u64;
        payload[12..20].copy_from_slice(&header_hash.to_le_bytes());

        let list_off = 88usize;
        payload[list_off..list_off + 4].copy_from_slice(&1u32.to_le_bytes()); // n_items
        payload[list_off + 4..list_off + 8].copy_from_slice(&48u32.to_le_bytes()); // frame_len

        let item_off = list_off + 8;
        payload[item_off..item_off + 4].copy_from_slice(&0u32.to_le_bytes()); // record-frame preamble
        payload[item_off + 4..item_off + 12].copy_from_slice(&0u64.to_le_bytes()); // min_entry
        payload[item_off + 12..item_off + 20].copy_from_slice(&10u64.to_le_bytes()); // span
        payload[item_off + 20..item_off + 24].copy_from_slice(&2u32.to_le_bytes()); // n_clusters
        payload[item_off + 24..item_off + 32].copy_from_slice(&300u64.to_le_bytes()); // page list len
        payload[item_off + 32..item_off + 36].copy_from_slice(&120u32.to_le_bytes()); // nbytes
        payload[item_off + 36..item_off + 40].copy_from_slice(&0x89abu32.to_le_bytes()); // pos lo
        payload[item_off + 40..item_off + 44].copy_from_slice(&0x0012u32.to_le_bytes()); // pos hi

        env.extend_from_slice(&payload);
        env.extend_from_slice(&0u64.to_le_bytes());

        let summary = parse_rntuple_footer_summary(&env).expect("footer summary parse failed");
        assert_eq!(summary.header_xxhash3_le, header_hash);
        assert_eq!(summary.cluster_groups.len(), 1);
        let cg = &summary.cluster_groups[0];
        assert_eq!(cg.min_entry, 0);
        assert_eq!(cg.entry_span, 10);
        assert_eq!(cg.n_clusters, 2);
        assert_eq!(cg.page_list_envelope_len, 300);
        assert_eq!(cg.page_list_locator.nbytes_on_storage, 120);
        assert_eq!(cg.page_list_locator.position, 0x0012_0000_89ab);
    }

    #[test]
    fn parse_cluster_groups_frame_without_record_preamble() {
        let mut payload = vec![0u8; 8 + 8 + 40];
        let list_off = 8usize;
        payload[list_off..list_off + 4].copy_from_slice(&1u32.to_le_bytes());
        payload[list_off + 4..list_off + 8].copy_from_slice(&40u32.to_le_bytes());

        let item_off = list_off + 8;
        payload[item_off..item_off + 8].copy_from_slice(&0u64.to_le_bytes()); // min_entry
        payload[item_off + 8..item_off + 16].copy_from_slice(&5u64.to_le_bytes()); // span
        payload[item_off + 16..item_off + 20].copy_from_slice(&1u32.to_le_bytes()); // n_clusters
        payload[item_off + 20..item_off + 28].copy_from_slice(&180u64.to_le_bytes()); // env len
        payload[item_off + 28..item_off + 32].copy_from_slice(&88u32.to_le_bytes()); // nbytes
        payload[item_off + 32..item_off + 36].copy_from_slice(&0x4567u32.to_le_bytes()); // pos lo
        payload[item_off + 36..item_off + 40].copy_from_slice(&0x0001u32.to_le_bytes()); // pos hi

        let groups = parse_cluster_groups(&payload).expect("frame variant should parse");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].min_entry, 0);
        assert_eq!(groups[0].entry_span, 5);
        assert_eq!(groups[0].n_clusters, 1);
        assert_eq!(groups[0].page_list_envelope_len, 180);
        assert_eq!(groups[0].page_list_locator.nbytes_on_storage, 88);
        assert_eq!(groups[0].page_list_locator.position, 0x0001_0000_4567);
    }

    #[test]
    fn parse_cluster_groups_scan_fallback_smoke() {
        let mut payload = vec![0xaau8, 0xbbu8, 0xccu8];
        payload.resize(3 + 80, 0u8);
        let rec0 = 3usize;
        payload[rec0..rec0 + 8].copy_from_slice(&0u64.to_le_bytes()); // min_entry
        payload[rec0 + 8..rec0 + 16].copy_from_slice(&2u64.to_le_bytes()); // span
        payload[rec0 + 16..rec0 + 20].copy_from_slice(&1u32.to_le_bytes()); // n_clusters
        payload[rec0 + 20..rec0 + 28].copy_from_slice(&120u64.to_le_bytes()); // env len
        payload[rec0 + 28..rec0 + 32].copy_from_slice(&48u32.to_le_bytes()); // nbytes
        payload[rec0 + 32..rec0 + 36].copy_from_slice(&0x3456u32.to_le_bytes()); // pos lo
        payload[rec0 + 36..rec0 + 40].copy_from_slice(&0u32.to_le_bytes()); // pos hi

        let rec1 = rec0 + 40;
        payload[rec1..rec1 + 8].copy_from_slice(&2u64.to_le_bytes()); // min_entry
        payload[rec1 + 8..rec1 + 16].copy_from_slice(&3u64.to_le_bytes()); // span
        payload[rec1 + 16..rec1 + 20].copy_from_slice(&1u32.to_le_bytes()); // n_clusters
        payload[rec1 + 20..rec1 + 28].copy_from_slice(&140u64.to_le_bytes()); // env len
        payload[rec1 + 28..rec1 + 32].copy_from_slice(&56u32.to_le_bytes()); // nbytes
        payload[rec1 + 32..rec1 + 36].copy_from_slice(&0x789au32.to_le_bytes()); // pos lo
        payload[rec1 + 36..rec1 + 40].copy_from_slice(&0u32.to_le_bytes()); // pos hi

        let groups = parse_cluster_groups(&payload).expect("scan fallback should parse");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].min_entry, 0);
        assert_eq!(groups[0].entry_span, 2);
        assert_eq!(groups[1].min_entry, 2);
        assert_eq!(groups[1].entry_span, 3);
        assert_eq!(groups[0].page_list_locator.position, 0x3456);
        assert_eq!(groups[1].page_list_locator.position, 0x789a);
    }

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }

    #[test]
    fn parse_footer_random_malformed_envelopes_do_not_panic() {
        for seed in 0u64..128 {
            let mut rng = seed.wrapping_add(1);
            let payload_len = (lcg_next(&mut rng) % 160) as usize;
            let mut payload = vec![0u8; payload_len];
            for b in &mut payload {
                *b = (lcg_next(&mut rng) & 0xff) as u8;
            }
            let total_len = 4 + payload_len + 8;
            if total_len > u16::MAX as usize {
                continue;
            }

            let mut env = Vec::with_capacity(total_len);
            env.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_FOOTER.to_le_bytes());
            env.extend_from_slice(&(total_len as u16).to_le_bytes());
            env.extend_from_slice(&payload);
            env.extend_from_slice(&lcg_next(&mut rng).to_le_bytes());

            let _ = parse_rntuple_footer_summary(&env);
            let _ = parse_rntuple_footer_field_tokens(&env);
        }
    }

    #[test]
    fn parse_pagelist_random_malformed_envelopes_do_not_panic() {
        for seed in 0u64..128 {
            let mut rng = seed.wrapping_add(17);
            let payload_len = (lcg_next(&mut rng) % 192) as usize;
            let mut payload = vec![0u8; payload_len];
            for b in &mut payload {
                *b = (lcg_next(&mut rng) & 0xff) as u8;
            }
            let total_len = 4 + payload_len + 8;
            if total_len > u16::MAX as usize {
                continue;
            }

            let mut env = Vec::with_capacity(total_len);
            env.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_PAGELIST.to_le_bytes());
            env.extend_from_slice(&(total_len as u16).to_le_bytes());
            env.extend_from_slice(&payload);
            env.extend_from_slice(&lcg_next(&mut rng).to_le_bytes());

            let _ = parse_rntuple_pagelist_summary(&env);
        }
    }

    #[test]
    fn parse_pagelist_summary_smoke() {
        let mut env = Vec::new();
        env.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_PAGELIST.to_le_bytes());
        env.extend_from_slice(&(164u16).to_le_bytes());

        // 152-byte payload = 72-byte preamble + 2*40-byte records.
        let mut payload = vec![0u8; 152];
        let footer_hash = 0x1122_3344_5566_7788u64;
        payload[4..12].copy_from_slice(&footer_hash.to_le_bytes());

        // record[0]
        let rec0 = 72usize;
        payload[rec0..rec0 + 8].copy_from_slice(&(-40i64).to_le_bytes());
        payload[rec0 + 8..rec0 + 12].copy_from_slice(&1u32.to_le_bytes());
        payload[rec0 + 12..rec0 + 16].copy_from_slice(&(-8i32).to_le_bytes());
        payload[rec0 + 16..rec0 + 20].copy_from_slice(&32u32.to_le_bytes());
        payload[rec0 + 20..rec0 + 24].copy_from_slice(&0x1234u32.to_le_bytes());
        payload[rec0 + 24..rec0 + 28].copy_from_slice(&0x0001u32.to_le_bytes());
        payload[rec0 + 28..rec0 + 32].copy_from_slice(&2u32.to_le_bytes());
        payload[rec0 + 32..rec0 + 36].copy_from_slice(&3u32.to_le_bytes());
        payload[rec0 + 36..rec0 + 40].copy_from_slice(&0x55u32.to_le_bytes());

        // record[1]
        let rec1 = rec0 + 40;
        payload[rec1..rec1 + 8].copy_from_slice(&(-40i64).to_le_bytes());
        payload[rec1 + 8..rec1 + 12].copy_from_slice(&1u32.to_le_bytes());
        payload[rec1 + 12..rec1 + 16].copy_from_slice(&(-4i32).to_le_bytes());
        payload[rec1 + 16..rec1 + 20].copy_from_slice(&16u32.to_le_bytes());
        payload[rec1 + 20..rec1 + 24].copy_from_slice(&0x00abu32.to_le_bytes());
        payload[rec1 + 24..rec1 + 28].copy_from_slice(&0u32.to_le_bytes());
        payload[rec1 + 28..rec1 + 32].copy_from_slice(&0u32.to_le_bytes());
        payload[rec1 + 32..rec1 + 36].copy_from_slice(&0u32.to_le_bytes());
        payload[rec1 + 36..rec1 + 40].copy_from_slice(&0x77u32.to_le_bytes());

        env.extend_from_slice(&payload);
        env.extend_from_slice(&0u64.to_le_bytes());

        let summary = parse_rntuple_pagelist_summary(&env).expect("pagelist summary parse failed");
        assert_eq!(summary.footer_xxhash3_le, footer_hash);
        assert_eq!(summary.pages.len(), 2);
        assert_eq!(summary.pages[0].record_tag_raw, -40);
        assert_eq!(summary.pages[0].repetition_raw, 1);
        assert_eq!(summary.pages[0].element_count_raw, -8);
        assert_eq!(summary.pages[0].nbytes_on_storage, 32);
        assert_eq!(summary.pages[0].position, 0x0001_0000_1234);
        assert_eq!(summary.pages[0].locator_type_raw, 2);
        assert_eq!(summary.pages[0].locator_reserved_raw, 3);
        assert_eq!(summary.pages[0].cluster_anchor_raw, 0x55);

        assert_eq!(summary.pages[1].element_count_raw, -4);
        assert_eq!(summary.pages[1].nbytes_on_storage, 16);
        assert_eq!(summary.pages[1].position, 0xabu64);
        assert_eq!(summary.pages[1].cluster_anchor_raw, 0x77);
    }

    #[test]
    fn parse_pagelist_summary_variable_preamble_smoke() {
        let mut env = Vec::new();
        env.extend_from_slice(&RNTUPLE_ENVELOPE_TYPE_PAGELIST.to_le_bytes());
        env.extend_from_slice(&(176u16).to_le_bytes());

        // 164-byte payload = 84-byte preamble + 2*40-byte records.
        let mut payload = vec![0u8; 164];
        let footer_hash = 0x9988_7766_5544_3322u64;
        payload[4..12].copy_from_slice(&footer_hash.to_le_bytes());

        // record[0]
        let rec0 = 84usize;
        payload[rec0..rec0 + 8].copy_from_slice(&(-40i64).to_le_bytes());
        payload[rec0 + 8..rec0 + 12].copy_from_slice(&1u32.to_le_bytes());
        payload[rec0 + 12..rec0 + 16].copy_from_slice(&(-32i32).to_le_bytes());
        payload[rec0 + 16..rec0 + 20].copy_from_slice(&128u32.to_le_bytes());
        payload[rec0 + 20..rec0 + 24].copy_from_slice(&0x0f0fu32.to_le_bytes());
        payload[rec0 + 24..rec0 + 28].copy_from_slice(&0x0002u32.to_le_bytes());
        payload[rec0 + 28..rec0 + 32].copy_from_slice(&1u32.to_le_bytes());
        payload[rec0 + 32..rec0 + 36].copy_from_slice(&0u32.to_le_bytes());
        payload[rec0 + 36..rec0 + 40].copy_from_slice(&0x11u32.to_le_bytes());

        // record[1]
        let rec1 = rec0 + 40;
        payload[rec1..rec1 + 8].copy_from_slice(&(-40i64).to_le_bytes());
        payload[rec1 + 8..rec1 + 12].copy_from_slice(&1u32.to_le_bytes());
        payload[rec1 + 12..rec1 + 16].copy_from_slice(&(-16i32).to_le_bytes());
        payload[rec1 + 16..rec1 + 20].copy_from_slice(&64u32.to_le_bytes());
        payload[rec1 + 20..rec1 + 24].copy_from_slice(&0x0ff0u32.to_le_bytes());
        payload[rec1 + 24..rec1 + 28].copy_from_slice(&0u32.to_le_bytes());
        payload[rec1 + 28..rec1 + 32].copy_from_slice(&1u32.to_le_bytes());
        payload[rec1 + 32..rec1 + 36].copy_from_slice(&0u32.to_le_bytes());
        payload[rec1 + 36..rec1 + 40].copy_from_slice(&0x22u32.to_le_bytes());

        env.extend_from_slice(&payload);
        env.extend_from_slice(&0u64.to_le_bytes());

        let summary = parse_rntuple_pagelist_summary(&env).expect("pagelist summary parse failed");
        assert_eq!(summary.footer_xxhash3_le, footer_hash);
        assert_eq!(summary.pages.len(), 2);
        assert_eq!(summary.pages[0].nbytes_on_storage, 128);
        assert_eq!(summary.pages[1].nbytes_on_storage, 64);
        assert_eq!(summary.pages[0].position, 0x0002_0000_0f0f);
        assert_eq!(summary.pages[1].position, 0x0ff0u64);
    }
}
