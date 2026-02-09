//! Format auto-detection for JSON workspace files.
//!
//! Distinguishes HS3 from pyhf JSON without full deserialization by checking
//! for the presence of key top-level fields.

/// Detected JSON workspace format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceFormat {
    /// pyhf JSON (has `"channels"` + `"measurements"` at top level).
    Pyhf,
    /// HS3 JSON (has `"distributions"` + `"metadata"` with `"hs3_version"` at top level).
    Hs3,
    /// Unknown format.
    Unknown,
}

/// Detect whether a JSON string is pyhf or HS3 format.
///
/// Uses a two-tier heuristic: first scans the first ~2000 bytes for key
/// markers (zero-alloc), then falls back to a full `serde_json::Value` parse
/// if the prefix scan is inconclusive.
///
/// # Detection rules
///
/// - **HS3**: top-level object has both `"distributions"` and `"metadata"`
///   (with `"hs3_version"` sub-field).
/// - **pyhf**: top-level object has both `"channels"` and `"measurements"`.
/// - Otherwise: `Unknown`.
pub fn detect_format(json: &str) -> WorkspaceFormat {
    // Fast path: scan the first ~2000 chars for key markers without full parse.
    // This avoids deserializing a 17 MB file just for format detection.
    // Use char-boundary-safe slicing to avoid panic on multi-byte UTF-8.
    let end = json.len().min(2000);
    let end = if end < json.len() {
        // Walk backwards to find a valid char boundary
        let mut e = end;
        while e > 0 && !json.is_char_boundary(e) {
            e -= 1;
        }
        e
    } else {
        end
    };
    let prefix = &json[..end];

    let has_distributions = prefix.contains("\"distributions\"");
    let has_hs3_version = prefix.contains("\"hs3_version\"");
    let has_channels = prefix.contains("\"channels\"");
    let has_measurements = prefix.contains("\"measurements\"");

    // HS3 check: distributions + hs3_version (highly specific)
    if has_distributions && has_hs3_version {
        return WorkspaceFormat::Hs3;
    }

    // pyhf check: channels + measurements
    if has_channels && has_measurements {
        return WorkspaceFormat::Pyhf;
    }

    // Fallback: try full parse for larger files where keys appear later
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json)
        && let Some(obj) = value.as_object()
    {
        let is_hs3 = obj.contains_key("distributions")
            && obj.get("metadata").and_then(|m| m.get("hs3_version")).is_some();
        if is_hs3 {
            return WorkspaceFormat::Hs3;
        }

        let is_pyhf = obj.contains_key("channels") && obj.contains_key("measurements");
        if is_pyhf {
            return WorkspaceFormat::Pyhf;
        }
    }

    WorkspaceFormat::Unknown
}
