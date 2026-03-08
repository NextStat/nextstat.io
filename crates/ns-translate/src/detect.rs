//! Format auto-detection for NextStat JSON artifacts.
//!
//! Distinguishes pyhf, HS3, and simplified-likelihood JSON without requiring
//! callers to know the schema up front.

/// Detected JSON workspace format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceFormat {
    /// pyhf JSON (has `"channels"` + `"measurements"` at top level).
    Pyhf,
    /// HS3 JSON (has `"distributions"` + `"metadata"` with `"hs3_version"`).
    Hs3,
    /// NextStat simplified-likelihood JSON.
    SimplifiedLikelihood,
    /// Unknown format.
    Unknown,
}

/// Detect whether a JSON string is pyhf, HS3, or simplified-likelihood format.
///
/// Uses a cheap prefix scan first, then falls back to a full JSON parse if
/// the prefix scan is inconclusive.
pub fn detect_format(json: &str) -> WorkspaceFormat {
    let end = json.len().min(2000);
    let end = if end < json.len() {
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
    let has_simplified_schema = prefix.contains("\"nextstat_simplified_likelihood_v0\"");

    if has_distributions && has_hs3_version {
        return WorkspaceFormat::Hs3;
    }
    if has_channels && has_measurements {
        return WorkspaceFormat::Pyhf;
    }
    if has_simplified_schema {
        return WorkspaceFormat::SimplifiedLikelihood;
    }

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

        let is_simplified = obj
            .get("schema_version")
            .and_then(|v| v.as_str())
            .map(|v| v == "nextstat_simplified_likelihood_v0")
            .unwrap_or(false);
        if is_simplified {
            return WorkspaceFormat::SimplifiedLikelihood;
        }
    }

    WorkspaceFormat::Unknown
}
