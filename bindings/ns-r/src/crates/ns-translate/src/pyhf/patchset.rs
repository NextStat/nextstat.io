//! pyhf PatchSet support (HEPData).
//!
//! A PatchSet is a collection of RFC 6902 JSON Patch operations that can be applied
//! to a base pyhf workspace JSON (typically a background-only workspace) to materialize
//! a signal+background workspace.

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A pyhf PatchSet (HEPData) container.
///
/// The common HEPData layout is:
/// - top-level `metadata`, `patches`, `version`
/// - each patch entry contains `metadata.name` and a `patch` array of JSON Patch ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchSet {
    /// Optional PatchSet metadata blob.
    #[serde(default)]
    pub metadata: Value,
    /// Patch entries.
    pub patches: Vec<PatchEntry>,
    /// Optional version string.
    #[serde(default)]
    pub version: Option<String>,
}

/// One named patch inside a PatchSet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchEntry {
    /// Patch metadata (name, parameters, etc.).
    pub metadata: PatchMetadata,
    /// RFC 6902 JSON Patch operations.
    pub patch: json_patch::Patch,
}

/// Patch metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchMetadata {
    /// Patch name (used to select which patch to apply).
    pub name: String,
    /// Optional patch parameters (HEPData typically stores a heterogeneous list).
    #[serde(default)]
    pub values: Vec<Value>,
}

impl PatchSet {
    /// Return patch names in the order they appear in the PatchSet.
    pub fn patch_names(&self) -> Vec<&str> {
        self.patches.iter().map(|p| p.metadata.name.as_str()).collect()
    }

    /// Find a patch entry by name.
    pub fn patch_by_name(&self, name: &str) -> Option<&PatchEntry> {
        self.patches.iter().find(|p| p.metadata.name == name)
    }

    fn select_patch(&self, patch_name: Option<&str>) -> Result<&PatchEntry> {
        if self.patches.is_empty() {
            return Err(Error::Validation("PatchSet contains no patches".to_string()));
        }

        if let Some(name) = patch_name {
            return self.patch_by_name(name).ok_or_else(|| {
                let mut names = self.patch_names();
                names.sort();
                Error::Validation(format!(
                    "Unknown PatchSet patch name '{name}'. Available: {}",
                    names.join(", ")
                ))
            });
        }

        Ok(&self.patches[0])
    }

    /// Apply a patch to a base JSON document (workspace JSON).
    pub fn apply_to_value(&self, base: &Value, patch_name: Option<&str>) -> Result<Value> {
        let entry = self.select_patch(patch_name)?;
        let mut out = base.clone();
        json_patch::patch(&mut out, &entry.patch).map_err(|e| {
            Error::Validation(format!(
                "Failed to apply PatchSet patch '{}': {e}",
                entry.metadata.name
            ))
        })?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyhf::Workspace;

    #[test]
    fn apply_patchset_add_and_replace() {
        let base_ws: Workspace =
            serde_json::from_str(include_str!("../../../../tests/fixtures/simple_workspace.json"))
                .expect("parse base workspace");
        let base_json = serde_json::to_value(&base_ws).expect("base to Value");

        let patchset_json = serde_json::json!({
            "metadata": {"description": "test patchset"},
            "version": "1.0.0",
            "patches": [
                {
                    "metadata": {"name": "p1", "values": []},
                    "patch": [
                        {"op": "add", "path": "/channels/0/samples/2", "value": {"name": "new_sample", "data": [1.0, 2.0], "modifiers": []}},
                        {"op": "replace", "path": "/observations/0/data/0", "value": 999.0}
                    ]
                },
                {
                    "metadata": {"name": "p2", "values": []},
                    "patch": [
                        {"op": "replace", "path": "/measurements/0/config/poi", "value": "mu_alt"}
                    ]
                }
            ]
        });
        let ps: PatchSet = serde_json::from_value(patchset_json).expect("parse PatchSet");

        let patched1 = ps.apply_to_value(&base_json, Some("p1")).expect("apply p1");
        let ws1: Workspace = serde_json::from_value(patched1).expect("patched workspace parses");
        assert_eq!(ws1.channels[0].samples.len(), 3);
        assert_eq!(ws1.channels[0].samples[2].name, "new_sample");
        assert_eq!(ws1.observations[0].data[0], 999.0);

        let patched2 = ps.apply_to_value(&base_json, Some("p2")).expect("apply p2");
        let ws2: Workspace = serde_json::from_value(patched2).expect("patched workspace parses");
        assert_eq!(ws2.measurements[0].config.poi, "mu_alt");
    }

    #[test]
    fn apply_patchset_defaults_to_first_patch() {
        let base_ws: Workspace =
            serde_json::from_str(include_str!("../../../../tests/fixtures/simple_workspace.json"))
                .expect("parse base workspace");
        let base_json = serde_json::to_value(&base_ws).expect("base to Value");

        let patchset_json = serde_json::json!({
            "patches": [
                {"metadata": {"name": "first", "values": []}, "patch": [{"op": "replace", "path": "/measurements/0/config/poi", "value": "first"}]},
                {"metadata": {"name": "second", "values": []}, "patch": [{"op": "replace", "path": "/measurements/0/config/poi", "value": "second"}]}
            ]
        });
        let ps: PatchSet = serde_json::from_value(patchset_json).expect("parse PatchSet");

        let patched = ps.apply_to_value(&base_json, None).expect("apply default");
        let ws: Workspace = serde_json::from_value(patched).expect("patched workspace parses");
        assert_eq!(ws.measurements[0].config.poi, "first");
    }
}
