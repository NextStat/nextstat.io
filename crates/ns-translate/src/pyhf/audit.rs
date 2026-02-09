//! Workspace audit: inspect pyhf JSON compatibility before loading.
//!
//! Works on raw `serde_json::Value` so it succeeds even when typed parsing would fail
//! (e.g. unknown modifier types).

use serde::Serialize;
use std::collections::HashMap;

/// Known (supported) modifier types.
pub const KNOWN_MODIFIER_TYPES: &[&str] =
    &["normfactor", "normsys", "histosys", "shapesys", "shapefactor", "staterror", "lumi"];

/// Summary of a measurement block.
#[derive(Debug, Clone, Serialize)]
pub struct MeasurementInfo {
    pub name: String,
    pub poi: String,
    pub n_fixed_params: usize,
}

/// Result of auditing a pyhf workspace JSON.
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceAudit {
    /// Measurements found in the workspace.
    pub measurements: Vec<MeasurementInfo>,
    /// Modifier type → occurrence count.
    pub modifier_types_used: HashMap<String, usize>,
    /// Human-readable warnings about unsupported features.
    pub unsupported_features: Vec<String>,
    /// Number of channels.
    pub channel_count: usize,
    /// Total bins across all channels.
    pub total_bins: usize,
    /// Total samples across all channels.
    pub total_samples: usize,
    /// Total modifier instances across all samples.
    pub total_modifiers: usize,
    /// Estimated parameter count (rough, may differ from actual model).
    pub parameter_count_estimate: usize,
}

/// Audit a raw pyhf workspace JSON value.
///
/// This works even if the JSON contains modifier types that would fail typed deserialization.
pub fn workspace_audit(json: &serde_json::Value) -> WorkspaceAudit {
    let mut audit = WorkspaceAudit {
        measurements: Vec::new(),
        modifier_types_used: HashMap::new(),
        unsupported_features: Vec::new(),
        channel_count: 0,
        total_bins: 0,
        total_samples: 0,
        total_modifiers: 0,
        parameter_count_estimate: 0,
    };

    // --- Channels ---
    let mut param_names = std::collections::HashSet::new();

    if let Some(channels) = json.get("channels").and_then(|v| v.as_array()) {
        audit.channel_count = channels.len();

        for channel in channels {
            if let Some(samples) = channel.get("samples").and_then(|v| v.as_array()) {
                for sample in samples {
                    audit.total_samples += 1;

                    // Count bins from the sample data array
                    if let Some(data) = sample.get("data").and_then(|v| v.as_array()) {
                        // Only count once per channel (all samples should have the same bin count)
                        // We'll count from the first sample of each channel below.
                        let _ = data.len();
                    }

                    // Count modifiers
                    if let Some(modifiers) = sample.get("modifiers").and_then(|v| v.as_array()) {
                        for modifier in modifiers {
                            audit.total_modifiers += 1;

                            let mod_type =
                                modifier.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
                            let mod_name =
                                modifier.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed");

                            *audit.modifier_types_used.entry(mod_type.to_string()).or_insert(0) +=
                                1;

                            if !KNOWN_MODIFIER_TYPES.contains(&mod_type) {
                                let msg = format!(
                                    "unsupported modifier type: \"{}\" (name: \"{}\")",
                                    mod_type, mod_name
                                );
                                if !audit.unsupported_features.contains(&msg) {
                                    audit.unsupported_features.push(msg);
                                }
                            }

                            // Track parameter names for estimate
                            param_names.insert(mod_name.to_string());
                        }
                    }
                }
            }

            // Count bins from the first sample of each channel
            if let Some(data) = channel
                .get("samples")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|s| s.get("data"))
                .and_then(|v| v.as_array())
            {
                audit.total_bins += data.len();
            }
        }
    }

    // --- Measurements ---
    if let Some(measurements) = json.get("measurements").and_then(|v| v.as_array()) {
        for meas in measurements {
            let name = meas.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed").to_string();
            let poi = meas
                .get("config")
                .and_then(|c| c.get("poi"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let n_fixed = meas
                .get("config")
                .and_then(|c| c.get("parameters"))
                .and_then(|v| v.as_array())
                .map(|params| {
                    params
                        .iter()
                        .filter(|p| p.get("fixed") == Some(&serde_json::json!(true)))
                        .count()
                })
                .unwrap_or(0);

            audit.measurements.push(MeasurementInfo { name, poi, n_fixed_params: n_fixed });
        }

        if measurements.len() > 1 {
            audit.unsupported_features.push(format!(
                "multiple measurements ({}) — only the first is used by default",
                measurements.len()
            ));
        }
    } else {
        audit.unsupported_features.push("no measurements block found".to_string());
    }

    // --- Observations ---
    if json.get("observations").and_then(|v| v.as_array()).is_none() {
        audit.unsupported_features.push("no observations block found".to_string());
    }

    // --- Parameter count estimate ---
    // This is approximate: unique modifier names + per-bin params for shapesys/staterror
    audit.parameter_count_estimate = param_names.len();

    audit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_simple_workspace() {
        let json_str = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let json: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let audit = workspace_audit(&json);

        assert_eq!(audit.channel_count, 1);
        assert!(audit.total_bins > 0);
        assert!(audit.total_samples > 0);
        assert!(audit.total_modifiers > 0);
        assert!(
            audit.unsupported_features.is_empty(),
            "simple workspace should have no unsupported features: {:?}",
            audit.unsupported_features
        );
        assert!(audit.parameter_count_estimate > 0);
        assert!(!audit.measurements.is_empty());

        // Should have known modifier types
        for typ in audit.modifier_types_used.keys() {
            assert!(
                KNOWN_MODIFIER_TYPES.contains(&typ.as_str()),
                "unexpected modifier type: {}",
                typ
            );
        }

        println!("Audit: {}", serde_json::to_string_pretty(&audit).unwrap());
    }

    #[test]
    fn test_audit_unknown_modifier() {
        let json: serde_json::Value = serde_json::json!({
            "channels": [{
                "name": "test",
                "samples": [{
                    "name": "sig",
                    "data": [10.0, 20.0],
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": null},
                        {"name": "custom", "type": "custom_mod", "data": {}}
                    ]
                }]
            }],
            "observations": [{"name": "test", "data": [12.0, 18.0]}],
            "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}]
        });

        let audit = workspace_audit(&json);

        assert_eq!(audit.channel_count, 1);
        assert_eq!(audit.total_bins, 2);
        assert_eq!(audit.total_samples, 1);
        assert_eq!(audit.total_modifiers, 2);
        assert!(!audit.unsupported_features.is_empty());
        assert!(audit.unsupported_features.iter().any(|s| s.contains("custom_mod")));

        println!("Unsupported: {:?}", audit.unsupported_features);
    }
}
