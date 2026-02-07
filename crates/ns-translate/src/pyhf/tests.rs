//! Tests for pyhf parser

use super::schema::*;

#[test]
fn test_parse_simple_workspace() {
    let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
    let ws: Workspace = serde_json::from_str(json).expect("Failed to parse simple_workspace.json");

    // Check channels
    assert_eq!(ws.channels.len(), 1);
    assert_eq!(ws.channels[0].name, "singlechannel");
    assert_eq!(ws.channels[0].samples.len(), 2);

    // Check signal sample
    let signal = &ws.channels[0].samples[0];
    assert_eq!(signal.name, "signal");
    assert_eq!(signal.data, vec![5.0, 10.0]);
    assert_eq!(signal.modifiers.len(), 1);

    // Check background sample
    let background = &ws.channels[0].samples[1];
    assert_eq!(background.name, "background");
    assert_eq!(background.data, vec![50.0, 60.0]);

    // Check observations
    assert_eq!(ws.observations.len(), 1);
    assert_eq!(ws.observations[0].name, "singlechannel");
    assert_eq!(ws.observations[0].data, vec![53.0, 65.0]);

    // Check measurements
    assert_eq!(ws.measurements.len(), 1);
    assert_eq!(ws.measurements[0].name, "GaussExample");
    assert_eq!(ws.measurements[0].config.poi, "mu");
}

#[test]
fn test_parse_complex_workspace() {
    let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
    let ws: Workspace = serde_json::from_str(json).expect("Failed to parse complex_workspace.json");

    // Check channels
    assert_eq!(ws.channels.len(), 2);
    assert!(ws.channels.iter().any(|c| c.name == "SR"));
    assert!(ws.channels.iter().any(|c| c.name == "CR"));

    // Check observations match channels
    assert_eq!(ws.observations.len(), 2);
    assert!(ws.observations.iter().any(|o| o.name == "SR"));
    assert!(ws.observations.iter().any(|o| o.name == "CR"));
}

#[test]
fn test_parse_all_modifier_types() {
    let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    let mut found_modifiers = std::collections::HashSet::new();

    for channel in &ws.channels {
        for sample in &channel.samples {
            for modifier in &sample.modifiers {
                let mod_type = match modifier {
                    Modifier::NormFactor { .. } => "normfactor",
                    Modifier::NormSys { .. } => "normsys",
                    Modifier::HistoSys { .. } => "histosys",
                    Modifier::ShapeSys { .. } => "shapesys",
                    Modifier::ShapeFactor { .. } => "shapefactor",
                    Modifier::StatError { .. } => "staterror",
                    Modifier::Lumi { .. } => "lumi",
                };
                found_modifiers.insert(mod_type);
            }
        }
    }

    // Verify we have multiple modifier types
    assert!(found_modifiers.contains("normfactor"));
    assert!(found_modifiers.contains("lumi"));
    assert!(found_modifiers.contains("normsys"));
    assert!(found_modifiers.contains("histosys"));
    assert!(found_modifiers.contains("staterror"));
    assert!(found_modifiers.contains("shapefactor"));
}

#[test]
fn test_serde_roundtrip() {
    let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    // Serialize back to JSON
    let serialized = serde_json::to_string_pretty(&ws).unwrap();

    // Parse again
    let ws2: Workspace = serde_json::from_str(&serialized).unwrap();

    // Check basic properties are preserved
    assert_eq!(ws.channels.len(), ws2.channels.len());
    assert_eq!(ws.observations.len(), ws2.observations.len());
    assert_eq!(ws.measurements.len(), ws2.measurements.len());
}

#[test]
fn test_parameter_config() {
    let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    // Check lumi parameter config
    let params = &ws.measurements[0].config.parameters;
    assert!(!params.is_empty());

    let lumi_param = params.iter().find(|p| p.name == "lumi");
    assert!(lumi_param.is_some());

    let lumi = lumi_param.unwrap();
    assert_eq!(lumi.inits, vec![1.0]);
    assert_eq!(lumi.auxdata, vec![1.0]);
    assert_eq!(lumi.sigmas, vec![0.02]);
}

#[test]
fn test_parameter_fixed_is_parsed_and_applied_as_bounds_clamp() {
    use ns_core::traits::LogDensityModel;

    let json = r#"
{
  "channels": [
    {
      "name": "ch",
      "samples": [
        {
          "name": "s",
          "data": [1.0],
          "modifiers": [
            {"name": "lumi", "type": "lumi", "data": null}
          ]
        }
      ]
    }
  ],
  "observations": [{"name": "ch", "data": [1.0]}],
  "measurements": [
    {
      "name": "meas",
      "config": {
        "poi": "mu",
        "parameters": [
          {
            "name": "lumi",
            "inits": [1.0],
            "bounds": [[0.5, 1.5]],
            "auxdata": [1.0],
            "sigmas": [0.1],
            "fixed": true
          }
        ]
      }
    }
  ],
  "version": "1.0.0"
}
"#;

    let ws: Workspace = serde_json::from_str(json).expect("parse workspace");
    let cfg = ws
        .measurements
        .first()
        .and_then(|m| m.config.parameters.first())
        .expect("parameter config present");
    assert!(cfg.fixed, "expected fixed=true");

    let model = super::HistFactoryModel::from_workspace(&ws).expect("build model");
    let names = model.parameter_names();
    let bounds = model.parameter_bounds();
    let lumi_idx = names.iter().position(|n| n == "lumi").expect("lumi param exists");
    assert_eq!(bounds[lumi_idx], (1.0, 1.0));
}

#[test]
fn test_model_rejects_observation_length_mismatch() {
    let json = include_str!("../../../../tests/fixtures/bad_observations_length_mismatch.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    let err = super::HistFactoryModel::from_workspace(&ws).unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("observations") && (msg.contains("length") || msg.contains("mismatch")),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_model_rejects_sample_length_mismatch() {
    let json = include_str!("../../../../tests/fixtures/bad_sample_length_mismatch.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    let err = super::HistFactoryModel::from_workspace(&ws).unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("sample") && (msg.contains("length") || msg.contains("mismatch")),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_model_rejects_histosys_template_length_mismatch() {
    let json =
        include_str!("../../../../tests/fixtures/bad_histosys_template_length_mismatch.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();

    let err = super::HistFactoryModel::from_workspace(&ws).unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("histosys") && (msg.contains("length") || msg.contains("mismatch")),
        "unexpected error: {}",
        msg
    );
}
