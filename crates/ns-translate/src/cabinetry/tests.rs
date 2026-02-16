use super::*;

const EXAMPLE_CONFIG: &str = r#"
General:
  Measurement: "minimal_example"
  POI: "Signal_norm"
  InputPath: "inputs/{SamplePath}"
  HistogramFolder: "histograms/"

Regions:
  - Name: "SR"
    Variable: "jet_pt"
    Binning: [200, 300, 400, 500]

Samples:
  - Name: "Data"
    Tree: "pseudodata"
    SamplePath: "data.root"
    Data: true

  - Name: "Signal"
    Tree: "signal"
    SamplePath: "prediction.root"
    Weight: "weight"

  - Name: "Background"
    Tree: "background"
    SamplePath: "prediction.root"
    Weight: "weight"

NormFactors:
  - Name: "Signal_norm"
    Samples: "Signal"
    Nominal: 1
    Bounds: [0, 10]

Systematics:
  - Name: "Luminosity"
    Up:
      Normalization: 0.05
    Down:
      Normalization: -0.05
    Type: Normalization

  - Name: "Modeling"
    Up:
      SamplePath: "prediction.root"
      Tree: "background_varied"
    Down:
      Symmetrize: true
    Samples: "Background"
    Type: NormPlusShape
"#;

#[test]
fn parse_config() {
    let config = parse_cabinetry_config(EXAMPLE_CONFIG).unwrap();
    assert_eq!(config.general.measurement, "minimal_example");
    assert_eq!(config.general.poi, "Signal_norm");
    assert_eq!(config.regions.len(), 1);
    assert_eq!(config.regions[0].name, "SR");
    assert_eq!(config.regions[0].binning.as_ref().unwrap(), &[200.0, 300.0, 400.0, 500.0]);
    assert_eq!(config.samples.len(), 3);
    assert!(config.samples[0].data);
    assert!(!config.samples[1].data);
    assert_eq!(config.norm_factors.len(), 1);
    assert_eq!(config.norm_factors[0].name, "Signal_norm");
    assert_eq!(config.norm_factors[0].bounds.unwrap(), [0.0, 10.0]);
    assert_eq!(config.systematics.len(), 2);
    assert_eq!(config.systematics[0].syst_type, SystematicType::Normalization);
    assert_eq!(config.systematics[1].syst_type, SystematicType::NormPlusShape);
}

#[test]
fn string_or_array_single() {
    let s: StringOrArray = serde_json::from_str(r#""Signal""#).unwrap();
    assert_eq!(s.to_vec(), vec!["Signal"]);
    assert!(s.contains("Signal"));
    assert!(!s.contains("Background"));
}

#[test]
fn string_or_array_array() {
    let s: StringOrArray = serde_json::from_str(r#"["Signal", "Background"]"#).unwrap();
    assert_eq!(s.to_vec(), vec!["Signal", "Background"]);
    assert!(s.contains("Background"));
}

#[test]
fn to_workspace_roundtrip() {
    let config = parse_cabinetry_config(EXAMPLE_CONFIG).unwrap();

    let histograms = vec![
        HistogramData {
            region: "SR".into(),
            sample: "Signal".into(),
            nominal: vec![10.0, 20.0, 5.0],
            stat_uncertainties: vec![],
            systematics: vec![],
        },
        HistogramData {
            region: "SR".into(),
            sample: "Background".into(),
            nominal: vec![100.0, 200.0, 50.0],
            stat_uncertainties: vec![10.0, 14.1, 7.1],
            systematics: vec![(
                "Modeling".into(),
                vec![110.0, 210.0, 55.0],
                vec![90.0, 190.0, 45.0],
            )],
        },
    ];

    let observations = vec![("SR".into(), vec![108.0, 215.0, 53.0])];

    let ws = to_workspace(&config, &histograms, &observations).unwrap();

    // Check structure.
    assert_eq!(ws.channels.len(), 1);
    assert_eq!(ws.channels[0].name, "SR");
    assert_eq!(ws.channels[0].samples.len(), 2); // Signal + Background (Data excluded)
    assert_eq!(ws.observations.len(), 1);
    assert_eq!(ws.measurements.len(), 1);
    assert_eq!(ws.measurements[0].config.poi, "Signal_norm");

    // Signal sample: normfactor + staterror.
    let sig = &ws.channels[0].samples[0];
    assert_eq!(sig.name, "Signal");
    assert_eq!(sig.data, vec![10.0, 20.0, 5.0]);
    assert!(sig.modifiers.iter().any(|m| matches!(m, crate::pyhf::schema::Modifier::NormFactor { name, .. } if name == "Signal_norm")));
    assert!(
        sig.modifiers.iter().any(|m| matches!(m, crate::pyhf::schema::Modifier::StatError { .. }))
    );

    // Background sample: lumi normsys + modeling histosys + staterror.
    let bkg = &ws.channels[0].samples[1];
    assert_eq!(bkg.name, "Background");
    assert!(bkg.modifiers.iter().any(
        |m| matches!(m, crate::pyhf::schema::Modifier::NormSys { name, .. } if name == "Luminosity")
    ));
    assert!(bkg.modifiers.iter().any(
        |m| matches!(m, crate::pyhf::schema::Modifier::HistoSys { name, .. } if name == "Modeling")
    ));

    // NormFactor config.
    let nf_pc = &ws.measurements[0].config.parameters[0];
    assert_eq!(nf_pc.name, "Signal_norm");
    assert_eq!(nf_pc.inits, vec![1.0]);
    assert_eq!(nf_pc.bounds, vec![[0.0, 10.0]]);

    // Workspace serializes to JSON (pyhf format).
    let json = serde_json::to_string_pretty(&ws).unwrap();
    assert!(json.contains("Signal_norm"));
    assert!(json.contains("\"SR\""));
}

#[test]
fn missing_histogram_error() {
    let config = parse_cabinetry_config(EXAMPLE_CONFIG).unwrap();
    let result = to_workspace(&config, &[], &[]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("missing histogram"));
}
