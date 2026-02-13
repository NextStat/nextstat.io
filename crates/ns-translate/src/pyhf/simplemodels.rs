//! Simple model builders for quick experiments and tutorials.
//!
//! Mirrors `pyhf.simplemodels`: convenience functions that construct a
//! [`Workspace`] (and optionally a [`HistFactoryModel`]) from minimal inputs.

use super::schema::{
    Channel, HistoSysData, Measurement, MeasurementConfig, Modifier, Observation, Sample, Workspace,
};

/// Build a workspace with a single channel, one signal sample and one background
/// sample whose per-bin uncertainties are **uncorrelated** (`shapesys` modifier).
///
/// Mirrors `pyhf.simplemodels.uncorrelated_background(signal, bkg, bkg_uncertainty)`.
///
/// # Arguments
/// * `signal` — expected signal yields per bin.
/// * `bkg` — expected background yields per bin.
/// * `bkg_uncertainty` — absolute per-bin background uncertainties (σ).
///
/// # Panics
/// Panics if the three slices have different lengths or are empty.
pub fn uncorrelated_background(signal: &[f64], bkg: &[f64], bkg_uncertainty: &[f64]) -> Workspace {
    assert!(!signal.is_empty(), "signal must not be empty");
    assert_eq!(signal.len(), bkg.len(), "signal and bkg must have the same length");
    assert_eq!(
        signal.len(),
        bkg_uncertainty.len(),
        "signal and bkg_uncertainty must have the same length"
    );

    Workspace {
        channels: vec![Channel {
            name: "singlechannel".to_string(),
            samples: vec![
                Sample {
                    name: "signal".to_string(),
                    data: signal.to_vec(),
                    modifiers: vec![Modifier::NormFactor { name: "mu".to_string(), data: None }],
                },
                Sample {
                    name: "background".to_string(),
                    data: bkg.to_vec(),
                    modifiers: vec![Modifier::ShapeSys {
                        name: "uncorr_bkguncrt".to_string(),
                        data: bkg_uncertainty.to_vec(),
                    }],
                },
            ],
        }],
        observations: vec![Observation {
            name: "singlechannel".to_string(),
            data: bkg.iter().zip(signal).map(|(b, s)| b + s).collect(),
        }],
        measurements: vec![Measurement {
            name: "Measurement".to_string(),
            config: MeasurementConfig { poi: "mu".to_string(), parameters: vec![] },
        }],
        version: Some("1.0.0".to_string()),
    }
}

/// Build a workspace with a single channel, one signal sample and one background
/// sample whose shape uncertainty is **correlated** across bins (`histosys` modifier).
///
/// Mirrors `pyhf.simplemodels.correlated_background(signal, bkg, bkg_up, bkg_down)`.
///
/// # Arguments
/// * `signal` — expected signal yields per bin.
/// * `bkg` — expected (nominal) background yields per bin.
/// * `bkg_up` — background template at +1σ.
/// * `bkg_down` — background template at −1σ.
///
/// # Panics
/// Panics if the four slices have different lengths or are empty.
pub fn correlated_background(
    signal: &[f64],
    bkg: &[f64],
    bkg_up: &[f64],
    bkg_down: &[f64],
) -> Workspace {
    assert!(!signal.is_empty(), "signal must not be empty");
    assert_eq!(signal.len(), bkg.len(), "signal and bkg must have the same length");
    assert_eq!(signal.len(), bkg_up.len(), "signal and bkg_up must have the same length");
    assert_eq!(signal.len(), bkg_down.len(), "signal and bkg_down must have the same length");

    Workspace {
        channels: vec![Channel {
            name: "singlechannel".to_string(),
            samples: vec![
                Sample {
                    name: "signal".to_string(),
                    data: signal.to_vec(),
                    modifiers: vec![Modifier::NormFactor { name: "mu".to_string(), data: None }],
                },
                Sample {
                    name: "background".to_string(),
                    data: bkg.to_vec(),
                    modifiers: vec![Modifier::HistoSys {
                        name: "corr_bkguncrt".to_string(),
                        data: HistoSysData { hi_data: bkg_up.to_vec(), lo_data: bkg_down.to_vec() },
                    }],
                },
            ],
        }],
        observations: vec![Observation {
            name: "singlechannel".to_string(),
            data: bkg.iter().zip(signal).map(|(b, s)| b + s).collect(),
        }],
        measurements: vec![Measurement {
            name: "Measurement".to_string(),
            config: MeasurementConfig { poi: "mu".to_string(), parameters: vec![] },
        }],
        version: Some("1.0.0".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyhf::HistFactoryModel;
    use ns_core::traits::LogDensityModel;

    #[test]
    fn test_uncorrelated_background_roundtrip() {
        let ws = uncorrelated_background(&[5.0, 10.0], &[50.0, 60.0], &[7.0, 8.0]);
        assert_eq!(ws.channels.len(), 1);
        assert_eq!(ws.channels[0].samples.len(), 2);
        assert_eq!(ws.observations[0].data, vec![55.0, 70.0]);

        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        assert!(model.poi_index().is_some());
        let exp = model.expected_data(&model.parameter_init()).unwrap();
        assert!(!exp.is_empty());
    }

    #[test]
    fn test_correlated_background_roundtrip() {
        let ws = correlated_background(&[5.0, 10.0], &[50.0, 60.0], &[55.0, 65.0], &[45.0, 55.0]);
        assert_eq!(ws.channels.len(), 1);
        assert_eq!(ws.channels[0].samples.len(), 2);

        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        assert!(model.poi_index().is_some());
        let exp = model.expected_data(&model.parameter_init()).unwrap();
        assert!(!exp.is_empty());
    }

    #[test]
    #[should_panic(expected = "signal must not be empty")]
    fn test_uncorrelated_empty_panics() {
        uncorrelated_background(&[], &[], &[]);
    }
}
