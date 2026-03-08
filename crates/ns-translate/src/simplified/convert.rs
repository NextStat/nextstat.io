use super::factorize::factorize_covariance_workspace;
use super::schema::{
    SIMPLIFIED_LIKELIHOOD_SCHEMA_V0, SimplifiedBasisComponent, SimplifiedLikelihoodWorkspace,
    SimplifiedUncertaintyModel,
};
use super::validate::validate_simplified_likelihood;
use crate::pyhf::{
    Channel, HistFactoryModel, HistoSysData, HistoSysInterpCode, Measurement, MeasurementConfig,
    Modifier, NormSysInterpCode, Observation, ParameterConfig, Sample, Workspace,
};
use ns_core::Result;
use std::collections::HashMap;

pub fn simplified_to_workspace(spec: &SimplifiedLikelihoodWorkspace) -> Result<Workspace> {
    validate_simplified_likelihood(spec)?;

    let factorized = match &spec.uncertainty_model {
        SimplifiedUncertaintyModel::Basis { .. } => None,
        SimplifiedUncertaintyModel::Covariance { .. } => {
            Some(factorize_covariance_workspace(spec)?)
        }
    };
    let components: &[SimplifiedBasisComponent] = match (&spec.uncertainty_model, &factorized) {
        (SimplifiedUncertaintyModel::Basis { components }, None) => components,
        (SimplifiedUncertaintyModel::Covariance { .. }, Some(result)) => {
            result.components.as_slice()
        }
        _ => unreachable!("covariance factorization state must match uncertainty model"),
    };

    let channel_groups = group_bins_by_channel(&spec.bins);
    let mut channels = Vec::with_capacity(channel_groups.len());
    let mut observations = Vec::with_capacity(channel_groups.len());

    for (channel_name, indices) in channel_groups {
        let mut samples = Vec::new();

        if let Some(signal_nominal) = &spec.signal_nominal {
            let signal_data = select_values(signal_nominal, &indices);
            samples.push(Sample {
                name: "signal".to_string(),
                data: signal_data,
                modifiers: vec![Modifier::NormFactor { name: spec.poi.name.clone(), data: None }],
            });
        }

        let mut background_modifiers = Vec::with_capacity(components.len());
        for component in components {
            background_modifiers.push(Modifier::HistoSys {
                name: component.name.clone(),
                data: HistoSysData {
                    hi_data: select_values(&component.hi, &indices),
                    lo_data: select_values(&component.lo, &indices),
                },
            });
        }

        samples.push(Sample {
            name: "total_background".to_string(),
            data: select_values(&spec.background_nominal, &indices),
            modifiers: background_modifiers,
        });

        channels.push(Channel { name: channel_name.clone(), samples });
        observations.push(Observation {
            name: channel_name,
            data: select_values(&spec.observed, &indices),
        });
    }

    Ok(Workspace {
        channels,
        observations,
        measurements: vec![Measurement {
            name: "SimplifiedLikelihood".to_string(),
            config: MeasurementConfig {
                poi: spec.poi.name.clone(),
                parameters: vec![ParameterConfig {
                    name: spec.poi.name.clone(),
                    inits: vec![spec.poi.init],
                    bounds: vec![spec.poi.bounds],
                    fixed: false,
                    auxdata: Vec::new(),
                    sigmas: Vec::new(),
                    constraint: None,
                }],
            },
        }],
        version: Some(SIMPLIFIED_LIKELIHOOD_SCHEMA_V0.to_string()),
    })
}

pub fn simplified_to_model(spec: &SimplifiedLikelihoodWorkspace) -> Result<HistFactoryModel> {
    let workspace = simplified_to_workspace(spec)?;
    HistFactoryModel::from_workspace_with_settings(
        &workspace,
        NormSysInterpCode::Code1,
        HistoSysInterpCode::Code0,
    )
}

fn group_bins_by_channel(
    bins: &[super::schema::SimplifiedLikelihoodBin],
) -> Vec<(String, Vec<usize>)> {
    let mut order = Vec::<String>::new();
    let mut by_channel = HashMap::<String, Vec<usize>>::new();

    for (idx, bin) in bins.iter().enumerate() {
        by_channel.entry(bin.channel.clone()).or_insert_with(|| {
            order.push(bin.channel.clone());
            Vec::new()
        });
        by_channel.get_mut(&bin.channel).expect("channel must exist").push(idx);
    }

    order
        .into_iter()
        .map(|channel| {
            let indices = by_channel.remove(&channel).expect("grouped channel indices must exist");
            (channel, indices)
        })
        .collect()
}

fn select_values(values: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&idx| values[idx]).collect()
}
