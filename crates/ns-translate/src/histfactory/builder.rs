//! Builder: XML + ROOT histograms → pyhf Workspace.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ns_core::{Error, Result};
use ns_root::{Histogram, RootFile};

use crate::pyhf::schema::{
    Channel, HistoSysData, Measurement, MeasurementConfig, Modifier, NormSysData, Observation,
    ParameterConfig, Sample, Workspace,
};

use super::channel::{self, ChannelXml, ModifierXml};
use super::combination::{self, CombinationConfig};

/// Extract per-channel bin edges from a HistFactory export.
///
/// This uses each channel's `data` histogram as the canonical binning source.
pub fn bin_edges_by_channel_from_xml(combination_path: &Path) -> Result<HashMap<String, Vec<f64>>> {
    bin_edges_by_channel_from_xml_with_basedir(combination_path, None)
}

/// Like [`bin_edges_by_channel_from_xml`] but with an explicit base directory
/// for resolving relative paths in the XML (ROOT files, sub-XMLs).
///
/// pyhf validation fixtures use paths relative to the *export root*, not
/// relative to the parent of `combination.xml`, so the caller must supply
/// `base_dir` explicitly in those cases.
pub fn bin_edges_by_channel_from_xml_with_basedir(
    combination_path: &Path,
    base_dir: Option<&Path>,
) -> Result<HashMap<String, Vec<f64>>> {
    let base_dir =
        base_dir.unwrap_or_else(|| combination_path.parent().unwrap_or_else(|| Path::new(".")));

    let config = combination::parse_combination(combination_path)?;
    let channels_xml: Vec<ChannelXml> = config
        .channel_files
        .iter()
        .map(|f| channel::parse_channel(&base_dir.join(f)))
        .collect::<Result<Vec<_>>>()?;

    let mut root_cache: HashMap<PathBuf, RootFile> = HashMap::new();
    let mut out: HashMap<String, Vec<f64>> = HashMap::new();

    for ch_xml in &channels_xml {
        let input_file = ch_xml.data.input_file.as_deref().or(ch_xml.input_file.as_deref());
        let histo_path = ch_xml.data.histo_path.as_deref().or(ch_xml.histo_path.as_deref());
        let input_file = input_file.ok_or_else(|| {
            Error::Xml(format!(
                "no InputFile specified for channel '{}' data histogram",
                ch_xml.name
            ))
        })?;

        let root_path = base_dir.join(input_file);
        if !root_cache.contains_key(&root_path) {
            let rf = RootFile::open(&root_path)
                .map_err(|e| Error::RootFile(format!("opening {}: {}", root_path.display(), e)))?;
            root_cache.insert(root_path.clone(), rf);
        }
        let rf = root_cache.get(&root_path).unwrap();

        let full_path = match histo_path {
            Some(hp) if !hp.is_empty() => format!("{}/{}", hp, ch_xml.data.histo_name),
            _ => ch_xml.data.histo_name.clone(),
        };
        let hist = rf.get_histogram(&full_path).map_err(|e| {
            Error::RootFile(format!(
                "reading histogram '{}' from {}: {}",
                full_path,
                root_path.display(),
                e
            ))
        })?;
        out.insert(ch_xml.name.clone(), hist.bin_edges);
    }

    Ok(out)
}

/// Parse a HistFactory `combination.xml` and its referenced ROOT files,
/// producing a `Workspace` identical to the pyhf JSON format.
pub fn from_xml(combination_path: &Path) -> Result<Workspace> {
    from_xml_with_basedir(combination_path, None)
}

/// Like [`from_xml`] but with an explicit base directory for resolving
/// relative paths (ROOT files, sub-XML files).
///
/// When `base_dir` is `None`, falls back to `combination_path.parent()`.
pub fn from_xml_with_basedir(
    combination_path: &Path,
    base_dir: Option<&Path>,
) -> Result<Workspace> {
    let base_dir =
        base_dir.unwrap_or_else(|| combination_path.parent().unwrap_or_else(|| Path::new(".")));

    let config = combination::parse_combination(combination_path)?;

    // Parse channel XML files
    let channels_xml: Vec<ChannelXml> = config
        .channel_files
        .iter()
        .map(|f| channel::parse_channel(&base_dir.join(f)))
        .collect::<Result<Vec<_>>>()?;

    // Open ROOT files (cached by path)
    let mut root_cache: HashMap<PathBuf, RootFile> = HashMap::new();

    // Build Workspace
    let mut ws_channels = Vec::new();
    let mut ws_observations = Vec::new();
    let mut normfactor_settings: HashMap<String, (f64, f64, f64)> = HashMap::new();

    for ch_xml in &channels_xml {
        let (channel, observation) =
            build_channel(ch_xml, base_dir, &config, &mut root_cache, &mut normfactor_settings)?;
        ws_channels.push(channel);
        ws_observations.push(observation);
    }

    let ws_measurements = build_measurements(&config, &normfactor_settings)?;

    Ok(Workspace {
        channels: ws_channels,
        observations: ws_observations,
        measurements: ws_measurements,
        version: Some("1.0.0".into()),
    })
}

/// Build a Channel + Observation from a parsed channel XML.
fn build_channel(
    ch: &ChannelXml,
    base_dir: &Path,
    config: &CombinationConfig,
    root_cache: &mut HashMap<PathBuf, RootFile>,
    normfactor_settings: &mut HashMap<String, (f64, f64, f64)>,
) -> Result<(Channel, Observation)> {
    // Read observed data
    let obs_hist = resolve_and_read_histogram(
        &ch.data.histo_name,
        ch.data.histo_path.as_deref().or(ch.histo_path.as_deref()),
        ch.data.input_file.as_deref().or(ch.input_file.as_deref()),
        base_dir,
        root_cache,
    )?;

    let observation = Observation { name: ch.name.clone(), data: obs_hist.bin_content };

    // Build samples
    let mut samples = Vec::new();
    for s in &ch.samples {
        let sample = build_sample(s, ch, base_dir, config, root_cache, normfactor_settings)?;
        samples.push(sample);
    }

    let channel = Channel { name: ch.name.clone(), samples };

    Ok((channel, observation))
}

/// Build a Sample from XML + ROOT data.
fn build_sample(
    s: &channel::SampleXml,
    ch: &ChannelXml,
    base_dir: &Path,
    config: &CombinationConfig,
    root_cache: &mut HashMap<PathBuf, RootFile>,
    normfactor_settings: &mut HashMap<String, (f64, f64, f64)>,
) -> Result<Sample> {
    // Resolve input file: sample > channel defaults
    let input_file = s.input_file.as_deref().or(ch.input_file.as_deref());
    let histo_path = s.histo_path.as_deref().or(ch.histo_path.as_deref());

    // Read nominal histogram
    let nominal =
        resolve_and_read_histogram(&s.histo_name, histo_path, input_file, base_dir, root_cache)?;

    // Build modifiers
    let mut modifiers = Vec::new();

    // Record NormFactor constraints (Val/Low/High) so they can be surfaced via MeasurementConfig.
    for m in &s.modifiers {
        if let ModifierXml::NormFactor { name, val, low, high } = m {
            match normfactor_settings.get(name) {
                None => {
                    normfactor_settings.insert(name.clone(), (*val, *low, *high));
                }
                Some((v0, lo0, hi0)) => {
                    if (*v0 - *val).abs() > 1e-12 || (*lo0 - *low).abs() > 1e-12 || (*hi0 - *high).abs() > 1e-12
                    {
                        return Err(Error::Validation(format!(
                            "NormFactor '{}' has inconsistent settings across channels/samples (existing val/lo/hi = {}/{}/{}; new = {}/{}/{})",
                            name, v0, lo0, hi0, val, low, high
                        )));
                    }
                }
            }
        }
    }

    // HistFactory lumi: apply a `lumi` modifier to samples normalized by theory.
    // The parameter name is conventionally "Lumi" (as used in ParamSetting in fixtures).
    if s.normalize_by_theory && config.measurements.iter().any(|m| m.lumi_rel_err > 0.0) {
        modifiers.push(Modifier::Lumi { name: "Lumi".to_string(), data: None });
    }

    for m in &s.modifiers {
        let mods =
            build_modifier(
                m,
                histo_path,
                input_file,
                base_dir,
                root_cache,
                &nominal,
                &ch.name,
                &s.name,
                ch.stat_error_config.as_ref().map(|c| c.constraint_type.as_str()),
            )?;
        modifiers.extend(mods);
    }

    Ok(Sample { name: s.name.clone(), data: nominal.bin_content, modifiers })
}

/// Build Modifier(s) from an XML modifier element.
fn build_modifier(
    m: &ModifierXml,
    default_histo_path: Option<&str>,
    default_input_file: Option<&str>,
    base_dir: &Path,
    root_cache: &mut HashMap<PathBuf, RootFile>,
    nominal: &Histogram,
    channel_name: &str,
    sample_name: &str,
    staterror_constraint_type: Option<&str>,
) -> Result<Vec<Modifier>> {
    match m {
        ModifierXml::NormFactor { name, .. } => {
            Ok(vec![Modifier::NormFactor { name: name.clone(), data: None }])
        }
        ModifierXml::OverallSys { name, low, high } => Ok(vec![Modifier::NormSys {
            name: name.clone(),
            data: NormSysData { hi: *high, lo: *low },
        }]),
        ModifierXml::HistoSys {
            name,
            histo_name_high,
            histo_name_low,
            histo_path_high,
            histo_path_low,
            input_file_high,
            input_file_low,
        } => {
            let hp_hi = histo_path_high.as_deref().or(default_histo_path);
            let hp_lo = histo_path_low.as_deref().or(default_histo_path);
            let if_hi = input_file_high.as_deref().or(default_input_file);
            let if_lo = input_file_low.as_deref().or(default_input_file);

            let hi_data =
                resolve_and_read_histogram(histo_name_high, hp_hi, if_hi, base_dir, root_cache)?;
            let lo_data =
                resolve_and_read_histogram(histo_name_low, hp_lo, if_lo, base_dir, root_cache)?;

            Ok(vec![Modifier::HistoSys {
                name: name.clone(),
                data: HistoSysData { hi_data: hi_data.bin_content, lo_data: lo_data.bin_content },
            }])
        }
        ModifierXml::ShapeSys { name, histo_name, histo_path, input_file, .. } => {
            let data = if let Some(hn) = histo_name {
                let hp = histo_path.as_deref().or(default_histo_path);
                let ifn = input_file.as_deref().or(default_input_file);
                let rel = resolve_and_read_histogram(hn, hp, ifn, base_dir, root_cache)?.bin_content;
                if rel.len() != nominal.bin_content.len() {
                    return Err(Error::Xml(format!(
                        "ShapeSys histogram '{}' bin count mismatch: got={} expected={} (channel={})",
                        hn,
                        rel.len(),
                        nominal.bin_content.len(),
                        channel_name,
                    )));
                }
                // HistFactory convention: ShapeSys histogram stores RELATIVE uncertainties.
                rel.iter()
                    .zip(nominal.bin_content.iter())
                    .map(|(r, n)| r * n)
                    .collect()
            } else {
                // If no histogram specified, fall back to Poisson-ish absolute uncertainties.
                nominal.bin_content.iter().map(|v| v.sqrt()).collect()
            };

            Ok(vec![Modifier::ShapeSys { name: name.clone(), data }])
        }
        ModifierXml::ShapeFactor { name } => {
            Ok(vec![Modifier::ShapeFactor { name: name.clone(), data: None }])
        }
        ModifierXml::StatError { histo_name, histo_path, input_file } => {
            // StatError:
            // - If a StatError histogram is provided, HistFactory convention is that it stores
            //   **relative** uncertainties per bin. pyhf's `staterror` modifier expects
            //   **absolute** per-bin sigmas, so we multiply by the nominal bin content.
            // - If no histogram is provided, we use sqrt(sumw2) (preferred) or sqrt(nominal).
            let data = if let Some(hn) = histo_name {
                let hp = histo_path.as_deref().or(default_histo_path);
                let ifn = input_file.as_deref().or(default_input_file);
                let rel =
                    resolve_and_read_histogram(hn, hp, ifn, base_dir, root_cache)?.bin_content;
                if rel.len() != nominal.bin_content.len() {
                    return Err(Error::Xml(format!(
                        "StatError histogram '{}' bin count mismatch: got={} expected={} (channel={})",
                        hn,
                        rel.len(),
                        nominal.bin_content.len(),
                        channel_name,
                    )));
                }
                rel.iter().zip(nominal.bin_content.iter()).map(|(r, n)| r * n).collect()
            } else {
                // Prefer nominal sumw2 if available (weighted MC templates).
                if let Some(sw2) = nominal.sumw2.as_ref() {
                    sw2.iter().map(|v| v.max(0.0).sqrt()).collect()
                } else {
                    // Fallback: Poisson-ish sqrt(nominal).
                    nominal.bin_content.iter().map(|v| v.sqrt()).collect()
                }
            };

            // StatError name follows pyhf convention: "staterror_{channel_name}"
            let constraint_type = staterror_constraint_type.unwrap_or("Gaussian");
            if constraint_type.eq_ignore_ascii_case("poisson") {
                // Poisson constraint implies Barlow–Beeston: use shapesys (per-sample).
                let name = format!("staterror_{}_{}", channel_name, sample_name);
                Ok(vec![Modifier::ShapeSys { name, data }])
            } else {
                let stat_name = format!("staterror_{}", channel_name);
                Ok(vec![Modifier::StatError { name: stat_name, data }])
            }
        }
    }
}

/// Build Measurement configs from combination.
fn build_measurements(
    config: &CombinationConfig,
    normfactor_settings: &HashMap<String, (f64, f64, f64)>,
) -> Result<Vec<Measurement>> {
    config
        .measurements
        .iter()
        .map(|m| {
            let mut parameters: Vec<ParameterConfig> = m
                .param_settings
                .iter()
                .flat_map(|ps| {
                    ps.names.iter().map(move |name| ParameterConfig {
                        name: name.clone(),
                        inits: ps.val.map(|v| vec![v]).unwrap_or_default(),
                        bounds: Vec::new(),
                        fixed: ps.is_const,
                        auxdata: Vec::new(),
                        sigmas: Vec::new(),
                    })
                })
                .collect();

            // Add/augment lumi constraint if configured.
            if m.lumi_rel_err > 0.0 {
                let idx = parameters.iter().position(|p| p.name == "Lumi");
                if let Some(i) = idx {
                    if parameters[i].auxdata.is_empty() {
                        parameters[i].auxdata = vec![1.0];
                    }
                    if parameters[i].sigmas.is_empty() {
                        parameters[i].sigmas = vec![m.lumi_rel_err];
                    }
                    if parameters[i].inits.is_empty() {
                        parameters[i].inits = vec![1.0];
                    }
                } else {
                    parameters.push(ParameterConfig {
                        name: "Lumi".to_string(),
                        inits: vec![1.0],
                        bounds: Vec::new(),
                        fixed: false,
                        auxdata: vec![1.0],
                        sigmas: vec![m.lumi_rel_err],
                    });
                }
            }

            // Surface NormFactor Val/Low/High as parameter inits/bounds (used by the model).
            for (name, (val, low, high)) in normfactor_settings {
                let idx = parameters.iter().position(|p| p.name == *name);
                if let Some(i) = idx {
                    parameters[i].inits = vec![*val];
                    parameters[i].bounds = vec![[*low, *high]];
                } else {
                    parameters.push(ParameterConfig {
                        name: name.clone(),
                        inits: vec![*val],
                        bounds: vec![[*low, *high]],
                        fixed: false,
                        auxdata: Vec::new(),
                        sigmas: Vec::new(),
                    });
                }
            }

            // Apply Measurement-level ConstraintTerm overrides (if present).
            //
            // Note: the pyhf JSON schema encodes constraint info in `auxdata` + `sigmas`
            // without an explicit distribution tag. We preserve the declared type by
            // mapping it into a (best-effort) Gaussian-equivalent width when possible.
            for ct in &m.constraint_terms {
                let sigma = match (ct.constraint_type.as_str(), ct.rel_uncertainty) {
                    ("LogNormal" | "lognormal" | "LOGNORMAL", Some(rel)) if rel > 0.0 => {
                        (1.0 + rel).ln()
                    }
                    (_, Some(rel)) if rel > 0.0 => rel,
                    _ => 1.0,
                };

                for name in &ct.names {
                    if name.is_empty() {
                        continue;
                    }

                    // Center convention:
                    // - Lumi-like multiplicative parameters are centered at 1
                    // - Most HistFactory nuisance parameters (overall/histo sys) are centered at 0
                    let center = if name.eq_ignore_ascii_case("Lumi") { 1.0 } else { 0.0 };

                    let idx = parameters.iter().position(|p| p.name == *name);
                    if let Some(i) = idx {
                        // Only fill auxdata/sigmas if not already set (avoid clobbering user config).
                        if parameters[i].auxdata.is_empty() {
                            parameters[i].auxdata = vec![center];
                        }
                        if parameters[i].sigmas.is_empty() {
                            parameters[i].sigmas = vec![sigma];
                        }
                    } else {
                        parameters.push(ParameterConfig {
                            name: name.clone(),
                            inits: Vec::new(),
                            bounds: Vec::new(),
                            fixed: false,
                            auxdata: vec![center],
                            sigmas: vec![sigma],
                        });
                    }
                }
            }

            Ok(Measurement {
                name: m.name.clone(),
                config: MeasurementConfig { poi: m.poi.clone(), parameters },
            })
        })
        .collect()
}

/// Resolve a histogram reference and read bin contents from a ROOT file.
fn resolve_and_read_histogram(
    histo_name: &str,
    histo_path: Option<&str>,
    input_file: Option<&str>,
    base_dir: &Path,
    root_cache: &mut HashMap<PathBuf, RootFile>,
) -> Result<Histogram> {
    let input_file = input_file.ok_or_else(|| {
        Error::Xml(format!("no InputFile specified for histogram '{}'", histo_name))
    })?;

    let root_path = base_dir.join(input_file);

    // Open or reuse cached ROOT file
    if !root_cache.contains_key(&root_path) {
        let rf = RootFile::open(&root_path)
            .map_err(|e| Error::RootFile(format!("opening {}: {}", root_path.display(), e)))?;
        root_cache.insert(root_path.clone(), rf);
    }

    let rf = root_cache.get(&root_path).unwrap();

    // Build full path: HistoPath/HistoName
    let full_path = match histo_path {
        Some(hp) if !hp.is_empty() => format!("{}/{}", hp, histo_name),
        _ => histo_name.to_string(),
    };

    let hist = rf.get_histogram(&full_path).map_err(|e| {
        Error::RootFile(format!(
            "reading histogram '{}' from {}: {}",
            full_path,
            root_path.display(),
            e
        ))
    })?;

    Ok(hist)
}
