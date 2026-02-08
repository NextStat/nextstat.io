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

fn resolve_input_file_path(base_dir: &Path, input_file: &str) -> PathBuf {
    let p = Path::new(input_file);
    if p.is_absolute() {
        // TREx/HistFactory export dirs often embed absolute paths. Prefer them when they exist,
        // but fall back to the export-dir copy (`base_dir/<basename>`) when the absolute path
        // is not available on this machine.
        if p.exists() {
            return p.to_path_buf();
        }
        if let Some(name) = p.file_name() {
            let candidate = base_dir.join(name);
            if candidate.exists() {
                return candidate;
            }
        }
        return p.to_path_buf();
    }
    base_dir.join(p)
}

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

        let root_path = resolve_input_file_path(base_dir, input_file);
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
    let mut staterror_gamma_rels: HashMap<String, f64> = HashMap::new();

    for ch_xml in &channels_xml {
        let (channel, observation) =
            build_channel(ch_xml, base_dir, &config, &mut root_cache, &mut normfactor_settings)?;

        // HistFactory/ROOT exports often omit `<StatErrorConfig>`. In that case, ROOT defaults to
        // Poisson/Gamma-constrained per-bin staterror parameters (gamma_stat_<ch>_bin_<i>).
        //
        // The pyhf JSON schema doesn't encode this directly for `staterror`, so we attach the
        // constraint semantics via a non-standard `measurements[].config.parameters[].constraint`
        // extension. The model layer interprets `constraint.type="Gamma"` for the named parameter.
        if ch_xml.stat_error_config.is_none() {
            let stat_name = format!("staterror_{}", channel.name);
            let mut sum_nominal: Option<Vec<f64>> = None;
            let mut sum_sigma2: Option<Vec<f64>> = None;

            for s in &channel.samples {
                for m in &s.modifiers {
                    let Modifier::StatError { name, data } = m else { continue };
                    if name != &stat_name {
                        continue;
                    }
                    if sum_nominal.is_none() {
                        sum_nominal = Some(vec![0.0; s.data.len()]);
                        sum_sigma2 = Some(vec![0.0; data.len()]);
                    }
                    let sn = sum_nominal.as_mut().unwrap();
                    let ss = sum_sigma2.as_mut().unwrap();
                    for (i, (&sigma_abs, &nom)) in data.iter().zip(s.data.iter()).enumerate() {
                        sn[i] += nom;
                        ss[i] += sigma_abs * sigma_abs;
                    }
                }
            }

            if let (Some(sn), Some(ss)) = (sum_nominal, sum_sigma2) {
                for (i, (nom, sigma2)) in sn.into_iter().zip(ss.into_iter()).enumerate() {
                    if nom <= 0.0 || sigma2 <= 0.0 {
                        continue;
                    }
                    let rel = sigma2.sqrt() / nom;
                    if rel.is_finite() && rel > 0.0 {
                        staterror_gamma_rels.insert(format!("{stat_name}[{i}]"), rel);
                    }
                }
            }
        }

        ws_channels.push(channel);
        ws_observations.push(observation);
    }

    let ws_measurements = build_measurements(&config, &normfactor_settings, &staterror_gamma_rels)?;

    let mut ws = Workspace {
        channels: ws_channels,
        observations: ws_observations,
        measurements: ws_measurements,
        version: Some("1.0.0".into()),
    };
    canonicalize_workspace_in_place(&mut ws);
    Ok(ws)
}

fn canonicalize_workspace_in_place(ws: &mut Workspace) {
    // HistFactory exports can include a large number of channels/samples/modifiers, and
    // parts of the import path naturally use HashMaps for aggregation. Ensure the final
    // Workspace has a deterministic ordering independent of HashMap iteration.
    ws.channels.sort_by(|a, b| a.name.cmp(&b.name));
    for ch in &mut ws.channels {
        for s in &mut ch.samples {
            s.modifiers.sort_by(|a, b| modifier_sort_key(a).cmp(&modifier_sort_key(b)));
        }
    }

    ws.observations.sort_by(|a, b| a.name.cmp(&b.name));
    ws.measurements.sort_by(|a, b| a.name.cmp(&b.name));
    for m in &mut ws.measurements {
        m.config.parameters.sort_by(|a, b| a.name.cmp(&b.name));
    }
}

fn modifier_sort_key(m: &Modifier) -> (&'static str, &str) {
    match m {
        Modifier::NormFactor { name, .. } => ("normfactor", name.as_str()),
        Modifier::StatError { name, .. } => ("staterror", name.as_str()),
        Modifier::HistoSys { name, .. } => ("histosys", name.as_str()),
        Modifier::NormSys { name, .. } => ("normsys", name.as_str()),
        Modifier::ShapeSys { name, .. } => ("shapesys", name.as_str()),
        Modifier::ShapeFactor { name, .. } => ("shapefactor", name.as_str()),
        Modifier::Lumi { name, .. } => ("lumi", name.as_str()),
    }
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
        let staterror_constraint_type =
            ch.stat_error_config.as_ref().map(|c| c.constraint_type.as_str());
        // ROOT/HistFactory `RelErrorThreshold` is optional. When absent, treat it as 0.0
        // (i.e., do not disable any bins by default).
        //
        // When `<StatErrorConfig>` is present and `RelErrorThreshold` is specified, use it.
        let staterror_rel_threshold = ch
            .stat_error_config
            .as_ref()
            .and_then(|c| c.rel_error_threshold)
            .unwrap_or(0.0);
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
                staterror_constraint_type,
                staterror_rel_threshold,
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
    staterror_rel_threshold: f64,
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
                rel.iter()
                    .zip(nominal.bin_content.iter())
                    .map(|(r, n)| {
                        // HistFactory convention: StatError histogram stores RELATIVE uncertainties.
                        // Apply the per-channel RelErrorThreshold (bins below threshold are treated as no staterror).
                        let rr = if *r >= staterror_rel_threshold { *r } else { 0.0 };
                        rr * n
                    })
                    .collect()
            } else {
                // Prefer nominal sumw2 if available (weighted MC templates).
                if let Some(sw2) = nominal.sumw2.as_ref() {
                    sw2.iter()
                        .zip(nominal.bin_content.iter())
                        .map(|(v, n)| {
                            let sigma_abs = v.max(0.0).sqrt();
                            let rel = if *n > 0.0 { sigma_abs / *n } else { 0.0 };
                            if rel >= staterror_rel_threshold { sigma_abs } else { 0.0 }
                        })
                        .collect()
                } else {
                    // Fallback: Poisson-ish sqrt(nominal).
                    nominal
                        .bin_content
                        .iter()
                        .map(|v| {
                            let sigma_abs = v.sqrt();
                            let rel = if *v > 0.0 { sigma_abs / *v } else { 0.0 };
                            if rel >= staterror_rel_threshold { sigma_abs } else { 0.0 }
                        })
                        .collect()
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
    staterror_gamma_rels: &HashMap<String, f64>,
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
                        constraint: None,
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
                        constraint: None,
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
                        constraint: None,
                    });
                }
            }

            // Non-standard extension: represent default (omitted StatErrorConfig) as Gamma constraints
            // on the staterror per-bin parameters.
            //
            // Apply this before explicit `<ConstraintTerm>` overrides so that measurement-level
            // metadata can replace it if present.
            for (pname, rel) in staterror_gamma_rels {
                if *rel <= 0.0 || !rel.is_finite() {
                    continue;
                }
                let idx = parameters.iter().position(|p| p.name == *pname);
                let spec = crate::pyhf::schema::ConstraintSpec {
                    constraint_type: "Gamma".to_string(),
                    rel_uncertainty: Some(*rel),
                };
                if let Some(i) = idx {
                    if parameters[i].constraint.is_none() {
                        parameters[i].constraint = Some(spec);
                    }
                } else {
                    parameters.push(ParameterConfig {
                        name: pname.clone(),
                        inits: Vec::new(),
                        bounds: Vec::new(),
                        fixed: false,
                        auxdata: Vec::new(),
                        sigmas: Vec::new(),
                        constraint: Some(spec),
                    });
                }
            }

            // Apply Measurement-level ConstraintTerm metadata (if present).
            //
            // ROOT/HistFactory uses `<ConstraintTerm>` to select alternative constraint-term semantics
            // (Gamma/LogNormal/Uniform/NoConstraint) for named nuisance parameters. This is not part of
            // the pyhf JSON schema, so we store it as a non-standard extension field on the matching
            // `measurements[].config.parameters[]` entries. The actual semantics are implemented
            // in NextStat's model layer.
            for ct in &m.constraint_terms {
                let ctype = ct.constraint_type.trim();
                let rel = ct.rel_uncertainty;

                for name in &ct.names {
                    let name = name.trim();
                    if name.is_empty() {
                        continue;
                    }

                    let idx = parameters.iter().position(|p| p.name == name);
                    if let Some(i) = idx {
                        parameters[i].constraint = Some(crate::pyhf::schema::ConstraintSpec {
                            constraint_type: ctype.to_string(),
                            rel_uncertainty: rel,
                        });
                    } else {
                        parameters.push(ParameterConfig {
                            name: name.to_string(),
                            inits: Vec::new(),
                            bounds: Vec::new(),
                            fixed: false,
                            auxdata: Vec::new(),
                            sigmas: Vec::new(),
                            constraint: Some(crate::pyhf::schema::ConstraintSpec {
                                constraint_type: ctype.to_string(),
                                rel_uncertainty: rel,
                            }),
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

    let root_path = resolve_input_file_path(base_dir, input_file);

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
