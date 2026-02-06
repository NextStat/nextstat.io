//! Parser for HistFactory `combination.xml`.

use ns_core::{Error, Result};
use std::path::{Path, PathBuf};

/// Parsed combination.xml content.
#[derive(Debug)]
pub struct CombinationConfig {
    /// Paths to channel XML files (relative to combination.xml directory).
    pub channel_files: Vec<PathBuf>,
    /// Measurement configurations.
    pub measurements: Vec<MeasurementXml>,
}

/// A <Measurement> element from combination.xml.
#[derive(Debug)]
pub struct MeasurementXml {
    /// Measurement name.
    pub name: String,
    /// Parameter of interest.
    pub poi: String,
    /// Luminosity value.
    pub lumi: f64,
    /// Relative luminosity uncertainty.
    pub lumi_rel_err: f64,
    /// Parameter settings.
    pub param_settings: Vec<ParamSetting>,
}

/// A <ParamSetting> element.
#[derive(Debug)]
pub struct ParamSetting {
    /// Parameter names (space-separated in XML).
    pub names: Vec<String>,
    /// Whether parameters are fixed.
    pub is_const: bool,
    /// Initial value (if specified).
    pub val: Option<f64>,
}

/// Parse a combination.xml file.
pub fn parse_combination(path: &Path) -> Result<CombinationConfig> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Xml(format!("reading {}: {}", path.display(), e)))?;

    // Strip DTD declaration — HistFactory XMLs always have <!DOCTYPE>
    // and roxmltree rejects DTD by default.
    let text: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with("<!DOCTYPE"))
        .collect::<Vec<_>>()
        .join("\n");

    let doc = roxmltree::Document::parse(&text)
        .map_err(|e| Error::Xml(format!("parsing {}: {}", path.display(), e)))?;

    let root = doc.root_element();

    // <Input> elements → channel file paths
    let channel_files: Vec<PathBuf> = root
        .children()
        .filter(|n| n.has_tag_name("Input"))
        .filter_map(|n| n.text())
        .map(|t| PathBuf::from(t.trim()))
        .collect();

    if channel_files.is_empty() {
        return Err(Error::Xml(
            "combination.xml has no <Input> elements".into(),
        ));
    }

    // <Measurement> elements
    let measurements: Vec<MeasurementXml> = root
        .children()
        .filter(|n| n.has_tag_name("Measurement"))
        .map(parse_measurement)
        .collect::<Result<Vec<_>>>()?;

    if measurements.is_empty() {
        return Err(Error::Xml(
            "combination.xml has no <Measurement> elements".into(),
        ));
    }

    Ok(CombinationConfig {
        channel_files,
        measurements,
    })
}

fn parse_measurement(node: roxmltree::Node) -> Result<MeasurementXml> {
    let name = node
        .attribute("Name")
        .unwrap_or("default")
        .to_string();

    let lumi: f64 = node
        .attribute("Lumi")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);

    let lumi_rel_err: f64 = node
        .attribute("LumiRelErr")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    // <POI> element text
    let poi = node
        .children()
        .find(|n| n.has_tag_name("POI"))
        .and_then(|n| n.text())
        .map(|t| t.trim().to_string())
        .unwrap_or_else(|| "mu".to_string());

    // <ParamSetting> elements
    let param_settings: Vec<ParamSetting> = node
        .children()
        .filter(|n| n.has_tag_name("ParamSetting"))
        .map(|n| {
            let is_const = n
                .attribute("Const")
                .map(|v| v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

            let val = n.attribute("Val").and_then(|v| v.parse().ok());

            let names: Vec<String> = n
                .text()
                .unwrap_or("")
                .split_whitespace()
                .map(String::from)
                .collect();

            ParamSetting {
                names,
                is_const,
                val,
            }
        })
        .collect();

    Ok(MeasurementXml {
        name,
        poi,
        lumi,
        lumi_rel_err,
        param_settings,
    })
}
