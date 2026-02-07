//! Parser for HistFactory channel XML files.

use ns_core::{Error, Result};
use std::path::Path;

/// A parsed channel XML file.
#[derive(Debug)]
pub struct ChannelXml {
    /// Channel name.
    pub name: String,
    /// Default ROOT input file (may be overridden per-sample).
    pub input_file: Option<String>,
    /// Default histogram path prefix.
    pub histo_path: Option<String>,
    /// Observed data histogram reference.
    pub data: DataRef,
    /// Samples in this channel.
    pub samples: Vec<SampleXml>,
}

/// Reference to observed data histogram.
#[derive(Debug)]
pub struct DataRef {
    /// Histogram name.
    pub histo_name: String,
    /// Input file (overrides channel default).
    pub input_file: Option<String>,
    /// Histogram path (overrides channel default).
    pub histo_path: Option<String>,
}

/// A <Sample> element.
#[derive(Debug)]
pub struct SampleXml {
    /// Sample name.
    pub name: String,
    /// Nominal histogram name.
    pub histo_name: String,
    /// Input file (overrides channel default).
    pub input_file: Option<String>,
    /// Histogram path (overrides channel default).
    pub histo_path: Option<String>,
    /// Modifiers on this sample.
    pub modifiers: Vec<ModifierXml>,
}

/// Modifier parsed from XML.
#[derive(Debug)]
pub enum ModifierXml {
    /// `<NormFactor>`
    NormFactor { name: String, val: f64, low: f64, high: f64 },
    /// `<OverallSys>`
    OverallSys { name: String, low: f64, high: f64 },
    /// `<HistoSys>`
    HistoSys {
        name: String,
        histo_name_high: String,
        histo_name_low: String,
        histo_path_high: Option<String>,
        histo_path_low: Option<String>,
        input_file_high: Option<String>,
        input_file_low: Option<String>,
    },
    /// `<ShapeSys>`
    ShapeSys {
        name: String,
        histo_name: Option<String>,
        histo_path: Option<String>,
        input_file: Option<String>,
        constraint_type: String,
    },
    /// `<ShapeFactor>`
    ShapeFactor { name: String },
    /// `<StatError Activate="True" />`
    StatError { histo_name: Option<String>, histo_path: Option<String>, input_file: Option<String> },
}

/// Parse a channel XML file.
pub fn parse_channel(path: &Path) -> Result<ChannelXml> {
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

    let name = root
        .attribute("Name")
        .ok_or_else(|| Error::Xml("Channel element missing Name attribute".into()))?
        .to_string();

    let input_file = root.attribute("InputFile").map(String::from);
    let histo_path = root.attribute("HistoPath").map(String::from);

    // <Data> element
    let data_node = root
        .children()
        .find(|n| n.has_tag_name("Data"))
        .ok_or_else(|| Error::Xml(format!("channel {} has no <Data> element", name)))?;

    let data = DataRef {
        histo_name: data_node
            .attribute("HistoName")
            .ok_or_else(|| Error::Xml("Data element missing HistoName".into()))?
            .to_string(),
        input_file: data_node.attribute("InputFile").map(String::from),
        histo_path: data_node.attribute("HistoPath").map(String::from),
    };

    // <Sample> elements
    let samples: Vec<SampleXml> = root
        .children()
        .filter(|n| n.has_tag_name("Sample"))
        .map(parse_sample)
        .collect::<Result<Vec<_>>>()?;

    Ok(ChannelXml { name, input_file, histo_path, data, samples })
}

fn parse_sample(node: roxmltree::Node) -> Result<SampleXml> {
    let name = node
        .attribute("Name")
        .ok_or_else(|| Error::Xml("Sample element missing Name attribute".into()))?
        .to_string();

    let histo_name = node
        .attribute("HistoName")
        .ok_or_else(|| Error::Xml(format!("Sample '{}' missing HistoName", name)))?
        .to_string();

    let input_file = node.attribute("InputFile").map(String::from);
    let histo_path = node.attribute("HistoPath").map(String::from);

    let mut modifiers = Vec::new();

    for child in node.children() {
        if child.is_element()
            && let Some(m) = parse_modifier(child)? {
                modifiers.push(m);
            }
    }

    Ok(SampleXml { name, histo_name, input_file, histo_path, modifiers })
}

fn parse_modifier(node: roxmltree::Node) -> Result<Option<ModifierXml>> {
    let tag = node.tag_name().name();

    match tag {
        "NormFactor" => {
            let name = attr_string(&node, "Name")?;
            let val = attr_f64(&node, "Val", 1.0);
            let low = attr_f64(&node, "Low", 0.0);
            let high = attr_f64(&node, "High", 10.0);
            Ok(Some(ModifierXml::NormFactor { name, val, low, high }))
        }
        "OverallSys" => {
            let name = attr_string(&node, "Name")?;
            let low = attr_f64(&node, "Low", 1.0);
            let high = attr_f64(&node, "High", 1.0);
            Ok(Some(ModifierXml::OverallSys { name, low, high }))
        }
        "HistoSys" => {
            let name = attr_string(&node, "Name")?;
            Ok(Some(ModifierXml::HistoSys {
                name,
                histo_name_high: attr_string(&node, "HistoNameHigh")?,
                histo_name_low: attr_string(&node, "HistoNameLow")?,
                histo_path_high: node.attribute("HistoPathHigh").map(String::from),
                histo_path_low: node.attribute("HistoPathLow").map(String::from),
                input_file_high: node.attribute("InputFileHigh").map(String::from),
                input_file_low: node.attribute("InputFileLow").map(String::from),
            }))
        }
        "ShapeSys" => {
            let name = attr_string(&node, "Name")?;
            let constraint_type = node.attribute("ConstraintType").unwrap_or("Poisson").to_string();
            Ok(Some(ModifierXml::ShapeSys {
                name,
                histo_name: node.attribute("HistoName").map(String::from),
                histo_path: node.attribute("HistoPath").map(String::from),
                input_file: node.attribute("InputFile").map(String::from),
                constraint_type,
            }))
        }
        "ShapeFactor" => {
            let name = attr_string(&node, "Name")?;
            Ok(Some(ModifierXml::ShapeFactor { name }))
        }
        "StatError" => {
            let activate =
                node.attribute("Activate").map(|v| v.eq_ignore_ascii_case("true")).unwrap_or(false);

            if !activate {
                return Ok(None);
            }

            Ok(Some(ModifierXml::StatError {
                histo_name: node.attribute("HistoName").map(String::from),
                histo_path: node.attribute("HistoPath").map(String::from),
                input_file: node.attribute("InputFile").map(String::from),
            }))
        }
        _ => Ok(None), // Ignore unknown elements
    }
}

fn attr_string(node: &roxmltree::Node, name: &str) -> Result<String> {
    node.attribute(name)
        .map(String::from)
        .ok_or_else(|| Error::Xml(format!("missing attribute '{}'", name)))
}

fn attr_f64(node: &roxmltree::Node, name: &str, default: f64) -> f64 {
    node.attribute(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}
