//! Export a pyhf [`Workspace`] to HistFactory XML format.
//!
//! Mirrors `pyhf json2xml`: produces a top-level combination XML and per-channel
//! XMLs that describe the workspace structure. Histogram data is written as
//! inline comments (actual ROOT file writing requires the `root-io` feature and
//! is a future enhancement).
//!
//! # Example
//! ```no_run
//! use ns_translate::pyhf::Workspace;
//! use ns_translate::pyhf::xml_export::workspace_to_xml;
//!
//! let json_str = "{}"; // your workspace JSON
//! let ws: Workspace = serde_json::from_str(json_str).unwrap();
//! let xml_set = workspace_to_xml(&ws, "output");
//! for (filename, content) in &xml_set.files {
//!     std::fs::write(filename, content).unwrap();
//! }
//! ```

use super::schema::{Modifier, Workspace};

/// Collection of XML files produced by the export.
#[derive(Debug, Clone)]
pub struct XmlExportSet {
    /// `(filename, xml_content)` pairs. First entry is the top-level combination XML.
    pub files: Vec<(String, String)>,
}

/// Convert a [`Workspace`] into HistFactory XML files.
///
/// `output_prefix` is used as the directory prefix for channel XML paths
/// referenced from the top-level combination file.
pub fn workspace_to_xml(ws: &Workspace, output_prefix: &str) -> XmlExportSet {
    let mut files = Vec::new();

    // --- Per-channel XMLs ---
    let mut channel_filenames = Vec::new();
    for channel in &ws.channels {
        let filename = format!("{}/{}.xml", output_prefix, channel.name);
        let mut xml = String::new();
        xml.push_str("<!DOCTYPE Channel SYSTEM 'HistFactorySchema.dtd'>\n\n");
        xml.push_str(&format!(
            "<Channel Name=\"{}\" InputFile=\"{}/{}.root\" HistoPath=\"\">\n",
            channel.name, output_prefix, channel.name
        ));

        // Observation data
        if let Some(obs) = ws.observations.iter().find(|o| o.name == channel.name) {
            let data_str: Vec<String> = obs.data.iter().map(|v| format!("{}", v)).collect();
            xml.push_str(&format!(
                "  <Data HistoName=\"obsData\" InputFile=\"{}/{}.root\" HistoPath=\"\">\n",
                output_prefix, channel.name
            ));
            xml.push_str(&format!("    <!-- observed: [{}] -->\n", data_str.join(", ")));
            xml.push_str("  </Data>\n\n");
        }

        for sample in &channel.samples {
            xml.push_str(&format!(
                "  <Sample Name=\"{}\" HistoName=\"{}\" InputFile=\"{}/{}.root\" HistoPath=\"\">\n",
                sample.name, sample.name, output_prefix, channel.name
            ));

            let nom_str: Vec<String> = sample.data.iter().map(|v| format!("{}", v)).collect();
            xml.push_str(&format!("    <!-- nominal: [{}] -->\n", nom_str.join(", ")));

            for modifier in &sample.modifiers {
                match modifier {
                    Modifier::NormFactor { name, .. } => {
                        xml.push_str(&format!(
                            "    <NormFactor Name=\"{}\" Val=\"1\" Low=\"0\" High=\"10\" />\n",
                            name
                        ));
                    }
                    Modifier::NormSys { name, data } => {
                        xml.push_str(&format!(
                            "    <OverallSys Name=\"{}\" Low=\"{}\" High=\"{}\" />\n",
                            name, data.lo, data.hi
                        ));
                    }
                    Modifier::HistoSys { name, data } => {
                        xml.push_str(&format!(
                            "    <HistoSys Name=\"{}\" HistoNameHigh=\"{}_Up\" HistoNameLow=\"{}_Down\" />\n",
                            name, name, name
                        ));
                        let hi_str: Vec<String> =
                            data.hi_data.iter().map(|v| format!("{}", v)).collect();
                        let lo_str: Vec<String> =
                            data.lo_data.iter().map(|v| format!("{}", v)).collect();
                        xml.push_str(&format!("    <!-- hi_data: [{}] -->\n", hi_str.join(", ")));
                        xml.push_str(&format!("    <!-- lo_data: [{}] -->\n", lo_str.join(", ")));
                    }
                    Modifier::ShapeSys { name, data } => {
                        xml.push_str(&format!(
                            "    <ShapeSys Name=\"{}\" ConstraintType=\"Poisson\" />\n",
                            name
                        ));
                        let unc_str: Vec<String> = data.iter().map(|v| format!("{}", v)).collect();
                        xml.push_str(&format!(
                            "    <!-- uncertainties: [{}] -->\n",
                            unc_str.join(", ")
                        ));
                    }
                    Modifier::ShapeFactor { name, .. } => {
                        xml.push_str(&format!("    <ShapeFactor Name=\"{}\" />\n", name));
                    }
                    Modifier::StatError { .. } => {
                        xml.push_str("    <StatError Activate=\"True\" />\n");
                    }
                    Modifier::Lumi { name, .. } => {
                        xml.push_str(&format!(
                            "    <NormFactor Name=\"{}\" Val=\"1\" Low=\"0.9\" High=\"1.1\" />\n",
                            name
                        ));
                        xml.push_str("    <!-- lumi modifier -->\n");
                    }
                }
            }

            xml.push_str("  </Sample>\n\n");
        }

        xml.push_str("</Channel>\n");
        channel_filenames.push(filename.clone());
        files.push((filename, xml));
    }

    // --- Top-level combination XML ---
    let mut combo = String::new();
    combo.push_str("<!DOCTYPE Combination SYSTEM 'HistFactorySchema.dtd'>\n\n");
    combo.push_str("<Combination OutputFilePrefix=\"./results\">\n\n");

    for cf in &channel_filenames {
        combo.push_str(&format!("  <Input>{}</Input>\n", cf));
    }
    combo.push('\n');

    for meas in &ws.measurements {
        combo.push_str(&format!(
            "  <Measurement Name=\"{}\" Lumi=\"1\" LumiRelErr=\"0\" ExportOnly=\"True\">\n",
            meas.name
        ));
        combo.push_str(&format!("    <POI>{}</POI>\n", meas.config.poi));

        for pc in &meas.config.parameters {
            let mut attrs = format!("Name=\"{}\"", pc.name);
            if let Some(v) = pc.inits.first() {
                attrs.push_str(&format!(" Val=\"{}\"", v));
            }
            if let Some(pair) = pc.bounds.first() {
                attrs.push_str(&format!(" Low=\"{}\" High=\"{}\"", pair[0], pair[1]));
            }
            if pc.fixed {
                attrs.push_str(" Const=\"True\"");
            }
            combo.push_str(&format!("    <ParamSetting {}  />\n", attrs));
        }

        combo.push_str("  </Measurement>\n\n");
    }

    combo.push_str("</Combination>\n");

    let combo_filename = format!("{}/combination.xml", output_prefix);
    files.insert(0, (combo_filename, combo));

    XmlExportSet { files }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_to_xml_simple() {
        let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        let xml_set = workspace_to_xml(&ws, "output");

        assert!(!xml_set.files.is_empty());
        let (combo_name, combo_content) = &xml_set.files[0];
        assert!(combo_name.contains("combination.xml"));
        assert!(combo_content.contains("<Combination"));
        assert!(combo_content.contains("<POI>"));
        assert!(combo_content.contains("<Input>"));

        // Should have 1 combination + N channel files
        assert_eq!(xml_set.files.len(), 1 + ws.channels.len());

        for (name, content) in &xml_set.files[1..] {
            assert!(name.ends_with(".xml"));
            assert!(content.contains("<Channel"));
            assert!(content.contains("<Sample"));
        }
    }

    #[test]
    fn test_workspace_to_xml_roundtrip_structure() {
        let ws = crate::pyhf::simplemodels::uncorrelated_background(
            &[5.0, 10.0],
            &[50.0, 60.0],
            &[7.0, 8.0],
        );
        let xml_set = workspace_to_xml(&ws, "test_output");

        assert_eq!(xml_set.files.len(), 2); // combination + 1 channel
        let (_, channel_xml) = &xml_set.files[1];
        assert!(channel_xml.contains("NormFactor"));
        assert!(channel_xml.contains("ShapeSys"));
        assert!(channel_xml.contains("singlechannel"));
    }
}
