use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::Value;

use ns_root::RootFile;

use super::{bin_edges_by_channel_from_xml, from_xml, from_xml_with_basedir};

fn fixture_combination_xml() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/histfactory/combination.xml")
}

fn fixture_pyhf_xmlimport() -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pyhf_xmlimport");
    (root.join("config/example.xml"), root)
}

fn fixture_pyhf_multichannel() -> (PathBuf, PathBuf) {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pyhf_multichannel");
    (root.join("config/example.xml"), root)
}

#[test]
fn histfactory_from_xml_is_deterministic() {
    let xml = fixture_combination_xml();

    let ws1 = from_xml(&xml).expect("from_xml(ws1)");
    let ws2 = from_xml(&xml).expect("from_xml(ws2)");

    // Compare as JSON values so map key ordering cannot cause spurious diffs.
    let v1: Value = serde_json::to_value(ws1).expect("to_value(ws1)");
    let v2: Value = serde_json::to_value(ws2).expect("to_value(ws2)");
    assert_eq!(v1, v2);
}

#[test]
fn histfactory_bin_edges_by_channel_is_deterministic() {
    let xml = fixture_combination_xml();

    let a = bin_edges_by_channel_from_xml(&xml).expect("bin_edges (a)");
    let b = bin_edges_by_channel_from_xml(&xml).expect("bin_edges (b)");

    // HashMap order is non-deterministic; compare via JSON Value map equality.
    let v1: Value = serde_json::to_value(a).expect("to_value(a)");
    let v2: Value = serde_json::to_value(b).expect("to_value(b)");
    assert_eq!(v1, v2);
}

#[test]
fn histfactory_from_xml_is_fast_enough_smoke() {
    let xml = fixture_combination_xml();

    // Very loose bound to catch accidental O(N^2) / pathological IO regressions without flaking CI.
    let t0 = Instant::now();
    let _ = from_xml(&xml).expect("from_xml");
    let elapsed = t0.elapsed();
    assert!(elapsed < Duration::from_secs(10), "HistFactory ingest too slow: {:?}", elapsed);
}

#[test]
fn histfactory_pyhf_multichannel_shapesys_hist_is_relative_uncertainty() {
    let (xml, basedir) = fixture_pyhf_multichannel();
    let ws = from_xml_with_basedir(&xml, Some(&basedir)).expect("from_xml_with_basedir");

    let ch1 = ws
        .channels
        .iter()
        .find(|c| c.name == "channel1")
        .expect("channel1");
    let bkg = ch1.samples.iter().find(|s| s.name == "bkg").expect("bkg sample");

    // Must keep ShapeSys modifier from XML, but with ABSOLUTE uncertainties.
    let shapesys = bkg
        .modifiers
        .iter()
        .find_map(|m| match m {
            crate::pyhf::schema::Modifier::ShapeSys { name, data } if name == "uncorrshape_signal" => {
                Some(data.clone())
            }
            _ => None,
        })
        .expect("ShapeSys uncorrshape_signal");

    // Fixture convention: the ShapeSys histogram stores RELATIVE uncertainties.
    // Convert to absolute using nominal yields and compare.
    let rf = RootFile::open(&basedir.join("data/data.root")).expect("open data.root");
    let nominal = rf.get_histogram("signal_bkg").expect("nominal signal_bkg");
    let rel = rf.get_histogram("signal_bkgerr").expect("rel signal_bkgerr");
    assert_eq!(nominal.bin_content.len(), rel.bin_content.len());

    let expected_abs: Vec<f64> = rel
        .bin_content
        .iter()
        .zip(nominal.bin_content.iter())
        .map(|(r, n)| r * n)
        .collect();

    assert_eq!(shapesys, expected_abs);
}

#[test]
fn histfactory_pyhf_xmlimport_staterrorconfig_poisson_maps_to_shapesys_and_lumi() {
    let (xml, basedir) = fixture_pyhf_xmlimport();
    let ws = from_xml_with_basedir(&xml, Some(&basedir)).expect("from_xml_with_basedir");

    let ch1 = ws
        .channels
        .iter()
        .find(|c| c.name == "channel1")
        .expect("channel1");
    let bkg1 = ch1
        .samples
        .iter()
        .find(|s| s.name == "background1")
        .expect("background1");

    // background1 has NormalizeByTheory=True in the XML fixture, so it should carry a lumi modifier.
    assert!(
        bkg1.modifiers.iter().any(|m| matches!(m, crate::pyhf::schema::Modifier::Lumi { name, .. } if name == "Lumi")),
        "expected Lumi modifier on background1"
    );

    // StatErrorConfig is Poisson in this fixture, so StatError Activate=True should become ShapeSys (Barlow–Beeston).
    let shapesys = bkg1
        .modifiers
        .iter()
        .find_map(|m| match m {
            crate::pyhf::schema::Modifier::ShapeSys { name, data }
                if name == "staterror_channel1_background1" =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .expect("ShapeSys staterror_channel1_background1");

    // StatError histogram stores RELATIVE uncertainties; convert to absolute.
    let rf = RootFile::open(&basedir.join("data/example.root")).expect("open example.root");
    let nominal = rf.get_histogram("background1").expect("nominal background1");
    let rel = rf
        .get_histogram("background1_statUncert")
        .expect("rel background1_statUncert");
    assert_eq!(nominal.bin_content.len(), rel.bin_content.len());
    let expected_abs: Vec<f64> = rel
        .bin_content
        .iter()
        .zip(nominal.bin_content.iter())
        .map(|(r, n)| r * n)
        .collect();

    assert_eq!(shapesys, expected_abs);

    // The POI normfactor should carry bounds/inits from the XML.
    let meas = ws
        .measurements
        .iter()
        .find(|m| m.name == "GaussExample")
        .expect("GaussExample measurement");
    let poi = meas.config.poi.as_str();
    assert_eq!(poi, "SigXsecOverSM");
    let poi_cfg = meas
        .config
        .parameters
        .iter()
        .find(|p| p.name == poi)
        .expect("POI parameter config exists");
    assert_eq!(poi_cfg.inits, vec![1.0]);
    assert_eq!(poi_cfg.bounds, vec![[0.0, 3.0]]);

    // Lumi should be fixed with the constraint sigma from LumiRelErr.
    let lumi_cfg = meas
        .config
        .parameters
        .iter()
        .find(|p| p.name == "Lumi")
        .expect("Lumi parameter config exists");
    assert!(lumi_cfg.fixed, "expected Lumi fixed via ParamSetting Const=True");
    assert_eq!(lumi_cfg.auxdata, vec![1.0]);
    assert_eq!(lumi_cfg.sigmas, vec![0.1]);
}
