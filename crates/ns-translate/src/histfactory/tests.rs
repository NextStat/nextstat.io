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
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pyhf_xmlimport");
    (root.join("config/example.xml"), root)
}

fn fixture_pyhf_multichannel() -> (PathBuf, PathBuf) {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pyhf_multichannel");
    (root.join("config/example.xml"), root)
}

fn fixture_trex_export_dir(name: &str) -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/trex_exports");
    let case_dir = root.join(name);
    (case_dir.join("combination.xml"), case_dir)
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
fn histfactory_missing_staterrorconfig_defaults_to_gamma_constraints_for_staterror_params() {
    let xml = fixture_combination_xml();
    let ws = from_xml(&xml).expect("from_xml");

    let ch = ws.channels.iter().find(|c| c.name == "SR").expect("SR channel");
    let bkg = ch.samples.iter().find(|s| s.name == "background").expect("background sample");

    let (stat_name, sigma_abs) = bkg
        .modifiers
        .iter()
        .find_map(|m| match m {
            crate::pyhf::schema::Modifier::StatError { name, data } => Some((name.as_str(), data)),
            _ => None,
        })
        .expect("StatError modifier on background");
    assert_eq!(stat_name, "staterror_SR");
    assert_eq!(sigma_abs.len(), bkg.data.len());

    let meas = ws
        .measurements
        .iter()
        .find(|m| m.name == "NominalMeasurement")
        .expect("NominalMeasurement");

    for (i, (&nom, &sig)) in bkg.data.iter().zip(sigma_abs.iter()).enumerate() {
        if nom <= 0.0 || sig <= 0.0 {
            continue;
        }
        let expected_rel = sig / nom;
        let pname = format!("staterror_SR[{i}]");
        let pcfg = meas
            .config
            .parameters
            .iter()
            .find(|p| p.name == pname)
            .unwrap_or_else(|| panic!("expected parameter config for {pname}"));
        let c = pcfg
            .constraint
            .as_ref()
            .unwrap_or_else(|| panic!("expected Gamma constraint spec for {pname}"));
        assert_eq!(c.constraint_type, "Gamma");
        let got_rel =
            c.rel_uncertainty.unwrap_or_else(|| panic!("expected rel_uncertainty for {pname}"));
        assert!(
            (got_rel - expected_rel).abs() < 1e-12,
            "rel_uncertainty mismatch for {pname}: got={got_rel} expected={expected_rel}"
        );
    }
}

#[test]
fn histfactory_absolute_inputfile_falls_back_to_basedir_basename() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Build a minimal export dir in a temp folder by copying the small local fixture
    // and rewriting the Channel InputFile to a non-existent absolute path.
    let base = std::env::temp_dir().join(format!(
        "nextstat_histfactory_abs_inputfile_{}",
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    fs::create_dir_all(&base).expect("mkdir base");

    let src_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/histfactory");
    fs::copy(src_dir.join("data.root"), base.join("data.root")).expect("copy data.root");

    let combo_src =
        fs::read_to_string(src_dir.join("combination.xml")).expect("read combination.xml");
    fs::write(base.join("combination.xml"), combo_src).expect("write combination.xml");

    let channel_src =
        fs::read_to_string(src_dir.join("channel_SR.xml")).expect("read channel_SR.xml");
    let channel_rewritten =
        channel_src.replace("InputFile=\"data.root\"", "InputFile=\"/nonexistent/data.root\"");
    fs::write(base.join("channel_SR.xml"), channel_rewritten).expect("write channel_SR.xml");

    let ws = from_xml_with_basedir(&base.join("combination.xml"), Some(&base))
        .expect("from_xml_with_basedir with abs fallback");
    assert_eq!(ws.channels.len(), 1);
    assert_eq!(ws.channels[0].name, "SR");

    // Best-effort cleanup (ignore failures).
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn histfactory_trex_export_dirs_ingest_is_deterministic_smoke() {
    // These are HistFactory export directories (combination.xml + data.root + channels/*.xml).
    // They intentionally exercise “real export dir” path semantics beyond the small unit fixtures.
    for name in ["hepdata.116034_DR_Int_EWK", "tttt-prod"] {
        let (xml, basedir) = fixture_trex_export_dir(name);

        let ws1 = from_xml_with_basedir(&xml, Some(&basedir)).expect("from_xml_with_basedir(ws1)");
        let ws2 = from_xml_with_basedir(&xml, Some(&basedir)).expect("from_xml_with_basedir(ws2)");

        assert!(!ws1.channels.is_empty(), "expected channels for fixture: {name}");
        assert!(!ws1.observations.is_empty(), "expected observations for fixture: {name}");

        // Compare as JSON values so map key ordering cannot cause spurious diffs.
        let v1: Value = serde_json::to_value(ws1).expect("to_value(ws1)");
        let v2: Value = serde_json::to_value(ws2).expect("to_value(ws2)");
        assert_eq!(v1, v2, "workspace not deterministic for fixture: {name}");
    }
}

#[test]
fn histfactory_missing_staterrorconfig_disables_small_staterror_bins_by_default() {
    // TREx/ROOT exports often omit `<StatErrorConfig>`. ROOT/HistFactory defaults to
    // `ConstraintType=Poisson` and `RelErrorThreshold=0.05`, so bins with small relative stat
    // errors are pruned (gamma_stat fixed at 1.0).
    let (xml, basedir) = fixture_trex_export_dir("tttt-prod");
    let ws = from_xml_with_basedir(&xml, Some(&basedir)).expect("from_xml_with_basedir");

    let ch = ws.channels.iter().find(|c| c.name == "SR_AllBDT").expect("SR_AllBDT channel");
    let tt_w = ch.samples.iter().find(|s| s.name == "ttW").expect("ttW sample");

    let sigmas = tt_w
        .modifiers
        .iter()
        .find_map(|m| match m {
            crate::pyhf::schema::Modifier::StatError { name, data }
                if name == "staterror_SR_AllBDT" =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .expect("StatError staterror_SR_AllBDT");

    // In this fixture, some bins have rel errors below 0.05 (e.g. idx 3/4), and should be pruned
    // under ROOT defaults when RelErrorThreshold is omitted.
    assert!(sigmas.len() > 16, "expected SR_AllBDT to have 17 bins");
    assert_eq!(sigmas[3], 0.0, "idx 3 should be pruned under ROOT default threshold");
    assert_eq!(sigmas[4], 0.0, "idx 4 should be pruned under ROOT default threshold");
    assert!(sigmas[13] > 0.0, "bin 13 should remain enabled");
    assert!(sigmas[16] > 0.0, "bin 16 should remain enabled");
}

#[test]
fn histfactory_pyhf_multichannel_shapesys_hist_is_relative_uncertainty() {
    let (xml, basedir) = fixture_pyhf_multichannel();
    let ws = from_xml_with_basedir(&xml, Some(&basedir)).expect("from_xml_with_basedir");

    let ch1 = ws.channels.iter().find(|c| c.name == "channel1").expect("channel1");
    let bkg = ch1.samples.iter().find(|s| s.name == "bkg").expect("bkg sample");

    // Must keep ShapeSys modifier from XML, but with ABSOLUTE uncertainties.
    let shapesys = bkg
        .modifiers
        .iter()
        .find_map(|m| match m {
            crate::pyhf::schema::Modifier::ShapeSys { name, data }
                if name == "uncorrshape_signal" =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .expect("ShapeSys uncorrshape_signal");

    // Fixture convention: the ShapeSys histogram stores RELATIVE uncertainties.
    // Convert to absolute using nominal yields and compare.
    let rf = RootFile::open(basedir.join("data/data.root")).expect("open data.root");
    let nominal = rf.get_histogram("signal_bkg").expect("nominal signal_bkg");
    let rel = rf.get_histogram("signal_bkgerr").expect("rel signal_bkgerr");
    assert_eq!(nominal.bin_content.len(), rel.bin_content.len());

    let expected_abs: Vec<f64> =
        rel.bin_content.iter().zip(nominal.bin_content.iter()).map(|(r, n)| r * n).collect();

    assert_eq!(shapesys, expected_abs);
}

#[test]
fn histfactory_pyhf_xmlimport_staterrorconfig_poisson_preserves_staterror_and_lumi() {
    let (xml, basedir) = fixture_pyhf_xmlimport();
    let ws = from_xml_with_basedir(&xml, Some(&basedir)).expect("from_xml_with_basedir");

    let ch1 = ws.channels.iter().find(|c| c.name == "channel1").expect("channel1");
    let bkg1 = ch1.samples.iter().find(|s| s.name == "background1").expect("background1");

    // background1 has NormalizeByTheory=True in the XML fixture, so it should carry a lumi modifier.
    assert!(
        bkg1.modifiers.iter().any(
            |m| matches!(m, crate::pyhf::schema::Modifier::Lumi { name, .. } if name == "Lumi")
        ),
        "expected Lumi modifier on background1"
    );

    // StatErrorConfig is Poisson in this fixture. We preserve `StatError` modifiers and encode
    // Poisson/Gamma constraint semantics via measurement-level constraint metadata.
    let staterror = bkg1
        .modifiers
        .iter()
        .find_map(|m| match m {
            crate::pyhf::schema::Modifier::StatError { name, data }
                if name == "staterror_channel1" =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .expect("StatError staterror_channel1");

    // StatError histogram stores RELATIVE uncertainties; convert to absolute.
    let rf = RootFile::open(basedir.join("data/example.root")).expect("open example.root");
    let nominal = rf.get_histogram("background1").expect("nominal background1");
    let rel = rf.get_histogram("background1_statUncert").expect("rel background1_statUncert");
    assert_eq!(nominal.bin_content.len(), rel.bin_content.len());
    let expected_abs: Vec<f64> =
        rel.bin_content.iter().zip(nominal.bin_content.iter()).map(|(r, n)| r * n).collect();

    assert_eq!(staterror, expected_abs);

    // The POI normfactor should carry bounds/inits from the XML.
    let meas = ws
        .measurements
        .iter()
        .find(|m| m.name == "GaussExample")
        .expect("GaussExample measurement");
    let poi = meas.config.poi.as_str();
    assert_eq!(poi, "SigXsecOverSM");
    let poi_cfg =
        meas.config.parameters.iter().find(|p| p.name == poi).expect("POI parameter config exists");
    assert_eq!(poi_cfg.inits, vec![1.0]);
    assert_eq!(poi_cfg.bounds, vec![[0.0, 3.0]]);

    // Poisson StatErrorConfig should attach Gamma constraint metadata for at least one bin param.
    assert!(
        meas.config.parameters.iter().any(|p| {
            p.name.starts_with("staterror_channel1[")
                && p.constraint.as_ref().is_some_and(|c| c.constraint_type == "Gamma")
        }),
        "expected at least one Gamma constraint spec for staterror_channel1[..]"
    );

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

#[test]
fn histfactory_pyhf_xmlimport_constraintterm_imported_into_measurement_params() {
    let (xml, basedir) = fixture_pyhf_xmlimport();
    let ws = from_xml_with_basedir(&xml, Some(&basedir)).expect("from_xml_with_basedir");

    let gauss = ws
        .measurements
        .iter()
        .find(|m| m.name == "GaussExample")
        .expect("GaussExample measurement");
    assert!(
        gauss.config.parameters.iter().all(|p| p.name != "syst2"),
        "GaussExample should not declare ConstraintTerm param config for syst2"
    );

    let gamma = ws
        .measurements
        .iter()
        .find(|m| m.name == "GammaExample")
        .expect("GammaExample measurement");
    let syst2_gamma = gamma
        .config
        .parameters
        .iter()
        .find(|p| p.name == "syst2")
        .expect("GammaExample should include parameter config for syst2");
    let c = syst2_gamma.constraint.as_ref().expect("GammaExample syst2 constraint spec");
    assert_eq!(c.constraint_type, "Gamma");
    assert_eq!(c.rel_uncertainty, Some(0.3));

    let lognorm = ws
        .measurements
        .iter()
        .find(|m| m.name == "LogNormExample")
        .expect("LogNormExample measurement");
    let syst2_logn = lognorm
        .config
        .parameters
        .iter()
        .find(|p| p.name == "syst2")
        .expect("LogNormExample should include parameter config for syst2");
    let c = syst2_logn.constraint.as_ref().expect("LogNormExample syst2 constraint spec");
    assert_eq!(c.constraint_type, "LogNormal");
    assert_eq!(c.rel_uncertainty, Some(0.3));
}
