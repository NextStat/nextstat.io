use ns_root::{CompiledExpr, HistogramSpec, RootFile, fill_histograms};
use ns_translate::ntuple::{ChannelConfig, NtupleWorkspaceBuilder, SampleConfig};
use std::collections::HashMap;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

#[test]
fn ntuple_workspace_builder_weight_sys_and_staterror_are_consistent() {
    let tree_path = fixture_path("simple_tree.root");
    if !tree_path.exists() {
        eprintln!("Fixture not found: run `python tests/fixtures/generate_root_fixtures.py`");
        return;
    }

    let ws = NtupleWorkspaceBuilder::new()
        .ntuple_path(fixture_path("."))
        .tree_name("events")
        .measurement("meas", "mu")
        .add_channel(
            ChannelConfig::new("SR")
                .variable("mbb")
                .binning(&[0., 50., 100., 150., 200., 300.])
                .selection("njet >= 4")
                .add_sample(
                    SampleConfig::new("signal", "simple_tree.root")
                        .weight("weight_mc")
                        .normfactor("mu")
                        .normsys("lumi", 0.9, 1.1)
                        .weight_sys("jes", "weight_jes_up", "weight_jes_down")
                        .staterror(),
                ),
        )
        .build()
        .expect("workspace build");

    assert_eq!(ws.channels.len(), 1);
    let ch = &ws.channels[0];
    assert_eq!(ch.name, "SR");
    assert_eq!(ch.samples.len(), 1);
    let s = &ch.samples[0];
    assert_eq!(s.name, "signal");
    assert_eq!(s.data.len(), 5);

    // Check modifiers exist.
    let mut has_nf = false;
    let mut has_normsys = false;
    let mut has_histosys = false;
    let mut staterror: Option<Vec<f64>> = None;
    let mut histosys_hi: Option<Vec<f64>> = None;
    let mut histosys_lo: Option<Vec<f64>> = None;

    for m in &s.modifiers {
        match m {
            ns_translate::pyhf::schema::Modifier::NormFactor { name, .. } => {
                if name == "mu" {
                    has_nf = true;
                }
            }
            ns_translate::pyhf::schema::Modifier::NormSys { name, .. } => {
                if name == "lumi" {
                    has_normsys = true;
                }
            }
            ns_translate::pyhf::schema::Modifier::HistoSys { name, data } => {
                if name == "jes" {
                    has_histosys = true;
                    histosys_hi = Some(data.hi_data.clone());
                    histosys_lo = Some(data.lo_data.clone());
                }
            }
            ns_translate::pyhf::schema::Modifier::StatError { data, .. } => {
                staterror = Some(data.clone());
            }
            _ => {}
        }
    }

    assert!(has_nf);
    assert!(has_normsys);
    assert!(has_histosys);
    let stat = staterror.expect("missing staterror modifier");
    let hi = histosys_hi.expect("missing histosys hi");
    let lo = histosys_lo.expect("missing histosys lo");
    assert_eq!(stat.len(), s.data.len());
    assert_eq!(hi.len(), s.data.len());
    assert_eq!(lo.len(), s.data.len());

    // StatError must be non-negative (sqrt(sumw2)).
    assert!(stat.iter().all(|&x| x >= 0.0));

    // HistoSys must not be identical to nominal for this fixture (randomized weights).
    assert_ne!(hi, s.data);
    assert_ne!(lo, s.data);

    // Check staterror matches ns_root filler sqrt(sumw2) for nominal.
    let f = RootFile::open(&tree_path).expect("open root");
    let tree = f.get_tree("events").expect("get tree");
    let mbb = f.branch_data(&tree, "mbb").unwrap();
    let weight_mc = f.branch_data(&tree, "weight_mc").unwrap();
    let njet = f.branch_data(&tree, "njet").unwrap();

    let mut columns = HashMap::new();
    columns.insert("mbb".to_string(), mbb);
    columns.insert("weight_mc".to_string(), weight_mc);
    columns.insert("njet".to_string(), njet);

    let spec = HistogramSpec {
        name: "h".into(),
        variable: CompiledExpr::compile("mbb").unwrap(),
        weight: Some(CompiledExpr::compile("weight_mc").unwrap()),
        selection: Some(CompiledExpr::compile("njet >= 4").unwrap()),
        bin_edges: vec![0., 50., 100., 150., 200., 300.],
        flow_policy: ns_root::FlowPolicy::Drop,
        negative_weight_policy: ns_root::NegativeWeightPolicy::Allow,
    };

    let mut filled = fill_histograms(&[spec], &columns).expect("fill");
    let h = filled.pop().unwrap();
    let want: Vec<f64> = h.sumw2.iter().map(|&s2| s2.sqrt()).collect();
    assert_eq!(want.len(), stat.len());
    for (i, (got, w)) in stat.iter().zip(want.iter()).enumerate() {
        assert!((got - w).abs() <= 1e-9, "staterror[{}]: got {} want {}", i, got, w);
    }
}
