//! Integration tests: read TTree from fixture ROOT files.

use ns_root::RootFile;
use std::collections::HashMap;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

const TREE_FIXTURES: &[&str] = &["simple_tree.root", "simple_tree_zstd.root"];

#[derive(serde::Deserialize)]
struct Expected {
    n_entries: u64,
    branches: HashMap<String, BranchExpected>,
    mbb_histogram: HistogramExpected,
    mbb_histogram_selected: HistogramExpectedSimple,
}

#[derive(serde::Deserialize)]
struct BranchExpected {
    #[serde(rename = "type")]
    _type: String,
    first_5: Vec<f64>,
    sum: f64,
}

#[derive(serde::Deserialize)]
struct HistogramExpected {
    bin_edges: Vec<f64>,
    bin_content_unweighted: Vec<f64>,
    bin_content_weighted: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct HistogramExpectedSimple {
    bin_edges: Vec<f64>,
    bin_content_weighted: Vec<f64>,
}

fn load_expected() -> Expected {
    let path = fixture_path("simple_tree_expected.json");
    let text = std::fs::read_to_string(&path).expect("simple_tree_expected.json not found");
    serde_json::from_str(&text).expect("failed to parse expected JSON")
}

#[test]
fn read_tree_metadata() {
    let expected = load_expected();

    for fixture in TREE_FIXTURES {
        let path = fixture_path(fixture);
        if !path.exists() {
            eprintln!("Fixture not found: {fixture}");
            continue;
        }

        let f = RootFile::open(&path).expect("failed to open ROOT file");
        let tree = f.get_tree("events").expect("failed to get tree 'events'");

        assert_eq!(tree.entries, expected.n_entries, "entry count mismatch in {fixture}");

        let names = tree.branch_names();
        eprintln!("[{}] Branches: {:?}", fixture, names);
        assert!(names.contains(&"pt"), "missing branch 'pt' in {fixture}");
        assert!(names.contains(&"eta"), "missing branch 'eta' in {fixture}");
        assert!(names.contains(&"njet"), "missing branch 'njet' in {fixture}");
        assert!(names.contains(&"mbb"), "missing branch 'mbb' in {fixture}");
        assert!(names.contains(&"weight_mc"), "missing branch 'weight_mc' in {fixture}");
    }
}

#[test]
fn read_branch_data() {
    let expected = load_expected();

    for fixture in TREE_FIXTURES {
        let path = fixture_path(fixture);
        if !path.exists() {
            continue;
        }

        let f = RootFile::open(&path).expect("failed to open ROOT file");
        let tree = f.get_tree("events").expect("failed to get tree");

        // Test each branch
        for (branch_name, exp) in &expected.branches {
            eprintln!("[{}] Testing branch: {}", fixture, branch_name);
            let data = f.branch_data(&tree, branch_name).unwrap_or_else(|e| {
                panic!("failed to read branch '{}' in {}: {}", branch_name, fixture, e)
            });

            assert_eq!(
                data.len(),
                expected.n_entries as usize,
                "branch '{}' length mismatch in {}",
                branch_name,
                fixture
            );

            // Check first 5 values
            for (i, (&got, &want)) in data.iter().zip(exp.first_5.iter()).enumerate() {
                let tol = if exp._type == "float32" { 1e-5 } else { 1e-10 };
                assert!(
                    (got - want).abs() < tol,
                    "branch '{}' [{}] in {}: got {} want {} (tol={})",
                    branch_name,
                    i,
                    fixture,
                    got,
                    want,
                    tol
                );
            }

            // Check sum
            let sum: f64 = data.iter().sum();
            let tol = if exp._type == "float32" {
                (exp.sum.abs() * 1e-5).max(1e-3)
            } else {
                (exp.sum.abs() * 1e-10).max(1e-6)
            };
            assert!(
                (sum - exp.sum).abs() < tol,
                "branch '{}' sum in {}: got {} want {} (tol={})",
                branch_name,
                fixture,
                sum,
                exp.sum,
                tol
            );
        }
    }
}

#[test]
fn histogram_from_tree() {
    let expected = load_expected();

    for fixture in TREE_FIXTURES {
        let path = fixture_path(fixture);
        if !path.exists() {
            continue;
        }

        let f = RootFile::open(&path).expect("failed to open ROOT file");
        let tree = f.get_tree("events").expect("failed to get tree");

        // Read required columns
        let mbb = f.branch_data(&tree, "mbb").unwrap();
        let weight_mc = f.branch_data(&tree, "weight_mc").unwrap();
        let njet = f.branch_data(&tree, "njet").unwrap();

        let mut columns = HashMap::new();
        columns.insert("mbb".to_string(), mbb);
        columns.insert("weight_mc".to_string(), weight_mc);
        columns.insert("njet".to_string(), njet);

        // Unweighted histogram
        let spec_unweighted = ns_root::HistogramSpec {
            name: "h_unw".into(),
            variable: ns_root::CompiledExpr::compile("mbb").unwrap(),
            weight: None,
            selection: None,
            bin_edges: expected.mbb_histogram.bin_edges.clone(),
            flow_policy: ns_root::FlowPolicy::Drop,
            negative_weight_policy: ns_root::NegativeWeightPolicy::Allow,
        };

        let results = ns_root::fill_histograms(&[spec_unweighted], &columns).unwrap();
        let h = &results[0];

        for (i, (&got, &want)) in h
            .bin_content
            .iter()
            .zip(expected.mbb_histogram.bin_content_unweighted.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 0.5,
                "unweighted bin[{}] in {}: got {} want {}",
                i,
                fixture,
                got,
                want
            );
        }

        // Weighted histogram
        let spec_weighted = ns_root::HistogramSpec {
            name: "h_w".into(),
            variable: ns_root::CompiledExpr::compile("mbb").unwrap(),
            weight: Some(ns_root::CompiledExpr::compile("weight_mc").unwrap()),
            selection: None,
            bin_edges: expected.mbb_histogram.bin_edges.clone(),
            flow_policy: ns_root::FlowPolicy::Drop,
            negative_weight_policy: ns_root::NegativeWeightPolicy::Allow,
        };

        let results = ns_root::fill_histograms(&[spec_weighted], &columns).unwrap();
        let h = &results[0];

        for (i, (&got, &want)) in
            h.bin_content.iter().zip(expected.mbb_histogram.bin_content_weighted.iter()).enumerate()
        {
            assert!(
                (got - want).abs() < 0.5,
                "weighted bin[{}] in {}: got {} want {}",
                i,
                fixture,
                got,
                want
            );
        }

        // Weighted + selected histogram
        let spec_sel = ns_root::HistogramSpec {
            name: "h_sel".into(),
            variable: ns_root::CompiledExpr::compile("mbb").unwrap(),
            weight: Some(ns_root::CompiledExpr::compile("weight_mc").unwrap()),
            selection: Some(ns_root::CompiledExpr::compile("njet >= 4").unwrap()),
            bin_edges: expected.mbb_histogram_selected.bin_edges.clone(),
            flow_policy: ns_root::FlowPolicy::Drop,
            negative_weight_policy: ns_root::NegativeWeightPolicy::Allow,
        };

        let results = ns_root::fill_histograms(&[spec_sel], &columns).unwrap();
        let h = &results[0];

        for (i, (&got, &want)) in h
            .bin_content
            .iter()
            .zip(expected.mbb_histogram_selected.bin_content_weighted.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 0.5,
                "selected bin[{}] in {}: got {} want {}",
                i,
                fixture,
                got,
                want
            );
        }
    }
}
