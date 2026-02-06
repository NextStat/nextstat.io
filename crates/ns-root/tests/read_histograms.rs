//! Integration tests: read TH1D histograms from fixture ROOT files.

use ns_root::RootFile;
use std::collections::HashMap;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[derive(serde::Deserialize)]
struct ExpectedHist {
    n_bins: usize,
    x_min: f64,
    x_max: f64,
    bin_content: Vec<f64>,
    bin_edges: Vec<f64>,
}

#[test]
fn read_simple_histos() {
    let root_path = fixture_path("simple_histos.root");
    if !root_path.exists() {
        eprintln!(
            "Fixture not found: {:?}. Run `python tests/fixtures/generate_root_fixtures.py` first.",
            root_path
        );
        return;
    }

    let expected_path = fixture_path("simple_histos_expected.json");
    let expected: HashMap<String, ExpectedHist> =
        serde_json::from_str(&std::fs::read_to_string(&expected_path).unwrap()).unwrap();

    let f = RootFile::open(&root_path).expect("failed to open ROOT file");

    // List keys
    let keys = f.list_keys().expect("failed to list keys");
    assert!(!keys.is_empty(), "expected at least one key");
    eprintln!("Keys: {:?}", keys.iter().map(|k| &k.name).collect::<Vec<_>>());

    // Test each expected histogram
    for (path, exp) in &expected {
        eprintln!("Reading histogram: {}", path);
        let h = f.get_histogram(path).unwrap_or_else(|e| {
            panic!("failed to read '{}': {}", path, e);
        });

        assert_eq!(h.n_bins, exp.n_bins, "{}: n_bins mismatch", path);
        assert!(
            (h.x_min - exp.x_min).abs() < 1e-10,
            "{}: x_min mismatch: {} vs {}",
            path,
            h.x_min,
            exp.x_min
        );
        assert!(
            (h.x_max - exp.x_max).abs() < 1e-10,
            "{}: x_max mismatch: {} vs {}",
            path,
            h.x_max,
            exp.x_max
        );

        assert_eq!(
            h.bin_content.len(),
            exp.bin_content.len(),
            "{}: bin_content length mismatch",
            path,
        );
        for (i, (got, want)) in h.bin_content.iter().zip(exp.bin_content.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "{}: bin_content[{}] mismatch: {} vs {}",
                path,
                i,
                got,
                want,
            );
        }

        assert_eq!(
            h.bin_edges.len(),
            exp.bin_edges.len(),
            "{}: bin_edges length mismatch",
            path,
        );
        for (i, (got, want)) in h.bin_edges.iter().zip(exp.bin_edges.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "{}: bin_edges[{}] mismatch: {} vs {}",
                path,
                i,
                got,
                want,
            );
        }
    }
}

#[test]
fn read_histfactory_data() {
    let root_path = fixture_path("histfactory/data.root");
    if !root_path.exists() {
        eprintln!(
            "Fixture not found: {:?}. Run `python tests/fixtures/generate_root_fixtures.py` first.",
            root_path
        );
        return;
    }

    let f = RootFile::open(&root_path).expect("failed to open ROOT file");
    let keys = f.list_keys().expect("failed to list keys");
    eprintln!("HF Keys: {:?}", keys.iter().map(|k| format!("{} ({})", k.name, k.class_name)).collect::<Vec<_>>());

    // Read histograms from subdirectory
    let obs = f.get_histogram("SR/data_obs").expect("failed to read SR/data_obs");
    assert_eq!(obs.n_bins, 3);
    assert!((obs.bin_content[0] - 15.0).abs() < 1e-10);
    assert!((obs.bin_content[1] - 25.0).abs() < 1e-10);
    assert!((obs.bin_content[2] - 12.0).abs() < 1e-10);

    let sig = f.get_histogram("SR/signal_nominal").expect("failed to read SR/signal_nominal");
    assert_eq!(sig.n_bins, 3);
    assert!((sig.bin_content[0] - 5.0).abs() < 1e-10);
    assert!((sig.bin_content[1] - 10.0).abs() < 1e-10);
    assert!((sig.bin_content[2] - 3.0).abs() < 1e-10);
}
