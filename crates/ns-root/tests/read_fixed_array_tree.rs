//! Integration tests: indexed access into fixed-length array branches.

use ns_root::RootFile;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

#[derive(serde::Deserialize)]
struct Expected {
    n_entries: usize,
    eig_0: Vec<f64>,
    eig_1: Vec<f64>,
    eig_3: Vec<f64>,
}

fn load_expected() -> Expected {
    let path = fixture_path("fixed_array_tree_expected.json");
    let text = std::fs::read_to_string(&path).expect("fixed_array_tree_expected.json not found");
    serde_json::from_str(&text).expect("failed to parse expected JSON")
}

#[test]
fn read_indexed_fixed_array_branch_materializes_scalar_columns() {
    let path = fixture_path("fixed_array_tree.root");
    if !path.exists() {
        eprintln!(
            "Fixture not found: run tests/fixtures/generate_root_fixtures.py (fixed_array_tree)"
        );
        return;
    }

    let expected = load_expected();

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree 'events'");
    assert_eq!(tree.entries as usize, expected.n_entries, "entry count mismatch");

    let eig0 = f.branch_data(&tree, "eig[0]").expect("eig[0] should materialize");
    let eig1 = f.branch_data(&tree, "eig[1]").expect("eig[1] should materialize");
    let eig3 = f.branch_data(&tree, "eig[3]").expect("eig[3] should materialize");

    assert_eq!(eig0, expected.eig_0);
    assert_eq!(eig1, expected.eig_1);
    assert_eq!(eig3, expected.eig_3);
}

#[test]
fn fixed_array_oor_returns_default() {
    let path = fixture_path("fixed_array_tree.root");
    if !path.exists() {
        eprintln!(
            "Fixture not found: run tests/fixtures/generate_root_fixtures.py (fixed_array_tree)"
        );
        return;
    }

    let expected = load_expected();

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree 'events'");

    // eig is a 4-element fixed array; eig[10] is OOR and should return 0.0 for all entries
    let eig10 = f.branch_data(&tree, "eig[10]").expect("eig[10] OOR should return Ok, not Err");
    assert_eq!(eig10.len(), expected.n_entries);
    assert!(eig10.iter().all(|&v| v == 0.0), "OOR fixed-array access should yield 0.0");
}
