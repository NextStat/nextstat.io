//! Integration tests: indexed access into variable-length (jagged) branches.

use ns_root::RootFile;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

#[derive(serde::Deserialize)]
struct Expected {
    n_entries: usize,
    jet_pt_0: Vec<f64>,
    jet_pt_1: Vec<f64>,
}

fn load_expected() -> Expected {
    let path = fixture_path("vector_tree_expected.json");
    let text = std::fs::read_to_string(&path).expect("vector_tree_expected.json not found");
    serde_json::from_str(&text).expect("failed to parse expected JSON")
}

#[test]
fn read_indexed_jagged_branch_materializes_scalar_columns() {
    let path = fixture_path("vector_tree.root");
    if !path.exists() {
        eprintln!("Fixture not found: run tests/fixtures/generate_root_fixtures.py (vector_tree)");
        return;
    }

    let expected = load_expected();

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree 'events'");
    assert_eq!(tree.entries as usize, expected.n_entries, "entry count mismatch");

    let jet_info = tree.find_branch("jet_pt").expect("missing jet_pt branch");
    assert!(jet_info.entry_offset_len > 0, "fixture should have entry offsets");

    let jet0 = f.branch_data(&tree, "jet_pt[0]").expect("jet_pt[0] should materialize");
    let jet1 = f.branch_data(&tree, "jet_pt[1]").expect("jet_pt[1] should materialize");

    assert_eq!(jet0, expected.jet_pt_0);
    assert_eq!(jet1, expected.jet_pt_1);
}
