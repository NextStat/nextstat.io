//! Integration tests: ROOT-written `std::vector<T>` branch decoding (small offset table).

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
    flat: Vec<f64>,
    offsets: Vec<usize>,
}

fn load_expected() -> Expected {
    let path = fixture_path("unsplit_vector_tree_small_expected.json");
    let text =
        std::fs::read_to_string(&path).expect("unsplit_vector_tree_small_expected.json not found");
    serde_json::from_str(&text).expect("failed to parse expected JSON")
}

#[test]
fn read_unsplit_vector_branch_with_small_entry_offset_len() {
    let path = fixture_path("unsplit_vector_tree_small.root");
    if !path.exists() {
        eprintln!(
            "Fixture not found: run `root -l -b -q tests/fixtures/generate_unsplit_vector_tree_small.C`"
        );
        return;
    }

    let expected = load_expected();

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree 'events'");
    assert_eq!(tree.entries as usize, expected.n_entries, "entry count mismatch");

    let jet_info = tree.find_branch("jet_pt").expect("missing jet_pt branch");
    assert_eq!(jet_info.entry_offset_len, 12, "expected small fEntryOffsetLen for 3-entry basket");

    let jet0 = f.branch_data(&tree, "jet_pt[0]").expect("jet_pt[0] should materialize");
    let jet1 = f.branch_data(&tree, "jet_pt[1]").expect("jet_pt[1] should materialize");
    assert_eq!(jet0, expected.jet_pt_0);
    assert_eq!(jet1, expected.jet_pt_1);

    let jagged = f.branch_data_jagged(&tree, "jet_pt").expect("jet_pt jagged read should work");
    assert_eq!(jagged.flat, expected.flat);
    assert_eq!(jagged.offsets, expected.offsets);
}
