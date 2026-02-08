//! Integration tests: ROOT-written `std::vector<T>` branch decoding.

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
    let path = fixture_path("unsplit_vector_tree_expected.json");
    let text = std::fs::read_to_string(&path).expect("unsplit_vector_tree_expected.json not found");
    serde_json::from_str(&text).expect("failed to parse expected JSON")
}

#[test]
fn read_indexed_unsplit_vector_branch_materializes_scalar_columns_and_jagged() {
    let path = fixture_path("unsplit_vector_tree.root");
    if !path.exists() {
        eprintln!(
            "Fixture not found: run `root -l -b -q tests/fixtures/generate_unsplit_vector_tree.C`"
        );
        return;
    }

    let expected = load_expected();

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree 'events'");
    assert_eq!(tree.entries as usize, expected.n_entries, "entry count mismatch");

    let jet_info = tree.find_branch("jet_pt").expect("missing jet_pt branch");
    assert!(
        jet_info.entry_offset_len > 0,
        "fixture should have entry offsets for jagged decoding"
    );

    let jet0 = match f.branch_data(&tree, "jet_pt[0]") {
        Ok(v) => v,
        Err(e) => {
            // Debug helper: dump first basket offset table decoding inputs.
            let file_bytes = std::fs::read(&path).expect("read ROOT file bytes");
            let payload =
                ns_root::basket::read_basket_data(&file_bytes, jet_info.basket_seek[0], f.is_large())
                    .expect("read_basket_data");
            eprintln!("jet_pt basket payload len = {}", payload.len());
            eprintln!("jet_pt entry_offset_len = {}", jet_info.entry_offset_len);
            eprintln!("jet_pt basket_entry = {:?}", jet_info.basket_entry);

            let bytes_per_offset = jet_info.entry_offset_len / 8;
            let n_entries = (jet_info.basket_entry[1] - jet_info.basket_entry[0]) as usize;
            let n_offsets = n_entries + 1;
            let tail_bytes = (n_offsets + 1) * bytes_per_offset;
            eprintln!("n_entries={n_entries} bytes_per_offset={bytes_per_offset} tail_bytes={tail_bytes}");

            if payload.len() >= tail_bytes && bytes_per_offset == 4 {
                let data_end = payload.len() - tail_bytes;
                let tail = &payload[data_end..];
                let count = u32::from_be_bytes(tail[0..4].try_into().unwrap());
                eprintln!("count(be)={count}");
                let mut offs = Vec::new();
                for i in 0..n_offsets.min(12) {
                    let s = 4 * (1 + i);
                    let w = u32::from_be_bytes(tail[s..s + 4].try_into().unwrap());
                    offs.push(w);
                }
                eprintln!("offsets(be)[0..{}]={offs:?}", offs.len());
            }

            panic!("jet_pt[0] should materialize: {e:?}");
        }
    };
    let jet1 = f.branch_data(&tree, "jet_pt[1]").expect("jet_pt[1] should materialize");

    assert_eq!(jet0, expected.jet_pt_0);
    assert_eq!(jet1, expected.jet_pt_1);

    let jagged = f.branch_data_jagged(&tree, "jet_pt").expect("jet_pt jagged read should work");
    assert_eq!(jagged.flat, expected.flat);
    assert_eq!(jagged.offsets, expected.offsets);
}
