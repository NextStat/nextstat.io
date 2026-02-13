//! Integration tests: LazyBranchReader parity with BranchReader.
//!
//! Verifies that LazyBranchReader produces bit-identical results to
//! BranchReader on real fixture ROOT files.

use ns_root::RootFile;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

/// Compare lazy single-entry reads with eager full-branch reads.
#[test]
fn lazy_read_f64_at_matches_eager_zlib() {
    lazy_read_f64_at_matches_eager("simple_tree.root");
}

#[test]
fn lazy_read_f64_at_matches_eager_zstd() {
    lazy_read_f64_at_matches_eager("simple_tree_zstd.root");
}

fn lazy_read_f64_at_matches_eager(fixture: &str) {
    let path = fixture_path(fixture);
    if !path.exists() {
        eprintln!("Fixture not found: {fixture}");
        return;
    }

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree 'events'");

    for branch_name in &["pt", "eta", "mbb", "weight_mc"] {
        let eager = f.branch_data(&tree, branch_name).unwrap();
        let lazy = f.lazy_branch_reader(&tree, branch_name).unwrap();

        assert_eq!(lazy.n_entries(), eager.len() as u64);

        // Spot-check entries: first 5, last 5, and middle.
        let n = eager.len();
        let indices: Vec<usize> =
            (0..5).chain((n.saturating_sub(5))..n).chain(std::iter::once(n / 2)).collect();

        for &i in &indices {
            let got = lazy.read_f64_at(i as u64).unwrap();
            assert!(
                (got - eager[i]).abs() < 1e-15,
                "[{fixture}] branch '{branch_name}' entry {i}: lazy={got} eager={}",
                eager[i]
            );
        }
    }
}

/// Compare lazy range reads with eager full-branch reads.
#[test]
fn lazy_read_f64_range_matches_eager_zlib() {
    lazy_read_f64_range_matches_eager("simple_tree.root");
}

#[test]
fn lazy_read_f64_range_matches_eager_zstd() {
    lazy_read_f64_range_matches_eager("simple_tree_zstd.root");
}

fn lazy_read_f64_range_matches_eager(fixture: &str) {
    let path = fixture_path(fixture);
    if !path.exists() {
        return;
    }

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree");

    for branch_name in &["pt", "eta", "mbb", "weight_mc"] {
        let eager = f.branch_data(&tree, branch_name).unwrap();
        let lazy = f.lazy_branch_reader(&tree, branch_name).unwrap();
        let n = eager.len() as u64;

        // Full range
        let full = lazy.read_f64_range(0, n).unwrap();
        assert_eq!(full.len(), eager.len(), "[{fixture}] {branch_name} full range length");
        for (i, (&got, &want)) in full.iter().zip(eager.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-15,
                "[{fixture}] {branch_name}[{i}]: range={got} eager={want}"
            );
        }

        // Sub-range in the middle
        let start = n / 4;
        let end = (3 * n) / 4;
        let sub = lazy.read_f64_range(start, end).unwrap();
        assert_eq!(sub.len(), (end - start) as usize);
        for (i, (&got, &want)) in
            sub.iter().zip(eager[start as usize..end as usize].iter()).enumerate()
        {
            assert!(
                (got - want).abs() < 1e-15,
                "[{fixture}] {branch_name} sub-range [{i}]: got={got} want={want}"
            );
        }
    }
}

/// Compare lazy read_all_f64 with eager as_f64.
#[test]
fn lazy_read_all_matches_eager_zlib() {
    lazy_read_all_matches_eager("simple_tree.root");
}

#[test]
fn lazy_read_all_matches_eager_zstd() {
    lazy_read_all_matches_eager("simple_tree_zstd.root");
}

fn lazy_read_all_matches_eager(fixture: &str) {
    let path = fixture_path(fixture);
    if !path.exists() {
        return;
    }

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree");

    for branch_name in &["pt", "eta", "mbb", "weight_mc", "njet"] {
        let eager = f.branch_data(&tree, branch_name).unwrap();
        let lazy = f.lazy_branch_reader(&tree, branch_name).unwrap();
        let all = lazy.read_all_f64().unwrap();

        assert_eq!(all.len(), eager.len(), "[{fixture}] {branch_name} length");
        for (i, (&got, &want)) in all.iter().zip(eager.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-15,
                "[{fixture}] {branch_name}[{i}]: lazy_all={got} eager={want}"
            );
        }
    }
}

/// Verify ChainedSlice via load_all_chained.
#[test]
fn chained_slice_total_bytes_matches() {
    let path = fixture_path("simple_tree.root");
    if !path.exists() {
        return;
    }

    let f = RootFile::open(&path).expect("failed to open ROOT file");
    let tree = f.get_tree("events").expect("failed to get tree");

    let lazy = f.lazy_branch_reader(&tree, "pt").unwrap();
    let chain = lazy.load_all_chained().unwrap();

    // Total bytes = n_entries * leaf_type.byte_size()
    let elem_size = lazy.branch().leaf_type.byte_size();
    let expected_bytes = lazy.n_entries() as usize * elem_size;
    assert_eq!(
        chain.len(),
        expected_bytes,
        "ChainedSlice total bytes should match n_entries * elem_size (elem_size={elem_size})"
    );
    assert_eq!(chain.n_segments(), lazy.n_baskets());
}
