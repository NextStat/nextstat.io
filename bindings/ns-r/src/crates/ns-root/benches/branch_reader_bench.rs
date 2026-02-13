#![allow(clippy::collapsible_if, dead_code)]
//! Baseline benchmarks for BranchReader performance.
//!
//! Measures the current eager basket-reading path before lazy caching is added.
//! Run: cargo bench -p ns-root --bench branch_reader_bench
//!
//! Scenarios:
//!   1. single_branch_f64      — read a Float64 branch once
//!   2. single_branch_f32      — read a Float32 branch once
//!   3. repeated_branch_read   — same branch read 3× (no cache = 3× decompression)
//!   4. parallel_branch_read   — as_f64_par() vs sequential
//!   5. multi_branch_read      — read 3 branches from same tree
//!   6. jagged_branch_read     — read jagged (variable-length) branch
//!   7. indexed_branch_read    — as_f64_indexed on jagged branch

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use std::sync::Arc;

use ns_root::{ChainedSlice, LeafType, RootFile};

// ── Fixture paths ──────────────────────────────────────────────

const SIMPLE_TREE: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/simple_tree.root");
const SIMPLE_TREE_ZSTD: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/simple_tree_zstd.root");
const VECTOR_TREE: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/vector_tree.root");
const FIXED_ARRAY_TREE: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/fixed_array_tree.root");

/// Tree name used in our test fixtures.
const TREE_NAME: &str = "events";

// ── Helpers ────────────────────────────────────────────────────

/// Open a ROOT file, panicking on error (bench setup).
fn open(path: &str) -> RootFile {
    RootFile::open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"))
}

// ── Benchmarks ─────────────────────────────────────────────────

fn bench_single_branch_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_branch_read");

    // simple_tree (zlib)
    if let Ok(f) = RootFile::open(SIMPLE_TREE) {
        if let Ok(tree) = f.get_tree(TREE_NAME) {
            let branches = tree.branch_names();
            if let Some(&first_branch) = branches.first() {
                group.bench_function(BenchmarkId::new("zlib", first_branch), |b| {
                    b.iter(|| {
                        let data = f.branch_data(&tree, first_branch).unwrap();
                        black_box(&data);
                    })
                });
            }
        }
    }

    // simple_tree_zstd
    if let Ok(f) = RootFile::open(SIMPLE_TREE_ZSTD) {
        if let Ok(tree) = f.get_tree(TREE_NAME) {
            let branches = tree.branch_names();
            if let Some(&first_branch) = branches.first() {
                group.bench_function(BenchmarkId::new("zstd", first_branch), |b| {
                    b.iter(|| {
                        let data = f.branch_data(&tree, first_branch).unwrap();
                        black_box(&data);
                    })
                });
            }
        }
    }

    group.finish();
}

fn bench_repeated_branch_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("repeated_branch_read");

    for (label, path) in [("zlib", SIMPLE_TREE), ("zstd", SIMPLE_TREE_ZSTD)] {
        if let Ok(f) = RootFile::open(path) {
            if let Ok(tree) = f.get_tree(TREE_NAME) {
                let branches = tree.branch_names();
                if let Some(&br) = branches.first() {
                    // Read same branch 3 times — simulates formula like `mu * sigma + norm`
                    // where the same branch appears multiple times in the expression tree.
                    group.bench_function(BenchmarkId::new(label, "3x_same"), |b| {
                        b.iter(|| {
                            for _ in 0..3 {
                                let data = f.branch_data(&tree, br).unwrap();
                                black_box(&data);
                            }
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_parallel_branch_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");

    for (label, path) in [("zlib", SIMPLE_TREE), ("zstd", SIMPLE_TREE_ZSTD)] {
        if let Ok(f) = RootFile::open(path) {
            if let Ok(tree) = f.get_tree(TREE_NAME) {
                let branches = tree.branch_names();
                if let Some(&br) = branches.first() {
                    group.bench_function(BenchmarkId::new("sequential", label), |b| {
                        b.iter(|| {
                            let reader = f.branch_reader(&tree, br).unwrap();
                            let data = reader.as_f64().unwrap();
                            black_box(&data);
                        })
                    });
                    group.bench_function(BenchmarkId::new("parallel", label), |b| {
                        b.iter(|| {
                            let reader = f.branch_reader(&tree, br).unwrap();
                            let data = reader.as_f64_par().unwrap();
                            black_box(&data);
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_multi_branch_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_branch_read");

    for (label, path) in [("zlib", SIMPLE_TREE), ("zstd", SIMPLE_TREE_ZSTD)] {
        if let Ok(f) = RootFile::open(path) {
            if let Ok(tree) = f.get_tree(TREE_NAME) {
                let branches = tree.branch_names();
                let n = branches.len().min(3);
                if n > 0 {
                    let target_branches: Vec<&str> = branches[..n].to_vec();
                    group.bench_function(BenchmarkId::new(label, format!("{}branches", n)), |b| {
                        b.iter(|| {
                            for &br in &target_branches {
                                let data = f.branch_data(&tree, br).unwrap();
                                black_box(&data);
                            }
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_jagged_branch_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("jagged_branch_read");

    if let Ok(f) = RootFile::open(VECTOR_TREE) {
        if let Ok(tree) = f.get_tree(TREE_NAME) {
            let branches = tree.branch_names();
            if let Some(&br) = branches.first() {
                group.bench_function("vector_tree", |b| {
                    b.iter(|| {
                        let data = f.branch_data_jagged(&tree, br).unwrap();
                        black_box(&data);
                    })
                });
            }
        }
    }

    group.finish();
}

fn bench_indexed_branch_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexed_branch_read");

    if let Ok(f) = RootFile::open(VECTOR_TREE) {
        if let Ok(tree) = f.get_tree(TREE_NAME) {
            let branches = tree.branch_names();
            if let Some(&br) = branches.first() {
                // Only bench index 0 — some branches are scalar and index > 0 will error
                let idx = 0usize;
                let name = format!("{}[{}]", br, idx);
                group.bench_function(BenchmarkId::new("vector_tree", &name), |b| {
                    b.iter(|| {
                        let reader = f.branch_reader(&tree, br).unwrap();
                        let data = reader.as_f64_indexed(idx, 0.0).unwrap();
                        black_box(&data);
                    })
                });
            }
        }
    }

    group.finish();
}

fn bench_fixed_array_branch_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_array_branch_read");

    if let Ok(f) = RootFile::open(FIXED_ARRAY_TREE) {
        if let Ok(tree) = f.get_tree(TREE_NAME) {
            let branches = tree.branch_names();
            if let Some(&br) = branches.first() {
                group.bench_function("fixed_array", |b| {
                    b.iter(|| {
                        let data = f.branch_data(&tree, br).unwrap();
                        black_box(&data);
                    })
                });
            }
        }
    }

    group.finish();
}

fn bench_lazy_single_entry(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_single_entry");

    for (label, path) in [("zlib", SIMPLE_TREE), ("zstd", SIMPLE_TREE_ZSTD)] {
        if let Ok(f) = RootFile::open(path) {
            if let Ok(tree) = f.get_tree(TREE_NAME) {
                let branches = tree.branch_names();
                if let Some(&br) = branches.first() {
                    let lazy = f.lazy_branch_reader(&tree, br).unwrap();
                    let mid = lazy.n_entries() / 2;
                    group.bench_function(BenchmarkId::new(label, br), |b| {
                        b.iter(|| {
                            let v = lazy.read_f64_at(black_box(mid)).unwrap();
                            black_box(v);
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_lazy_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_range_read");

    for (label, path) in [("zlib", SIMPLE_TREE), ("zstd", SIMPLE_TREE_ZSTD)] {
        if let Ok(f) = RootFile::open(path) {
            if let Ok(tree) = f.get_tree(TREE_NAME) {
                let branches = tree.branch_names();
                if let Some(&br) = branches.first() {
                    let lazy = f.lazy_branch_reader(&tree, br).unwrap();
                    let n = lazy.n_entries();
                    let quarter = n / 4;

                    group.bench_function(BenchmarkId::new(label, "25%_range"), |b| {
                        b.iter(|| {
                            let v = lazy.read_f64_range(quarter, 2 * quarter).unwrap();
                            black_box(&v);
                        })
                    });

                    group.bench_function(BenchmarkId::new(label, "full_range"), |b| {
                        b.iter(|| {
                            let v = lazy.read_all_f64().unwrap();
                            black_box(&v);
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_lazy_vs_eager(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_vs_eager");

    for (label, path) in [("zlib", SIMPLE_TREE), ("zstd", SIMPLE_TREE_ZSTD)] {
        if let Ok(f) = RootFile::open(path) {
            if let Ok(tree) = f.get_tree(TREE_NAME) {
                let branches = tree.branch_names();
                if let Some(&br) = branches.first() {
                    group.bench_function(BenchmarkId::new("eager", label), |b| {
                        b.iter(|| {
                            let data = f.branch_data(&tree, br).unwrap();
                            black_box(&data);
                        })
                    });
                    group.bench_function(BenchmarkId::new("lazy_all", label), |b| {
                        let lazy = f.lazy_branch_reader(&tree, br).unwrap();
                        b.iter(|| {
                            let data = lazy.read_all_f64().unwrap();
                            black_box(&data);
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_chained_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained_slice");

    if let Ok(f) = RootFile::open(SIMPLE_TREE) {
        if let Ok(tree) = f.get_tree(TREE_NAME) {
            let branches = tree.branch_names();
            if let Some(&br) = branches.first() {
                let lazy = f.lazy_branch_reader(&tree, br).unwrap();
                let chain = lazy.load_all_chained().unwrap();
                let elem_size = lazy.branch().leaf_type.byte_size();

                group.bench_function("sequential_decode_all", |b| {
                    b.iter(|| {
                        let n = lazy.n_entries() as usize;
                        let mut out = Vec::with_capacity(n);
                        for i in 0..n {
                            out.push(chain.decode_f64_at(i * elem_size, lazy.branch().leaf_type));
                        }
                        black_box(&out);
                    })
                });

                group.bench_function("random_access_1000", |b| {
                    let n = lazy.n_entries() as usize;
                    let indices: Vec<usize> = (0..1000).map(|i| (i * 7) % n).collect();
                    b.iter(|| {
                        let mut sum = 0.0f64;
                        for &i in &indices {
                            sum += chain.decode_f64_at(i * elem_size, lazy.branch().leaf_type);
                        }
                        black_box(sum);
                    })
                });
            }
        }
    }

    group.finish();
}

fn bench_decode_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_kernels");

    let n = 200_000usize;
    let mut bytes = Vec::with_capacity(n * 8);
    for i in 0..n {
        bytes.extend_from_slice(&(i as f64).to_be_bytes());
    }
    let seg: Arc<[u8]> = Arc::from(bytes);
    let chain = ChainedSlice::new(vec![seg]);

    group.bench_function("chained_f64_200k", |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..n {
                sum += chain.decode_f64_at(i * 8, LeafType::F64);
            }
            black_box(sum);
        })
    });

    group.finish();
}

fn bench_guardrail_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("guardrail_full_scan");

    for (codec, path) in [("zlib", SIMPLE_TREE), ("zstd", SIMPLE_TREE_ZSTD)] {
        if let Ok(f) = RootFile::open(path) {
            if let Ok(tree) = f.get_tree(TREE_NAME) {
                let branches = tree.branch_names();
                if let Some(&br) = branches.first() {
                    group.bench_function(BenchmarkId::new("eager", codec), |b| {
                        b.iter(|| {
                            let v = f.branch_data(&tree, br).unwrap();
                            black_box(v);
                        })
                    });
                    group.bench_function(BenchmarkId::new("lazy", codec), |b| {
                        let lazy = f.lazy_branch_reader(&tree, br).unwrap();
                        b.iter(|| {
                            let v = lazy.read_all_f64().unwrap();
                            black_box(v);
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_branch_read,
    bench_repeated_branch_read,
    bench_parallel_branch_read,
    bench_multi_branch_read,
    bench_jagged_branch_read,
    bench_indexed_branch_read,
    bench_fixed_array_branch_read,
    bench_lazy_single_entry,
    bench_lazy_range,
    bench_lazy_vs_eager,
    bench_chained_slice,
    bench_decode_kernels,
    bench_guardrail_full_scan,
);
criterion_main!(benches);
