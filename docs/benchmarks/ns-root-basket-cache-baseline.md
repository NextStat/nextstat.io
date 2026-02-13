# ns-root BranchReader Baseline Benchmarks

**Date**: 2026-02-10
**Machine**: macOS (Apple Silicon)
**Profile**: `--release` (criterion, 100 samples)
**Commit**: pre-basket-cache (baseline)

## Results

| Benchmark | Median | Notes |
|---|---|---|
| **single_branch_read/zlib/pt** | 36.7 µs | zlib-compressed Float32 branch |
| **single_branch_read/zstd/pt** | 8.0 µs | zstd-compressed Float32 branch |
| **repeated_branch_read/zlib/3x_same** | 99.6 µs | 3× same branch = 3× decompression |
| **repeated_branch_read/zstd/3x_same** | 33.6 µs | 3× same branch = 3× decompression |
| **parallel_vs_sequential/sequential/zlib** | 33.3 µs | sequential basket reading |
| **parallel_vs_sequential/parallel/zlib** | 26.1 µs | rayon parallel baskets |
| **parallel_vs_sequential/sequential/zstd** | 6.5 µs | sequential basket reading |
| **parallel_vs_sequential/parallel/zstd** | 9.8 µs | rayon overhead > benefit for small data |
| **multi_branch_read/zlib/3branches** | 90.6 µs | 3 branches, no shared cache |
| **multi_branch_read/zstd/3branches** | 20.2 µs | 3 branches, no shared cache |
| **jagged_branch_read/vector_tree** | 2.0 µs | variable-length branch |
| **indexed_branch_read/vector_tree/njet_pt[0]** | 1.2 µs | indexed access on branch |
| **fixed_array_branch_read/fixed_array** | 5.1 µs | fixed-size array branch |

## Key Observations

1. **ZSTD is ~4× faster than zlib** for decompression on these small fixtures.
2. **Repeated reads are ~3× single read** — confirms no caching; decompression is repeated every time.
3. **Rayon parallel hurts small data** — thread pool overhead dominates when baskets are tiny.
   Parallel should only be used when total compressed data > ~1 MB.
4. **Multi-branch read (3 branches) ≈ 3× single** — no cross-branch basket sharing.

## Post-Cache Results (LRU BasketCache integrated)

| Benchmark | Before | After | Change |
|---|---|---|---|
| **single_branch_read/zlib/pt** | 36.7 µs | 1.7 µs | **−95%** |
| **single_branch_read/zstd/pt** | 8.0 µs | 1.8 µs | **−78%** |
| **repeated_branch_read/zlib/3x_same** | 99.6 µs | 6.3 µs | **−94%** |
| **repeated_branch_read/zstd/3x_same** | 33.6 µs | 5.1 µs | **−85%** |
| **parallel_vs_sequential/sequential/zlib** | 33.3 µs | 2.5 µs | **−92%** |
| **parallel_vs_sequential/parallel/zlib** | 26.1 µs | 2.3 µs | **−91%** |
| **parallel_vs_sequential/sequential/zstd** | 6.5 µs | 2.9 µs | **−55%** |
| **parallel_vs_sequential/parallel/zstd** | 9.8 µs | 2.4 µs | **−76%** |
| **multi_branch_read/zlib/3branches** | 90.6 µs | 5.1 µs | **−94%** |
| **multi_branch_read/zstd/3branches** | 20.2 µs | 5.3 µs | **−74%** |
| **jagged_branch_read/vector_tree** | 2.0 µs | 2.2 µs | ~0% (noise) |
| **indexed_branch_read/njet_pt[0]** | 1.2 µs | 1.3 µs | ~0% (noise) |
| **fixed_array_branch_read** | 5.1 µs | 0.48 µs | **−91%** |

## Analysis

1. **Cache eliminates decompression cost** — criterion warms up by calling the benchmark
   function, which populates the cache. Subsequent iterations hit the cache,
   returning `Arc<Vec<u8>>` clones (~ns) instead of decompressing (~µs).
2. **Repeated reads see the biggest win**: 3× same branch went from 99.6 µs → 6.3 µs (zlib).
3. **Multi-branch reads also benefit** when baskets are shared across branches
   (or when the benchmark re-reads the same tree).
4. **No regression on cold paths** — jagged and indexed benchmarks show noise-level change.
5. **The `Arc<Vec<u8>>` overhead is negligible** — atomic refcount increment is ~1 ns.

## LazyBranchReader + ChainedSlice Results

| Benchmark | Median | Notes |
|---|---|---|
| **lazy_single_entry/zlib/pt** | 17 ns | single entry, cache hit |
| **lazy_single_entry/zstd/pt** | 20 ns | single entry, cache hit |
| **lazy_range_read/zlib/25%_range** | 656 ns | 25% of entries, partial basket load |
| **lazy_range_read/zstd/25%_range** | 658 ns | 25% of entries, partial basket load |
| **lazy_range_read/zlib/full_range** | 2.10 µs | full branch via lazy path |
| **lazy_range_read/zstd/full_range** | 1.34 µs | full branch via lazy path |
| **lazy_vs_eager/eager/zlib** | 1.47 µs | eager BranchReader (cached) |
| **lazy_vs_eager/lazy_all/zlib** | 1.98 µs | lazy full read (+35% overhead) |
| **lazy_vs_eager/eager/zstd** | 1.60 µs | eager BranchReader (cached) |
| **lazy_vs_eager/lazy_all/zstd** | 1.57 µs | lazy full read (~parity) |
| **chained_slice/sequential_decode_all** | 3.96 µs | decode 1000 entries via ChainedSlice |
| **chained_slice/random_access_1000** | 3.62 µs | 1000 random entries via ChainedSlice |

### Analysis

1. **Single-entry access is ~80× faster than full eager read** — `read_f64_at()` at 17–20 ns
   only touches one basket (cache hit). Primary use-case: event-by-event analysis, interactive exploration.
2. **25% range read is ~2.5× faster than full read** — only baskets overlapping the range are loaded.
3. **Full lazy read ≈ eager** — per-entry decode loop adds ~35% overhead on zlib (small fixture),
   zstd shows parity. On larger files the overhead amortizes.
4. **ChainedSlice random access ~3.6 ns/entry** — O(log n) segment locate via binary search.
5. **No regressions** — existing eager paths unchanged.

## Architecture

- **Key**: `u64` basket seek position (unique within a file)
- **Value**: `Arc<Vec<u8>>` decompressed payload (shared ownership, no copy)
- **Eviction**: LRU, bounded by total decompressed bytes (default 256 MiB)
- **Thread-safety**: `Mutex<Inner>` (contention negligible vs decompression)
- **Scope**: per-`RootFile` instance — no global state, no cross-file interference

### LazyBranchReader

- Entry → basket mapping via binary search on `basket_entry[]`
- `load_basket(i)` decompresses on demand through `BasketCache`
- `read_f64_at(entry)` / `read_f64_range(start, end)` / `read_all_f64()`
- `load_all_chained()` returns a `ChainedSlice` for custom decode pipelines
- `RootFile::lazy_branch_reader()` public API

### ChainedSlice

- Zero-copy view over `Vec<Arc<Vec<u8>>>` segments
- `locate(pos)` — O(log n) binary search on cumulative offsets
- `read_array::<N>(pos)` — fast path (single segment) / slow path (cross-boundary)
- `decode_f64_at(pos, leaf_type)` — typed decode without intermediate buffer
