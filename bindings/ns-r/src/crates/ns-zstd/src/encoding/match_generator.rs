//! Matching algorithm used find repeated parts in the original data
//!
//! The Zstd format relies on finden repeated sequences of data and compressing these sequences as instructions to the decoder.
//! A sequence basically tells the decoder "Go back X bytes and copy Y bytes to the end of your decode buffer".
//!
//! The task here is to efficiently find matches in the already encoded data for the current suffix of the not yet encoded data.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use super::CompressionLevel;
use super::Matcher;
use super::Sequence;

mod match_generator_hc;
use match_generator_hc::{HashChainMatchGenerator, HashChainStore};

const MIN_MATCH_LEN: usize = 5;
// Roughly in the ballpark of upstream zstd "default" levels.
const DEFAULT_HASH_LOG: u32 = 17;
// Upper bound on hash-chain traversal. The matcher typically requests smaller depths in hot paths.
const DEFAULT_MAX_CHAIN_DEPTH: usize = 4;

/// This is the default implementation of the `Matcher` trait. It allocates and reuses the buffers when possible.
pub struct MatchGeneratorDriver {
    vec_pool: Vec<Vec<u8>>,
    suffix_pool: Vec<SuffixStore>,
    hc_pool: Vec<HashChainStore>,
    match_generator: MatchGenerator,
    hc_generator: HashChainMatchGenerator,
    slice_size: usize,
    use_hash_chain: bool,
    window_size_bytes: usize,
}

impl MatchGeneratorDriver {
    /// slice_size says how big the slices should be that are allocated to work with
    /// max_slices_in_window says how many slices should at most be used while looking for matches
    pub(crate) fn new(slice_size: usize, max_slices_in_window: usize) -> Self {
        Self {
            vec_pool: Vec::new(),
            suffix_pool: Vec::new(),
            hc_pool: Vec::new(),
            match_generator: MatchGenerator::new(max_slices_in_window * slice_size),
            hc_generator: HashChainMatchGenerator::new(max_slices_in_window * slice_size),
            slice_size,
            use_hash_chain: false,
            window_size_bytes: max_slices_in_window * slice_size,
        }
    }
}

impl Matcher for MatchGeneratorDriver {
    fn reset(&mut self, level: CompressionLevel) {
        // Zstd "window size" refers to history distance. Since we match per-block in a separate
        // "space", we must keep (history + current block) bytes resident to enable history matches.
        //
        // Keep Fastest minimal (no cross-block history). For Default, use 256KiB history.
        let (history_slices, store_slices) = match level {
            CompressionLevel::Default => (2usize, 3usize), // 256KiB history + 128KiB current
            _ => (1usize, 1usize), // only current block (no cross-block matches)
        };
        self.use_hash_chain = matches!(level, CompressionLevel::Default);
        self.window_size_bytes = history_slices * self.slice_size;
        self.match_generator.max_window_size = store_slices * self.slice_size;
        self.hc_generator.max_window_size = store_slices * self.slice_size;

        let vec_pool = &mut self.vec_pool;
        let suffix_pool = &mut self.suffix_pool;

        self.match_generator.reset(|mut data, mut suffixes| {
            // Keep the Vec length at capacity so FrameCompressor can read into `&mut data[..]`
            // without re-initializing bytes. This avoids a 128KiB memset per recycled block.
            //
            // SAFETY: `data` comes from our own pool and was originally allocated as fully
            // initialized bytes (Vec<u8>). Truncating does not uninitialize memory, so it's safe
            // to restore length back to capacity without writing.
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(data.capacity());
            }
            vec_pool.push(data);
            suffixes.slots.clear();
            suffixes.slots.resize(suffixes.slots.capacity(), None);
            suffix_pool.push(suffixes);
        });

        let vec_pool = &mut self.vec_pool;
        let hc_pool = &mut self.hc_pool;
        self.hc_generator.reset(|mut data, store| {
            // See comment above: avoid memset when recycling spaces.
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(data.capacity());
            }
            vec_pool.push(data);
            hc_pool.push(store);
        });
    }

    fn window_size(&self) -> u64 {
        self.window_size_bytes as u64
    }

    fn get_next_space(&mut self) -> Vec<u8> {
        self.vec_pool.pop().unwrap_or_else(|| alloc::vec![0; self.slice_size])
    }

    fn get_last_space(&mut self) -> &[u8] {
        if self.use_hash_chain {
            self.hc_generator.get_last_space()
        } else {
            self.match_generator.window.last().unwrap().data.as_slice()
        }
    }

    fn commit_space(&mut self, space: Vec<u8>) {
        if self.use_hash_chain {
            let vec_pool = &mut self.vec_pool;
            let store = self.hc_pool.pop().unwrap_or_else(|| {
                HashChainStore::new(space.len(), DEFAULT_HASH_LOG, DEFAULT_MAX_CHAIN_DEPTH)
            });
            let hc_pool = &mut self.hc_pool;
            self.hc_generator.add_data(space, store, |mut data, store| {
                #[allow(clippy::uninit_vec)]
                unsafe {
                    data.set_len(data.capacity());
                }
                vec_pool.push(data);
                hc_pool.push(store);
            });
        } else {
            let vec_pool = &mut self.vec_pool;
            let suffixes =
                self.suffix_pool.pop().unwrap_or_else(|| SuffixStore::with_capacity(space.len()));
            let suffix_pool = &mut self.suffix_pool;
            self.match_generator.add_data(space, suffixes, |mut data, mut suffixes| {
                #[allow(clippy::uninit_vec)]
                unsafe {
                    data.set_len(data.capacity());
                }
                vec_pool.push(data);
                suffixes.slots.clear();
                suffixes.slots.resize(suffixes.slots.capacity(), None);
                suffix_pool.push(suffixes);
            });
        }
    }

    fn start_matching(&mut self, mut handle_sequence: impl for<'a> FnMut(Sequence<'a>)) {
        let handle_ptr: *mut _ = &mut handle_sequence;
        self.start_matching_split(
            |literals| unsafe {
                // `start_matching_split` drives callbacks synchronously.
                (*handle_ptr)(Sequence::Literals { literals });
            },
            |literals, offset, match_len| {
                unsafe {
                    // See comment above.
                    (*handle_ptr)(Sequence::Triple { literals, offset, match_len });
                }
            },
        );
    }
    fn start_matching_split(
        &mut self,
        mut handle_literals: impl for<'a> FnMut(&'a [u8]),
        mut handle_triple: impl for<'a> FnMut(&'a [u8], usize, usize),
    ) {
        if self.use_hash_chain {
            self.hc_generator.start_matching_split(&mut handle_literals, &mut handle_triple);
        } else {
            while self.match_generator.next_sequence(|seq| match seq {
                Sequence::Literals { literals } => handle_literals(literals),
                Sequence::Triple { literals, offset, match_len } => {
                    handle_triple(literals, offset, match_len)
                }
            }) {}
        }
    }
    fn skip_matching(&mut self) {
        if self.use_hash_chain {
            self.hc_generator.skip_matching();
        } else {
            self.match_generator.skip_matching();
        }
    }
}

#[cfg(all(test, feature = "std", not(target_arch = "wasm32")))]
mod benches {
    use super::{MatchGeneratorDriver, Matcher};
    use crate::encoding::CompressionLevel;
    use alloc::vec::Vec;
    use std::time::Instant;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
    }

    fn gen_rootish_bytes(len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut i: u32 = 0;
        while out.len() + 20 <= len {
            out.extend_from_slice(&i.to_le_bytes());
            out.push((i % 9) as u8);
            out.extend_from_slice(&[0, 0, 0]);

            let pt = 25.0_f32 + (i as f32 % 200.0);
            out.extend_from_slice(&pt.to_le_bytes());
            let eta = -2.5_f32 + (i as f32 % 50.0) * 0.1;
            out.extend_from_slice(&eta.to_le_bytes());
            let w = if i.is_multiple_of(7) { 1.05_f32 } else { 1.0_f32 };
            out.extend_from_slice(&w.to_le_bytes());

            i = i.wrapping_add(1);
        }
        out.resize(len, 0);
        out
    }

    fn mbps(bytes: usize, secs: f64) -> f64 {
        (bytes as f64) / (1024.0 * 1024.0) / secs
    }

    #[test]
    #[ignore]
    fn bench_default_matcher_throughput() {
        let mb = env_usize("NS_ZSTD_MATCH_MB", 32);
        let iters = env_usize("NS_ZSTD_MATCH_ITERS", 10);

        let input = gen_rootish_bytes(mb * 1024 * 1024);
        let mut driver = MatchGeneratorDriver::new(128 * 1024, 1);

        let mut best = 0.0f64;
        let mut total = 0.0f64;

        for _ in 0..iters {
            driver.reset(CompressionLevel::Default);
            let mut seqs = 0usize;

            let t0 = Instant::now();
            for chunk in input.chunks(128 * 1024) {
                let mut space = driver.get_next_space();
                space[..chunk.len()].copy_from_slice(chunk);
                space.truncate(chunk.len());

                driver.commit_space(space);
                driver.start_matching(|_seq| seqs += 1);
            }
            let dt = t0.elapsed().as_secs_f64();
            std::hint::black_box(seqs);

            let tput = mbps(input.len(), dt);
            best = best.max(tput);
            total += tput;
        }

        ::std::eprintln!(
            "default matcher: best={:.1} MiB/s avg={:.1} MiB/s (iters={})",
            best,
            total / iters as f64,
            iters
        );
    }
}

/// This stores the index of a suffix of a string by hashing the first few bytes of that suffix
/// This means that collisions just overwrite and that you need to check validity after a get
struct SuffixStore {
    // We use NonZeroUsize to enable niche optimization here.
    // On store we do +1 and on get -1
    // This is ok since usize::MAX is never a valid offset
    slots: Vec<Option<NonZeroUsize>>,
    len_log: u32,
}

impl SuffixStore {
    fn with_capacity(capacity: usize) -> Self {
        Self { slots: alloc::vec![None; capacity], len_log: capacity.ilog2() }
    }

    #[inline(always)]
    fn insert(&mut self, suffix: &[u8], idx: usize) {
        let key = self.key(suffix);
        self.slots[key] = Some(NonZeroUsize::new(idx + 1).unwrap());
    }

    #[inline(always)]
    fn contains_key(&self, suffix: &[u8]) -> bool {
        let key = self.key(suffix);
        self.slots[key].is_some()
    }

    #[inline(always)]
    fn get(&self, suffix: &[u8]) -> Option<usize> {
        let key = self.key(suffix);
        self.slots[key].map(|x| <NonZeroUsize as Into<usize>>::into(x) - 1)
    }

    #[inline(always)]
    fn key(&self, suffix: &[u8]) -> usize {
        let s0 = suffix[0] as u64;
        let s1 = suffix[1] as u64;
        let s2 = suffix[2] as u64;
        let s3 = suffix[3] as u64;
        let s4 = suffix[4] as u64;

        const POLY: u64 = 0xCF3BCCDCABu64;

        let s0 = (s0 << 24).wrapping_mul(POLY);
        let s1 = (s1 << 32).wrapping_mul(POLY);
        let s2 = (s2 << 40).wrapping_mul(POLY);
        let s3 = (s3 << 48).wrapping_mul(POLY);
        let s4 = (s4 << 56).wrapping_mul(POLY);

        let index = s0 ^ s1 ^ s2 ^ s3 ^ s4;
        let index = index >> (64 - self.len_log);
        index as usize % self.slots.len()
    }
}

/// We keep a window of a few of these entries
/// All of these are valid targets for a match to be generated for
struct WindowEntry {
    data: Vec<u8>,
    /// Stores indexes into data
    suffixes: SuffixStore,
    /// Makes offset calculations efficient
    base_offset: usize,
}

pub(crate) struct MatchGenerator {
    max_window_size: usize,
    /// Data window we are operating on to find matches
    /// The data we want to find matches for is in the last slice
    window: Vec<WindowEntry>,
    window_size: usize,
    #[cfg(debug_assertions)]
    concat_window: Vec<u8>,
    /// Index in the last slice that we already processed
    suffix_idx: usize,
    /// Gets updated when a new sequence is returned to point right behind that sequence
    last_idx_in_sequence: usize,
}

impl MatchGenerator {
    /// max_size defines how many bytes will be used at most in the window used for matching
    fn new(max_size: usize) -> Self {
        Self {
            max_window_size: max_size,
            window: Vec::new(),
            window_size: 0,
            #[cfg(debug_assertions)]
            concat_window: Vec::new(),
            suffix_idx: 0,
            last_idx_in_sequence: 0,
        }
    }

    fn reset(&mut self, mut reuse_space: impl FnMut(Vec<u8>, SuffixStore)) {
        self.window_size = 0;
        #[cfg(debug_assertions)]
        self.concat_window.clear();
        self.suffix_idx = 0;
        self.last_idx_in_sequence = 0;
        self.window.drain(..).for_each(|entry| {
            reuse_space(entry.data, entry.suffixes);
        });
    }

    /// Processes bytes in the current window until either a match is found or no more matches can be found
    /// * If a match is found handle_sequence is called with the Triple variant
    /// * If no more matches can be found but there are bytes still left handle_sequence is called with the Literals variant
    /// * If no more matches can be found and no more bytes are left this returns false
    fn next_sequence(&mut self, mut handle_sequence: impl for<'a> FnMut(Sequence<'a>)) -> bool {
        loop {
            let last_entry = self.window.last().unwrap();
            let data_slice = &last_entry.data;

            // We already reached the end of the window, check if we need to return a Literals{}
            if self.suffix_idx >= data_slice.len() {
                if self.last_idx_in_sequence != self.suffix_idx {
                    let literals = &data_slice[self.last_idx_in_sequence..];
                    self.last_idx_in_sequence = self.suffix_idx;
                    handle_sequence(Sequence::Literals { literals });
                    return true;
                } else {
                    return false;
                }
            }

            // If the remaining data is smaller than the minimum match length we can stop and return a Literals{}
            let data_slice = &data_slice[self.suffix_idx..];
            if data_slice.len() < MIN_MATCH_LEN {
                let last_idx_in_sequence = self.last_idx_in_sequence;
                self.last_idx_in_sequence = last_entry.data.len();
                self.suffix_idx = last_entry.data.len();
                handle_sequence(Sequence::Literals {
                    literals: &last_entry.data[last_idx_in_sequence..],
                });
                return true;
            }

            // This is the key we are looking to find a match for
            let key = &data_slice[..MIN_MATCH_LEN];

            // Look in each window entry
            let mut candidate = None;
            for (match_entry_idx, match_entry) in self.window.iter().enumerate() {
                let is_last = match_entry_idx == self.window.len() - 1;
                if let Some(match_index) = match_entry.suffixes.get(key) {
                    let match_slice = if is_last {
                        &match_entry.data[match_index..self.suffix_idx]
                    } else {
                        &match_entry.data[match_index..]
                    };

                    // Check how long the common prefix actually is
                    let match_len = Self::common_prefix_len(match_slice, data_slice);

                    // Collisions in the suffix store might make this check fail
                    if match_len >= MIN_MATCH_LEN {
                        let offset = match_entry.base_offset + self.suffix_idx - match_index;

                        // If we are in debug/tests make sure the match we found is actually at the offset we calculated
                        #[cfg(debug_assertions)]
                        {
                            let unprocessed = last_entry.data.len() - self.suffix_idx;
                            let start = self.concat_window.len() - unprocessed - offset;
                            let end = start + match_len;
                            let check_slice = &self.concat_window[start..end];
                            debug_assert_eq!(check_slice, &match_slice[..match_len]);
                        }

                        if let Some((old_offset, old_match_len)) = candidate {
                            if match_len > old_match_len
                                || (match_len == old_match_len && offset < old_offset)
                            {
                                candidate = Some((offset, match_len));
                            }
                        } else {
                            candidate = Some((offset, match_len));
                        }
                    }
                }
            }

            if let Some((offset, match_len)) = candidate {
                // For each index in the match we found we do not need to look for another match
                // But we still want them registered in the suffix store
                self.add_suffixes_till(self.suffix_idx + match_len);

                // All literals that were not included between this match and the last are now included here
                let last_entry = self.window.last().unwrap();
                let literals = &last_entry.data[self.last_idx_in_sequence..self.suffix_idx];

                // Update the indexes, all indexes upto and including the current index have been included in a sequence now
                self.suffix_idx += match_len;
                self.last_idx_in_sequence = self.suffix_idx;
                handle_sequence(Sequence::Triple { literals, offset, match_len });

                return true;
            }

            let last_entry = self.window.last_mut().unwrap();
            let key = &last_entry.data[self.suffix_idx..self.suffix_idx + MIN_MATCH_LEN];
            if !last_entry.suffixes.contains_key(key) {
                last_entry.suffixes.insert(key, self.suffix_idx);
            }
            self.suffix_idx += 1;
        }
    }

    /// Find the common prefix length between two byte slices
    #[inline(always)]
    fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
        Self::mismatch_chunks::<8>(a, b)
    }

    /// Find the common prefix length between two byte slices with a configurable chunk length
    /// This enables vectorization optimizations
    fn mismatch_chunks<const N: usize>(xs: &[u8], ys: &[u8]) -> usize {
        let off = core::iter::zip(xs.chunks_exact(N), ys.chunks_exact(N))
            .take_while(|(x, y)| x == y)
            .count()
            * N;
        off + core::iter::zip(&xs[off..], &ys[off..]).take_while(|(x, y)| x == y).count()
    }

    /// Process bytes and add the suffixes to the suffix store up to a specific index
    #[inline(always)]
    fn add_suffixes_till(&mut self, idx: usize) {
        let last_entry = self.window.last_mut().unwrap();
        if last_entry.data.len() < MIN_MATCH_LEN {
            return;
        }
        let slice = &last_entry.data[self.suffix_idx..idx];
        for (key_index, key) in slice.windows(MIN_MATCH_LEN).enumerate() {
            if !last_entry.suffixes.contains_key(key) {
                last_entry.suffixes.insert(key, self.suffix_idx + key_index);
            }
        }
    }

    /// Skip matching for the whole current window entry
    fn skip_matching(&mut self) {
        let len = self.window.last().unwrap().data.len();
        self.add_suffixes_till(len);
        self.suffix_idx = len;
        self.last_idx_in_sequence = len;
    }

    /// Add a new window entry. Will panic if the last window entry hasn't been processed properly.
    /// If any resources are released by pushing the new entry they are returned via the callback
    fn add_data(
        &mut self,
        data: Vec<u8>,
        suffixes: SuffixStore,
        reuse_space: impl FnMut(Vec<u8>, SuffixStore),
    ) {
        assert!(
            self.window.is_empty() || self.suffix_idx == self.window.last().unwrap().data.len()
        );
        self.reserve(data.len(), reuse_space);
        #[cfg(debug_assertions)]
        self.concat_window.extend_from_slice(&data);

        if let Some(last_len) = self.window.last().map(|last| last.data.len()) {
            for entry in self.window.iter_mut() {
                entry.base_offset += last_len;
            }
        }

        let len = data.len();
        self.window.push(WindowEntry { data, suffixes, base_offset: 0 });
        self.window_size += len;
        self.suffix_idx = 0;
        self.last_idx_in_sequence = 0;
    }

    /// Reserve space for a new window entry
    /// If any resources are released by pushing the new entry they are returned via the callback
    fn reserve(&mut self, amount: usize, mut reuse_space: impl FnMut(Vec<u8>, SuffixStore)) {
        assert!(self.max_window_size >= amount);
        while self.window_size + amount > self.max_window_size {
            let removed = self.window.remove(0);
            self.window_size -= removed.data.len();
            #[cfg(debug_assertions)]
            self.concat_window.drain(0..removed.data.len());

            let WindowEntry { suffixes, data: leaked_vec, base_offset: _ } = removed;
            reuse_space(leaked_vec, suffixes);
        }
    }
}

#[test]
fn matches() {
    let mut matcher = MatchGenerator::new(1000);
    let mut original_data = Vec::new();
    let mut reconstructed = Vec::new();

    let assert_seq_equal = |seq1: Sequence<'_>, seq2: Sequence<'_>, reconstructed: &mut Vec<u8>| {
        assert_eq!(seq1, seq2);
        match seq2 {
            Sequence::Literals { literals } => reconstructed.extend_from_slice(literals),
            Sequence::Triple { literals, offset, match_len } => {
                reconstructed.extend_from_slice(literals);
                let start = reconstructed.len() - offset;
                let end = start + match_len;
                reconstructed.extend_from_within(start..end);
            }
        }
    };

    matcher.add_data(
        alloc::vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        SuffixStore::with_capacity(100),
        |_, _| {},
    );
    original_data.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[0, 0, 0, 0, 0], offset: 5, match_len: 5 },
            &mut reconstructed,
        )
    });

    assert!(!matcher.next_sequence(|_| {}));

    matcher.add_data(
        alloc::vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0,],
        SuffixStore::with_capacity(100),
        |_, _| {},
    );
    original_data
        .extend_from_slice(&[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0]);

    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[1, 2, 3, 4, 5, 6], offset: 6, match_len: 6 },
            &mut reconstructed,
        )
    });
    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[], offset: 12, match_len: 6 },
            &mut reconstructed,
        )
    });
    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[], offset: 28, match_len: 5 },
            &mut reconstructed,
        )
    });
    assert!(!matcher.next_sequence(|_| {}));

    matcher.add_data(
        alloc::vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0],
        SuffixStore::with_capacity(100),
        |_, _| {},
    );
    original_data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0]);

    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[], offset: 23, match_len: 6 },
            &mut reconstructed,
        )
    });
    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[7, 8, 9, 10, 11], offset: 16, match_len: 5 },
            &mut reconstructed,
        )
    });
    assert!(!matcher.next_sequence(|_| {}));

    matcher.add_data(alloc::vec![0, 0, 0, 0, 0], SuffixStore::with_capacity(100), |_, _| {});
    original_data.extend_from_slice(&[0, 0, 0, 0, 0]);

    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[], offset: 5, match_len: 5 },
            &mut reconstructed,
        )
    });
    assert!(!matcher.next_sequence(|_| {}));

    matcher.add_data(alloc::vec![7, 8, 9, 10, 11], SuffixStore::with_capacity(100), |_, _| {});
    original_data.extend_from_slice(&[7, 8, 9, 10, 11]);

    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[], offset: 15, match_len: 5 },
            &mut reconstructed,
        )
    });
    assert!(!matcher.next_sequence(|_| {}));

    matcher.add_data(alloc::vec![1, 3, 5, 7, 9], SuffixStore::with_capacity(100), |_, _| {});
    matcher.skip_matching();
    original_data.extend_from_slice(&[1, 3, 5, 7, 9]);
    reconstructed.extend_from_slice(&[1, 3, 5, 7, 9]);
    assert!(!matcher.next_sequence(|_| {}));

    matcher.add_data(alloc::vec![1, 3, 5, 7, 9], SuffixStore::with_capacity(100), |_, _| {});
    original_data.extend_from_slice(&[1, 3, 5, 7, 9]);

    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[], offset: 5, match_len: 5 },
            &mut reconstructed,
        )
    });
    assert!(!matcher.next_sequence(|_| {}));

    matcher.add_data(
        alloc::vec![0, 0, 11, 13, 15, 17, 20, 11, 13, 15, 17, 20, 21, 23],
        SuffixStore::with_capacity(100),
        |_, _| {},
    );
    original_data.extend_from_slice(&[0, 0, 11, 13, 15, 17, 20, 11, 13, 15, 17, 20, 21, 23]);

    matcher.next_sequence(|seq| {
        assert_seq_equal(
            seq,
            Sequence::Triple { literals: &[0, 0, 11, 13, 15, 17, 20], offset: 5, match_len: 5 },
            &mut reconstructed,
        )
    });
    matcher.next_sequence(|seq| {
        assert_seq_equal(seq, Sequence::Literals { literals: &[21, 23] }, &mut reconstructed)
    });
    assert!(!matcher.next_sequence(|_| {}));

    assert_eq!(reconstructed, original_data);
}
