//! Hash-chain matcher for `CompressionLevel::Default`.
//!
//! Goal: improve compression ratio over the greedy single-slot hash by
//! searching multiple candidates (bounded depth) and doing a basic lazy
//! match check.

use alloc::vec::Vec;

use super::Sequence;

// zstd "default" levels typically use a higher minMatch than fast levels.
// Keep this relatively high to avoid emitting too many short matches.
const MIN_MATCH: usize = 6;
// Upstream zstd level 3 targets ~256 bytes as a "sufficiently good" match. Past this point,
// searching/exactly extending matches has diminishing returns but significant CPU cost.
const TARGET_MATCH_LEN: usize = 256;
const NONE_POS: u32 = u32::MAX;

#[derive(Clone)]
pub(crate) struct HashChainStore {
    head: Vec<u32>,
    head_stamp: Vec<u32>,
    chain: Vec<u32>,
    cur_stamp: u32,
    hash_log: u32,
    max_chain_depth: usize,
}

impl HashChainStore {
    pub(crate) fn new(data_len: usize, hash_log: u32, max_chain_depth: usize) -> Self {
        let head_len = 1usize << hash_log;
        Self {
            // `head` entries are considered valid only when `head_stamp[h] == cur_stamp`.
            // This avoids O(head_len) clearing on every block reset.
            head: alloc::vec![0u32; head_len],
            head_stamp: alloc::vec![0u32; head_len],
            // `chain[pos]` is written only for positions inserted into the chain. We never read
            // from non-inserted positions because traversal starts from `head`.
            chain: alloc::vec![0u32; data_len],
            cur_stamp: 1,
            hash_log,
            max_chain_depth,
        }
    }

    pub(crate) fn reset_for_len(&mut self, data_len: usize) {
        self.chain.clear();
        if self.chain.capacity() < data_len {
            self.chain.reserve(data_len - self.chain.capacity());
        }
        // Avoid clearing the whole chain: it's safe because we only ever read chain entries for
        // positions that were inserted in this generation.
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.chain.set_len(data_len);
        }

        self.cur_stamp = self.cur_stamp.wrapping_add(1);
        // If we wrapped, clear stamps (rare).
        if self.cur_stamp == 0 {
            self.head_stamp.fill(0);
            self.cur_stamp = 1;
        }
    }

    #[inline(always)]
    fn hash4(&self, bytes: &[u8]) -> usize {
        debug_assert!(bytes.len() >= 4);
        let v = unsafe { core::ptr::read_unaligned(bytes.as_ptr().cast::<u32>()) };
        let v = u32::from_le(v);
        let h = v.wrapping_mul(0x9E37_79B1);
        (h >> (32 - self.hash_log)) as usize
    }

    #[inline(always)]
    pub(crate) fn insert(&mut self, data: &[u8], pos: usize) {
        if pos + MIN_MATCH > data.len() {
            return;
        }
        let h = self.hash4(&data[pos..pos + 4]);
        let prev = if self.head_stamp[h] == self.cur_stamp { self.head[h] } else { NONE_POS };
        self.chain[pos] = prev;
        self.head[h] = pos as u32;
        self.head_stamp[h] = self.cur_stamp;
    }

    pub(crate) fn find_best_match(
        &self,
        entry_data: &[u8],
        _query_pos: usize,
        query: &[u8],
        // For matches within the current (last) entry, forbid candidates at/after query_pos.
        max_candidate_pos: Option<usize>,
        max_depth: usize,
    ) -> Option<(usize, usize)> {
        if query.len() < MIN_MATCH {
            return None;
        }
        let h = self.hash4(&query[..4]);
        let mut cand = if self.head_stamp[h] == self.cur_stamp { self.head[h] } else { NONE_POS };
        let mut best: Option<(usize, usize)> = None; // (pos, len_bound)

        let depth = core::cmp::min(self.max_chain_depth, max_depth);
        for _ in 0..depth {
            if cand == NONE_POS {
                break;
            }
            let cpos = cand as usize;
            if let Some(max_pos) = max_candidate_pos {
                if cpos >= max_pos {
                    cand = self.chain[cpos];
                    continue;
                }
            }
            if cpos + MIN_MATCH > entry_data.len() {
                cand = self.chain[cpos];
                continue;
            }

            let mlen_bound =
                common_prefix_len_bounded(&entry_data[cpos..], query, TARGET_MATCH_LEN);
            if mlen_bound >= MIN_MATCH {
                match best {
                    None => best = Some((cpos, mlen_bound)),
                    Some((best_pos, best_len)) => {
                        if mlen_bound > best_len || (mlen_bound == best_len && cpos > best_pos) {
                            // Prefer longer matches; on ties prefer the most recent (closest) match.
                            best = Some((cpos, mlen_bound));
                        }
                    }
                }
            }

            // If we reached the "good enough" threshold, stop searching for longer matches.
            if mlen_bound >= TARGET_MATCH_LEN {
                break;
            }

            cand = self.chain[cpos];
        }

        // If we found a sufficiently-long match, extend it fully once for better ratio.
        if let Some((best_pos, best_len_bound)) = best {
            if best_len_bound >= TARGET_MATCH_LEN {
                let full = common_prefix_len_bounded(&entry_data[best_pos..], query, query.len());
                return Some((best_pos, full));
            }
        }
        best
    }
}

pub(crate) struct WindowEntryHc {
    pub(crate) data: Vec<u8>,
    pub(crate) store: HashChainStore,
    pub(crate) base_offset: usize,
    pub(crate) inserted_upto: usize,
}

pub(crate) struct HashChainMatchGenerator {
    pub(crate) max_window_size: usize,
    pub(crate) window: Vec<WindowEntryHc>,
    pub(crate) window_size: usize,

    pos: usize,
    last_literal: usize,
}

impl HashChainMatchGenerator {
    pub(crate) fn new(max_window_size: usize) -> Self {
        Self { max_window_size, window: Vec::new(), window_size: 0, pos: 0, last_literal: 0 }
    }

    pub(crate) fn reset(&mut self, mut reuse_space: impl FnMut(Vec<u8>, HashChainStore)) {
        self.window_size = 0;
        self.pos = 0;
        self.last_literal = 0;
        self.window.drain(..).for_each(|entry| {
            reuse_space(entry.data, entry.store);
        });
    }

    pub(crate) fn get_last_space(&self) -> &[u8] {
        self.window.last().unwrap().data.as_slice()
    }

    pub(crate) fn add_data(
        &mut self,
        data: Vec<u8>,
        mut store: HashChainStore,
        reuse_space: impl FnMut(Vec<u8>, HashChainStore),
    ) {
        assert!(self.window.is_empty() || self.pos == self.window.last().unwrap().data.len());
        self.reserve(data.len(), reuse_space);

        if let Some(last_len) = self.window.last().map(|last| last.data.len()) {
            for entry in self.window.iter_mut() {
                entry.base_offset += last_len;
            }
        }

        let len = data.len();
        store.reset_for_len(len);
        self.window.push(WindowEntryHc { data, store, base_offset: 0, inserted_upto: 0 });
        self.window_size += len;
        self.pos = 0;
        self.last_literal = 0;
    }

    pub(crate) fn skip_matching(&mut self) {
        let len = self.window.last().unwrap().data.len();
        self.insert_until(len);
        self.pos = len;
        self.last_literal = len;
    }

    pub(crate) fn start_matching(&mut self, mut handle_sequence: impl for<'a> FnMut(Sequence<'a>)) {
        let last_len = self.window.last().unwrap().data.len();
        if last_len < MIN_MATCH {
            handle_sequence(Sequence::Literals {
                literals: self.window.last().unwrap().data.as_slice(),
            });
            self.pos = last_len;
            self.last_literal = last_len;
            return;
        }

        while self.pos + MIN_MATCH <= last_len {
            // Ensure all positions up to current pos are in the chain.
            self.insert_until(self.pos);
            let m1 = self.find_best_match(self.pos);
            if m1.is_none() {
                self.pos += 1;
                continue;
            }
            let (off1, len1) = m1.unwrap();

            // Lazy check at pos+1.
            let mut use_pos = self.pos;
            let mut use_off = off1;
            let mut use_len = len1;
            // Keep lazy matching cheap: only check P+1 when the match is "small enough" that
            // it might plausibly be beaten, and only within the current block entry (most wins
            // are local). This keeps ratio benefits while reducing worst-case per-byte overhead.
            const LAZY_CHECK_MAX_LEN: usize = 64;
            if use_len < LAZY_CHECK_MAX_LEN && self.pos + 1 + MIN_MATCH <= last_len {
                self.insert_until(self.pos + 1);
                if let Some((off2, len2)) = self.find_best_match_last_entry(self.pos + 1) {
                    if len2 > len1 {
                        use_pos = self.pos + 1;
                        use_off = off2;
                        use_len = len2;
                    }
                }
            }

            // Emit sequence at use_pos.
            let lit_start = self.last_literal;
            let lit_end = use_pos;
            // Insert all skipped positions (including the match region) so future matches can see them.
            self.insert_until(use_pos + use_len);

            let data = self.window.last().unwrap().data.as_slice();
            let literals = &data[lit_start..lit_end];
            handle_sequence(Sequence::Triple { literals, offset: use_off, match_len: use_len });

            self.pos = use_pos + use_len;
            self.last_literal = self.pos;
        }

        if self.last_literal < last_len {
            let data = self.window.last().unwrap().data.as_slice();
            handle_sequence(Sequence::Literals { literals: &data[self.last_literal..] });
        }
        self.pos = last_len;
        self.last_literal = last_len;
    }

    fn reserve(&mut self, amount: usize, mut reuse_space: impl FnMut(Vec<u8>, HashChainStore)) {
        assert!(self.max_window_size >= amount);
        while self.window_size + amount > self.max_window_size {
            let removed = self.window.remove(0);
            self.window_size -= removed.data.len();
            reuse_space(removed.data, removed.store);
        }
    }

    fn insert_until(&mut self, upto: usize) {
        let last = self.window.last_mut().unwrap();
        let len = last.data.len();
        if len < MIN_MATCH {
            return;
        }
        let max_insert = usize::min(upto, len - MIN_MATCH);
        while last.inserted_upto <= max_insert {
            last.store.insert(last.data.as_slice(), last.inserted_upto);
            last.inserted_upto += 1;
        }
    }

    fn find_best_match(&self, pos: usize) -> Option<(usize, usize)> {
        let last = self.window.last().unwrap();
        let data = last.data.as_slice();
        let query = &data[pos..];
        if query.len() < MIN_MATCH {
            return None;
        }

        // If we already have a "reasonably long" match in the current block, skip searching older
        // history slices: it's a major CPU win and rarely improves compression meaningfully.
        const HISTORY_SKIP_LEN: usize = 64;

        let mut best: Option<(usize, usize)> = None; // (offset, len)

        // Current entry first.
        if let Some((cpos, mlen)) =
            last.store.find_best_match(data, pos, query, Some(pos), usize::MAX)
        {
            let offset = pos - cpos;
            if offset != 0 {
                best = Some((offset, mlen));
                if mlen >= HISTORY_SKIP_LEN {
                    return best;
                }
            }
        }

        // Then older entries (history).
        const HISTORY_CHAIN_DEPTH: usize = 1;
        for entry in self.window[..self.window.len().saturating_sub(1)].iter().rev() {
            if let Some((cpos, mlen)) = entry.store.find_best_match(
                entry.data.as_slice(),
                pos,
                query,
                None,
                HISTORY_CHAIN_DEPTH,
            ) {
                let offset = entry.base_offset + pos - cpos;
                if offset == 0 {
                    continue;
                }
                match best {
                    None => best = Some((offset, mlen)),
                    Some((best_off, best_len)) => {
                        if mlen > best_len || (mlen == best_len && offset < best_off) {
                            best = Some((offset, mlen));
                        }
                    }
                }
            }

            if let Some((_off, best_len)) = best {
                if best_len >= TARGET_MATCH_LEN {
                    break;
                }
            }
        }
        best
    }

    #[inline(always)]
    fn find_best_match_last_entry(&self, pos: usize) -> Option<(usize, usize)> {
        let last = self.window.last().unwrap();
        let data = last.data.as_slice();
        let query = &data[pos..];
        if query.len() < MIN_MATCH {
            return None;
        }
        let (cpos, mlen) = last.store.find_best_match(data, pos, query, Some(pos), usize::MAX)?;
        let offset = pos - cpos;
        (offset != 0).then_some((offset, mlen))
    }
}

#[inline(always)]
fn common_prefix_len_bounded(a: &[u8], b: &[u8], limit: usize) -> usize {
    let n = limit.min(a.len()).min(b.len());
    let mut i = 0usize;

    // Compare 8 bytes at a time and locate the first mismatch via trailing_zeros.
    while i + 8 <= n {
        let x = unsafe {
            let ap = a.as_ptr().add(i).cast::<u64>();
            let bp = b.as_ptr().add(i).cast::<u64>();
            core::ptr::read_unaligned(ap) ^ core::ptr::read_unaligned(bp)
        };
        if x != 0 {
            return i + ((x.trailing_zeros() as usize) >> 3);
        }
        i += 8;
    }

    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}
