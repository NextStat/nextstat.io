//! Hash-chain matcher for `CompressionLevel::Default`.
//!
//! Goal: improve compression ratio over the greedy single-slot hash by
//! searching multiple candidates (bounded depth) and doing a basic lazy
//! match check.

use alloc::vec::Vec;

// zstd "default" levels typically use a higher minMatch than fast levels.
// Keep this relatively high to avoid emitting too many short matches.
const MIN_MATCH: usize = 6;
// "Good-enough" bounded match length for fast-path scans.
// 64 keeps ratio stable on our rootish guardrail while reducing matcher CPU.
const TARGET_MATCH_LEN: usize = 64;
// Insert every Nth position into the hash chain.
//
// This is a major throughput lever: inserting every byte is expensive, and level-3 style matching
// usually doesn't need full-density insertion to achieve a good ratio.
const INSERT_STEP: usize = 1;
const NONE_POS: u32 = u32::MAX;
const NONE_PACKED: u64 = u64::MAX;

#[inline(always)]
fn read_u32(ptr: *const u8) -> u32 {
    unsafe { core::ptr::read_unaligned(ptr.cast::<u32>()) }
}

#[inline(always)]
fn read_u64(ptr: *const u8) -> u64 {
    unsafe { core::ptr::read_unaligned(ptr.cast::<u64>()) }
}

#[inline(always)]
fn read_u16(ptr: *const u8) -> u16 {
    unsafe { u16::from_le(core::ptr::read_unaligned(ptr.cast::<u16>())) }
}

#[inline(always)]
fn prefetch_t0(ptr: *const u8) {
    #[allow(unused_variables)]
    let _ = ptr;
    // Best-effort prefetch to overlap random-candidate loads with the cand0 scan.
    // Keep this extremely small and platform-gated; WASM and most non-x86 targets compile it out.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), core::arch::x86_64::_MM_HINT_T0);
    }
}

#[derive(Clone)]
pub(crate) struct HashChainStore {
    // Packs (pos, tag) into a single u64:
    // - low  32 bits: position (u32)
    // - high 16 bits: tag = data[pos+4..pos+6] (u16)
    // This rejects most hash collisions without touching `entry_data` at all.
    head: Vec<u64>,
    chain: Vec<u32>,
    // Per-position 2-byte tag (data[pos+4..pos+6]) for already-inserted positions.
    // Lets us reject most chain candidates without touching entry_data.
    tags: Vec<u16>,
    hash_log: u32,
    max_chain_depth: usize,
}

impl HashChainStore {
    pub(crate) fn new(data_len: usize, hash_log: u32, max_chain_depth: usize) -> Self {
        let head_len = 1usize << hash_log;
        Self {
            // Level 3 uses 128KiB blocks by default, so clearing ~512KiB per block is cheap.
            // This is faster than maintaining a parallel stamp array on every insert.
            head: alloc::vec![NONE_PACKED; head_len],
            // `chain[pos]` is written only for positions inserted into the chain. We never read
            // from non-inserted positions because traversal starts from `head`.
            chain: alloc::vec![0u32; data_len],
            tags: alloc::vec![0u16; data_len],
            hash_log,
            max_chain_depth,
        }
    }

    pub(crate) fn reset_for_len(&mut self, data_len: usize) {
        self.chain.clear();
        self.tags.clear();
        if self.chain.capacity() < data_len {
            self.chain.reserve(data_len - self.chain.capacity());
        }
        if self.tags.capacity() < data_len {
            self.tags.reserve(data_len - self.tags.capacity());
        }
        // Avoid clearing the whole chain: it's safe because we only ever read chain entries for
        // positions that were inserted in this generation.
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.chain.set_len(data_len);
            self.tags.set_len(data_len);
        }
        unsafe {
            // `NONE_POS` is `u32::MAX`, so this is a fast memset to 0xFF.
            core::ptr::write_bytes(self.head.as_mut_ptr(), 0xFF, self.head.len());
        }
    }

    #[inline(always)]
    fn hash4_ptr(&self, ptr: *const u8) -> usize {
        let v = unsafe { core::ptr::read_unaligned(ptr.cast::<u32>()) };
        self.hash4_u32(v)
    }

    #[inline(always)]
    fn hash4_u32(&self, v: u32) -> usize {
        let v = u32::from_le(v);
        let h = v.wrapping_mul(0x9E37_79B1);
        (h >> (32 - self.hash_log)) as usize
    }

    #[inline(always)]
    pub(crate) fn insert(&mut self, data: &[u8], pos: usize) {
        debug_assert!(pos + MIN_MATCH <= data.len());
        let h = self.hash4_ptr(unsafe { data.as_ptr().add(pos) });
        let tag = read_u16(unsafe { data.as_ptr().add(pos + 4) });
        let packed = (pos as u64) | ((tag as u64) << 48);
        unsafe {
            // SAFETY: `h` is always in-bounds by construction of `hash4_ptr`, and `pos` is always
            // a valid chain index for the current generation (inserted_upto <= pos < data.len()).
            let head_ptr = self.head.as_mut_ptr().add(h);
            let prev = *head_ptr as u32;
            *self.chain.get_unchecked_mut(pos) = prev;
            *self.tags.get_unchecked_mut(pos) = tag;
            *head_ptr = packed;
        }
    }

    pub(crate) fn find_best_match(
        &self,
        entry_data: &[u8],
        query: &[u8],
        hash: usize,
        q0: u32,
        q8: Option<u64>,
        max_depth: usize,
    ) -> Option<(usize, usize)> {
        if query.len() < MIN_MATCH {
            return None;
        }
        let h = hash;

        let depth = core::cmp::min(self.max_chain_depth, max_depth);
        if depth == 0 {
            return None;
        }

        // Hot-path: we almost always query with depth=1 (throughput-first Default).
        if depth == 1 {
            let cand = unsafe { *self.head.get_unchecked(h) };
            if cand == NONE_PACKED {
                return None;
            }
            let cpos = (cand as u32) as usize;
            debug_assert!(cpos + MIN_MATCH <= entry_data.len());
            let qtag = read_u16(unsafe { query.as_ptr().add(4) });
            let ctag = (cand >> 48) as u16;
            if ctag != qtag {
                return None;
            }
            // Hash collisions are common; reject early without touching more bytes.
            if let Some(q8) = q8 {
                if cpos + 8 <= entry_data.len() {
                    let c8 = read_u64(unsafe { entry_data.as_ptr().add(cpos) });
                    if c8 != q8 {
                        return None;
                    }
                    // u64 match implies first 4 bytes match too; avoid the extra load/compare.
                    let mlen_bound =
                        common_prefix_len_bounded(&entry_data[cpos..], query, TARGET_MATCH_LEN);
                    if mlen_bound < MIN_MATCH {
                        return None;
                    }
                    if mlen_bound >= TARGET_MATCH_LEN {
                        let full =
                            common_prefix_len_bounded(&entry_data[cpos..], query, query.len());
                        return Some((cpos, full));
                    }
                    return Some((cpos, mlen_bound));
                }
            }
            let c0 = read_u32(unsafe { entry_data.as_ptr().add(cpos) });
            if c0 != q0 {
                return None;
            }
            let mlen_bound =
                common_prefix_len_bounded(&entry_data[cpos..], query, TARGET_MATCH_LEN);
            if mlen_bound < MIN_MATCH {
                return None;
            }
            if mlen_bound >= TARGET_MATCH_LEN {
                let full = common_prefix_len_bounded(&entry_data[cpos..], query, query.len());
                return Some((cpos, full));
            }
            return Some((cpos, mlen_bound));
        }

        // Hot-path for the Default matcher: depth=2 reduces sequence count enough to often be a net win.
        if depth == 2 {
            let cand0 = unsafe { *self.head.get_unchecked(h) };
            if cand0 == NONE_PACKED {
                return None;
            }
            let cpos0 = (cand0 as u32) as usize;
            debug_assert!(cpos0 + MIN_MATCH <= entry_data.len());

            // Pull the next candidate (chain link) early and prefetch its payload while we scan cand0.
            // This is profitable when `entry_data` is larger than L1 and the chain candidate is cold.
            let cand1 = unsafe { *self.chain.get_unchecked(cpos0) };
            if cand1 != NONE_POS {
                let cpos1 = cand1 as usize;
                if cpos1 < entry_data.len() {
                    let ptr = unsafe { entry_data.as_ptr().add(cpos1) };
                    prefetch_t0(ptr);
                }
            }

            // Candidate 0 (most recent) first.
            let mut best_pos = 0usize;
            let mut best_len_bound = 0usize;
            let mut has_best = false;

            let qtag = read_u16(unsafe { query.as_ptr().add(4) });
            let ctag0 = (cand0 >> 48) as u16;
            if ctag0 == qtag {
                let cand0_match = if let Some(q8) = q8 {
                    if cpos0 + 8 <= entry_data.len() {
                        let c8 = read_u64(unsafe { entry_data.as_ptr().add(cpos0) });
                        c8 == q8
                    } else {
                        let c0 = read_u32(unsafe { entry_data.as_ptr().add(cpos0) });
                        c0 == q0
                    }
                } else {
                    let c0 = read_u32(unsafe { entry_data.as_ptr().add(cpos0) });
                    c0 == q0
                };
                if cand0_match {
                    let mlen0_bound =
                        common_prefix_len_bounded(&entry_data[cpos0..], query, TARGET_MATCH_LEN);
                    if mlen0_bound >= MIN_MATCH {
                        best_pos = cpos0;
                        best_len_bound = mlen0_bound;
                        has_best = true;
                        if mlen0_bound >= TARGET_MATCH_LEN {
                            let full =
                                common_prefix_len_bounded(&entry_data[cpos0..], query, query.len());
                            return Some((cpos0, full));
                        }
                        // Throughput-first: if the most recent candidate already yields a decent match,
                        // skip probing the next link in the chain.
                        const EARLY_RETURN_LEN: usize = MIN_MATCH;
                        if mlen0_bound >= EARLY_RETURN_LEN {
                            return Some((cpos0, mlen0_bound));
                        }
                    }
                }
            }

            // Candidate 1 (older link) if needed.
            if cand1 != NONE_POS {
                let cpos1 = cand1 as usize;
                debug_assert!(cpos1 + MIN_MATCH <= entry_data.len());

                let ctag1 = unsafe { *self.tags.get_unchecked(cpos1) };
                if ctag1 == qtag {
                    // If cand0 already matched, cheaply prove cand1 cannot beat it before
                    // loading 8/4 bytes from cand1 payload.
                    let mut do_scan = true;
                    if has_best
                        && best_len_bound < TARGET_MATCH_LEN
                        && best_len_bound < query.len()
                        && cpos1 + best_len_bound < entry_data.len()
                    {
                        unsafe {
                            if *entry_data.get_unchecked(cpos1 + best_len_bound)
                                != *query.get_unchecked(best_len_bound)
                            {
                                do_scan = false;
                            }
                        }
                    }
                    if do_scan {
                        let cand1_match = if let Some(q8) = q8 {
                            if cpos1 + 8 <= entry_data.len() {
                                let c8 = read_u64(unsafe { entry_data.as_ptr().add(cpos1) });
                                c8 == q8
                            } else {
                                let c0 = read_u32(unsafe { entry_data.as_ptr().add(cpos1) });
                                c0 == q0
                            }
                        } else {
                            let c0 = read_u32(unsafe { entry_data.as_ptr().add(cpos1) });
                            c0 == q0
                        };
                        if cand1_match {
                            let mlen1_bound = common_prefix_len_bounded(
                                &entry_data[cpos1..],
                                query,
                                TARGET_MATCH_LEN,
                            );
                            if mlen1_bound >= MIN_MATCH
                                && (!has_best || mlen1_bound > best_len_bound)
                            {
                                best_pos = cpos1;
                                best_len_bound = mlen1_bound;
                                has_best = true;
                            }
                        }
                    }
                }
            }

            if !has_best {
                return None;
            }
            if best_len_bound >= TARGET_MATCH_LEN {
                let full = common_prefix_len_bounded(&entry_data[best_pos..], query, query.len());
                return Some((best_pos, full));
            }
            return Some((best_pos, best_len_bound));
        }

        let first = unsafe { *self.head.get_unchecked(h) };
        if first == NONE_PACKED {
            return None;
        }
        let qtag = read_u16(unsafe { query.as_ptr().add(4) });
        let mut cand = first as u32;
        let mut best: Option<(usize, usize)> = None; // (pos, len_bound)
        for i in 0..depth {
            if cand == NONE_POS {
                break;
            }
            let cpos = cand as usize;
            debug_assert!(cpos + MIN_MATCH <= entry_data.len());

            let ctag = if i == 0 {
                (first >> 48) as u16
            } else {
                unsafe { *self.tags.get_unchecked(cpos) }
            };
            if ctag == qtag {
                let cand_match = if let Some(q8) = q8 {
                    if cpos + 8 <= entry_data.len() {
                        let c8 = read_u64(unsafe { entry_data.as_ptr().add(cpos) });
                        c8 == q8
                    } else {
                        let c0 = read_u32(unsafe { entry_data.as_ptr().add(cpos) });
                        c0 == q0
                    }
                } else {
                    let c0 = read_u32(unsafe { entry_data.as_ptr().add(cpos) });
                    c0 == q0
                };

                if cand_match {
                    let mlen_bound =
                        common_prefix_len_bounded(&entry_data[cpos..], query, TARGET_MATCH_LEN);
                    if mlen_bound >= MIN_MATCH {
                        match best {
                            None => best = Some((cpos, mlen_bound)),
                            Some((best_pos, best_len)) => {
                                if mlen_bound > best_len
                                    || (mlen_bound == best_len && cpos > best_pos)
                                {
                                    // Prefer longer matches; on ties prefer the most recent (closest) match.
                                    best = Some((cpos, mlen_bound));
                                }
                            }
                        }
                        // If we reached the "good enough" threshold, stop searching for longer matches.
                        if mlen_bound >= TARGET_MATCH_LEN {
                            break;
                        }
                    }
                }
            }

            cand = unsafe { *self.chain.get_unchecked(cpos) };
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

    pub(crate) fn start_matching_split(
        &mut self,
        mut handle_literals: impl for<'a> FnMut(&'a [u8]),
        mut handle_triple: impl for<'a> FnMut(&'a [u8], usize, usize),
    ) {
        let last_len = self.window.last().unwrap().data.len();
        if last_len < MIN_MATCH {
            handle_literals(self.window.last().unwrap().data.as_slice());
            self.pos = last_len;
            self.last_literal = last_len;
            return;
        }

        // Standard "skip" heuristic: on long no-match streaks, advance faster to reduce
        // per-byte match queries. Resets to 1 on the next match.
        let mut no_match_streak = 0usize;
        while self.pos + MIN_MATCH <= last_len {
            // Ensure all positions *before* `pos` are in the chain (no self matches).
            debug_assert_eq!(self.window.last().unwrap().inserted_upto, self.pos);
            let m1 = self.find_best_match(self.pos);
            if m1.is_none() {
                // No match: insert this position and move on (it may help future matches).
                {
                    let last = self.window.last_mut().unwrap();
                    if last.inserted_upto == self.pos {
                        last.store.insert(last.data.as_slice(), self.pos);
                        last.inserted_upto = self.pos + INSERT_STEP;
                    }
                }
                no_match_streak = no_match_streak.saturating_add(1);
                let step = (1 + (no_match_streak >> 4)).min(4);
                self.pos += step;
                self.skip_inserts_until(self.pos);
                continue;
            }
            no_match_streak = 0;
            let (off1, len1) = m1.unwrap();

            // Lazy check at pos+1.
            let mut use_pos = self.pos;
            let mut use_off = off1;
            let mut use_len = len1;
            // Keep lazy matching cheap: only check P+1 when the match is "small enough" that
            // it might plausibly be beaten, and only within the current block entry (most wins
            // are local). This keeps ratio benefits while reducing worst-case per-byte overhead.
            if use_len == MIN_MATCH && self.pos + 1 + MIN_MATCH <= last_len {
                {
                    let last = self.window.last_mut().unwrap();
                    if last.inserted_upto == self.pos {
                        last.store.insert(last.data.as_slice(), self.pos);
                        last.inserted_upto = self.pos + INSERT_STEP;
                    }
                }
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
            // Insert skipped positions so future matches can see them.
            //
            // Inserting every byte of a long match region is expensive (hash+chain updates per
            // position) and often brings diminishing ratio returns. For throughput, insert only a
            // short prefix, then advance `inserted_upto` to skip the rest.
            const INSERT_SKIP_THRESHOLD: usize = 8;
            const INSERT_PREFIX: usize = 1;
            if use_len >= INSERT_SKIP_THRESHOLD {
                // Fast-path: with INSERT_PREFIX=1 we only need to insert `use_pos` itself.
                // Avoid the overhead of `insert_until` for this extremely common case.
                debug_assert_eq!(INSERT_PREFIX, 1);
                {
                    let last = self.window.last_mut().unwrap();
                    if last.inserted_upto == use_pos {
                        last.store.insert(last.data.as_slice(), use_pos);
                        last.inserted_upto = use_pos + INSERT_STEP;
                    }
                }
                self.skip_inserts_until(use_pos + use_len);
            } else {
                self.insert_until(use_pos + use_len);
            }

            let data = self.window.last().unwrap().data.as_slice();
            let literals = &data[lit_start..lit_end];
            handle_triple(literals, use_off, use_len);

            self.pos = use_pos + use_len;
            self.last_literal = self.pos;
        }

        if self.last_literal < last_len {
            let data = self.window.last().unwrap().data.as_slice();
            handle_literals(&data[self.last_literal..]);
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
        // Insert positions in `[inserted_upto, upto)` (exclusive upper bound).
        // Only positions <= len - MIN_MATCH are valid match start points.
        let max_valid_exclusive = len - MIN_MATCH + 1;
        let max_insert_exclusive = usize::min(upto, max_valid_exclusive);
        let data = last.data.as_slice();
        let store = &mut last.store;
        let mut pos = last.inserted_upto;
        // Unroll the common case (INSERT_STEP=1) to reduce loop overhead in the hot path.
        while pos + 4 * INSERT_STEP <= max_insert_exclusive {
            store.insert(data, pos);
            store.insert(data, pos + INSERT_STEP);
            store.insert(data, pos + 2 * INSERT_STEP);
            store.insert(data, pos + 3 * INSERT_STEP);
            pos += 4 * INSERT_STEP;
        }
        while pos < max_insert_exclusive {
            store.insert(data, pos);
            pos += INSERT_STEP;
        }
        last.inserted_upto = pos;
    }

    fn skip_inserts_until(&mut self, upto: usize) {
        let last = self.window.last_mut().unwrap();
        let len = last.data.len();
        if len < MIN_MATCH {
            return;
        }
        let max_valid_exclusive = len - MIN_MATCH + 1;
        let upto = usize::min(upto, max_valid_exclusive);
        // Keep insertion parity stable: `insert_until` increments by INSERT_STEP.
        let upto = upto.div_ceil(INSERT_STEP) * INSERT_STEP;
        if upto > last.inserted_upto {
            last.inserted_upto = upto;
        }
    }

    fn find_best_match(&self, pos: usize) -> Option<(usize, usize)> {
        let last = self.window.last().unwrap();
        let data = last.data.as_slice();
        let query = &data[pos..];
        if query.len() < MIN_MATCH {
            return None;
        }
        let q0 = read_u32(query.as_ptr());
        let q8 = (query.len() >= 8).then(|| read_u64(query.as_ptr()));
        let h = last.store.hash4_u32(q0);

        const CURRENT_CHAIN_DEPTH: usize = 2;

        // Current entry first. Throughput-first: only search older history slices when the current
        // block has no match at all.
        if let Some((cpos, mlen)) =
            last.store.find_best_match(data, query, h, q0, q8, CURRENT_CHAIN_DEPTH)
        {
            let offset = pos - cpos;
            if offset != 0 {
                return Some((offset, mlen));
            }
        }

        // Then older entries (history).
        const HISTORY_CHAIN_DEPTH: usize = 1;
        let mut best: Option<(usize, usize)> = None; // (offset, len)
        for entry in self.window[..self.window.len().saturating_sub(1)].iter().rev() {
            if let Some((cpos, mlen)) = entry.store.find_best_match(
                entry.data.as_slice(),
                query,
                h,
                q0,
                q8,
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
        const CURRENT_CHAIN_DEPTH: usize = 2;
        let q0 = read_u32(query.as_ptr());
        let q8 = (query.len() >= 8).then(|| read_u64(query.as_ptr()));
        let h = last.store.hash4_u32(q0);
        let (cpos, mlen) =
            last.store.find_best_match(data, query, h, q0, q8, CURRENT_CHAIN_DEPTH)?;
        let offset = pos - cpos;
        (offset != 0).then_some((offset, mlen))
    }
}

#[inline(always)]
fn common_prefix_len_bounded(a: &[u8], b: &[u8], limit: usize) -> usize {
    let n = limit.min(a.len()).min(b.len());
    let mut i = 0usize;

    // Compare 16 bytes at a time and locate the first mismatch via trailing_zeros.
    // u128 path reduces the number of loads (2 per 16 bytes instead of 4) on little-endian targets.
    #[cfg(target_endian = "little")]
    while i + 16 <= n {
        let x = unsafe {
            let ap = a.as_ptr().add(i).cast::<u128>();
            let bp = b.as_ptr().add(i).cast::<u128>();
            core::ptr::read_unaligned(ap) ^ core::ptr::read_unaligned(bp)
        };
        if x != 0 {
            return i + ((x.trailing_zeros() as usize) >> 3);
        }
        i += 16;
    }

    #[cfg(not(target_endian = "little"))]
    while i + 16 <= n {
        let (x0, x1) = unsafe {
            let ap = a.as_ptr().add(i);
            let bp = b.as_ptr().add(i);
            let a0 = core::ptr::read_unaligned(ap.cast::<u64>());
            let b0 = core::ptr::read_unaligned(bp.cast::<u64>());
            let a1 = core::ptr::read_unaligned(ap.add(8).cast::<u64>());
            let b1 = core::ptr::read_unaligned(bp.add(8).cast::<u64>());
            (a0 ^ b0, a1 ^ b1)
        };
        if x0 != 0 {
            return i + ((x0.trailing_zeros() as usize) >> 3);
        }
        if x1 != 0 {
            return i + 8 + ((x1.trailing_zeros() as usize) >> 3);
        }
        i += 16;
    }

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
