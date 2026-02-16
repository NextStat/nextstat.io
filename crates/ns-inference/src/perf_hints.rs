use std::sync::atomic::{AtomicUsize, Ordering};

// Process-wide hint used by hot model-eval paths. Best-effort only.
static MAMS_CHAIN_HINT: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub(crate) fn set_mams_chain_hint(n_chains: usize) {
    MAMS_CHAIN_HINT.store(n_chains, Ordering::Relaxed);
}

#[inline]
pub(crate) fn clear_mams_chain_hint() {
    MAMS_CHAIN_HINT.store(0, Ordering::Relaxed);
}

#[inline]
pub(crate) fn mams_chain_hint() -> usize {
    MAMS_CHAIN_HINT.load(Ordering::Relaxed)
}
