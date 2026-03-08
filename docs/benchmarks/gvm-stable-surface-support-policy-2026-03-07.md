# GVM Stable-Surface Support Policy

**Date**: 2026-03-07  
**Status**: Release-hardening policy draft  
**Scope**: scalar HEP measurement combinations on the GVM engine

## Purpose

This document turns the readiness memo into a concrete release policy.

It accompanies the current stable-first promotion.
Today the foundational GVM subset is documented as `stable`, while the broader
reporting/parity stack remains `research-grade`.

What this document does define is:

- which evidence is required before stable promotion
- what operating envelope is considered supported by current evidence
- what release checklist must be satisfied before a stable designation is
  applied to any GVM surface

## Current policy status

Today the GVM surface is split deliberately:

- `stable` for the foundational inference subset
- `research-grade` for the wider reporting/parity pyramid
- versioned and tested across both layers

## Evidence-backed support envelope

The current repository evidence supports the following envelope:

- literature-backed scalar combination structure around `15x25`
- synthetic stress tiers:
  - `32x24`
  - `64x48`
  - `96x64`
  - `128x96`
- low-`epsilon` multi-start trust evidence for `numerical-paper`
- single-thread and Rayon thread-scaling evidence on:
  - Apple M5
  - AMD EPYC 7502P

This envelope is sufficient for:

- the current stable-first release decision on the **core scalar combination path**
- performance and robustness statements tied to the committed snapshots

This envelope is **not** sufficient for:

- unrestricted scale claims beyond `128x96`
- claims about arbitrary GVM workflows outside scalar measurement combinations
- blanket stability claims for the full reporting/parity pyramid

## Required evidence before further stable promotion

Any additional GVM surface promoted to stable beyond the current first wave
must satisfy all of the following:

1. **Correctness evidence**
   - Rust core tests green
   - CLI tests green
   - Python tests green
   - no unresolved fixture drift

2. **Performance evidence**
   - current benchmark snapshot is published and linked
   - no unexplained regression against the accepted operating envelope

3. **Robustness evidence**
   - current robustness snapshot is published and linked
   - `numerical-paper` trust evidence remains green inside the committed
     tolerance envelopes

4. **Observability**
   - `requested_solver` and `effective_solver` remain surfaced
   - diagnostics contract is documented for the promoted subset

5. **Scope discipline**
   - stable subset is named explicitly
   - non-promoted surfaces remain marked `research-grade`

## Release checklist

Before a release can promote any additional GVM surface to `stable`, all items below must
be completed.

### Contract

- stable subset named explicitly in CLI docs
- stable subset named explicitly in Python docs
- current status wording updated deliberately, not implicitly

### Evidence

- [GVM Benchmark Snapshot](/docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md) up to date
- [GVM NumericalPaper Robustness Snapshot](/docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md) up to date
- evidence envelope still matches the promoted subset

### Verification

- Rust core suite green
- CLI suite green
- Python suite green
- any required slow trust gates green

### Communication

- tutorial updated to distinguish stable subset vs research-grade extensions
- release notes state exactly what was promoted
- advanced reporting/parity layers that remain research-grade are called out
  explicitly

## Stable support classes

This policy uses three support classes:

| Class | Meaning |
| --- | --- |
| `stable` | covered by the public compatibility promise for the designated subset |
| `research-grade` | supported, versioned, and tested, but still evolving without stable-surface promise |

## What stable does not imply

Even after promotion, `stable` would mean:

- the contract is intentionally supported
- the surface has explicit evidence and release backing

It does **not** mean:

- universal scale independence
- arbitrary unrestricted performance guarantees
- immunity from scientific limitations outside the documented envelope

## Relationship to the readiness memo

This policy is the normative companion to:

- [GVM Stable-Surface Readiness Memo](/docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md)

The readiness memo answers:
- "where are we now?"

This policy answers:
- "what must be true before we expand the stable subset further?"

For the currently adopted stable subset itself, see:

- [GVM Stable-First Support Matrix](/docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md)
- [GVM Stable-First Release Notes](/docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md)
- [GVM Stable-First Release Candidate: v0.10.0](/docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md)
- [GVM Stable-First Release PR Checklist](/docs/benchmarks/gvm-stable-first-release-pr-checklist-2026-03-07.md)
- [GVM Stable-First Launch Checklist](/docs/benchmarks/gvm-stable-first-launch-checklist-2026-03-07.md)
