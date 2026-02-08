<!--
Market Research Note (lightweight)
Generated: 2026-02-08 18:30:00
Git Commit: ca857674a0831f2fed212641caec1aaed2e2ef52
Scope: market landscape + prioritization
-->

# NextStat Market Research (Feb 2026)

This is **not** a “TAM/SAM/SOM” slide deck. It’s a pragmatic landscape + what to build next to win trust and adoption.

## 1) What I actually researched (today)

I reviewed:

- Repo state: `README.md`, `docs/benchmarks.md`, `docs/pyhf-parity-contract.md`, `docs/WHITEPAPER.md`, GPU/server surfaces, Apex2 harness.
- External landscape (sources linked below): open-source + commercial pharmacometrics tooling and the HEP HistFactory ecosystem.

## 2) Landscape (competitors / alternatives)

### HEP (HistFactory)

- **pyhf**: pure-Python HistFactory implementation; supports ML backends (TF/PyTorch/JAX) and is actively used/published in the community.
  - Sources: https://github.com/scikit-hep/pyhf and https://arxiv.org/abs/2211.15838

Implication: benchmarking vs pyhf is the right “spec oracle + adoption anchor”. You already do this in-repo (Apex2 + God Run).

### Pharma / PK-NLME

There is a real competitive set beyond NONMEM/Monolix:

- **nlmixr2** (open-source R): explicitly positions itself as open-source NLME for PopPK/PKPD and “suitable for regulatory submissions”.
  - Source: https://nlmixr2.org/
- **Torsten** (open-source): PK/PD library for Stan, NMTRAN/PREDPP-like event conventions, used in Bayesian pharmacometrics workflows.
  - Sources: https://metrumresearchgroup.github.io/Torsten/ and https://github.com/metrumresearchgroup/Torsten
- **Pumas** (commercial): “comprehensive platform” for pharmacometrics workflows; emphasizes scaling and ML integration.
  - Source: https://juliahub.com/products/pumas/
- **NONMEM** (commercial): ICON states it is licensed via an **annual license subscription fee** (fee changes; optional renewal).
  - Source: https://www.iconplc.com/solutions/technologies/nonmem
- **MonolixSuite** (commercial): Lixoft documentation describes a term license “renewable on an annual basis”; academic/non-profit can get free licenses for non-commercial.
  - Source: https://lixoft.com/downloads/ and license doc excerpts in https://lixoft.com/downloads/full-documentation/

Implication: for “Trust Offensive” in Pharma, *publicly reproducible* comparisons should prioritize **nlmixr2** and **Torsten** first, then include NONMEM/Monolix as “methodology + optional run” because pricing/runtime access is not universally reproducible.

### Econometrics / Time Series

Market is fragmented; users care less about “new engine” and more about:

- streaming latency, deterministic backtests, and integration into their stack.
- easy APIs (Python/R), and credible validation vs known references (scalar references, optionally statsmodels).

Implication: here the wedge is not “feature completeness”, it’s a **real-time demo** + **validation report** that looks like an engineering artifact.

## 3) What from your list is актуально vs неактуально (Feb 2026)

### Direction 1: Trust Offensive

**Актуально (top priority).**

But: much of it already exists *inside the repo*:
- “God Run” vs pyhf is already in `README.md`.
- Apex2 validation harness + baseline manifests are already documented in `docs/benchmarks.md`.
- Numerical parity contract is already formalized in `docs/pyhf-parity-contract.md`.

What’s missing is packaging + “trust artifacts”:
- a **public benchmarks repo** (or a `benchmarks/` distribution) with pinned environments and “press-ready” tables.
- a **Validation Report PDF** generator (GxP-shaped artifact).

**Not актуально / needs reframing:** “vs ROOT/RooFit (время, точность NLL)” only makes sense on clearly defined comparable surfaces. RooFit is not a single “spec oracle” the way pyhf is; treat it as *workflow parity* (HistFactory XML import + export parity), not as a single NLL benchmark headline.

### Direction 2: AI for Science Bridge

**Актуально, но только если сделать это как demo+tooling, не как ‘AGI’.**

Repo already has:
- PyTorch layers and ML hooks (SignificanceLoss/SoftHistogram notebooks exist).
- A server mode (`ns-server`) and a native ROOT reader, which are the building blocks for agent demos.

What’s missing:
- a stable tool API (JSON schema, determinism, seed policy).
- one killer demo end-to-end (Physics Assistant or Discovery Pipeline).

### Direction 3: Vertical Expansion

**Актуально**, but the market reality is:
- Pharma adoption requires validation artifacts and R integration (at minimum).
- Quants adoption requires hard latency numbers + integration story.

So “vertical expansion” is blocked primarily by **trust artifacts + connectors**, not by core math.

### Backlog section (docs/connectors/community)

- **Documentation Translation**: актуально и срочно. It’s the cheapest way to unlock “non-HEP brain” adoption.
- **Ecosystem Connectors**: актуально. Python sklearn adapters already exist in repo; R bindings are the big missing piece for pharma/econ.
- **Community (Discord/Weekly Challenge)**: актуально позже. Without public benchmark+validation artifacts, community will be noise.

## 4) What I would add (missing from your list)

1) **Pharma: add nlmixr2 + Torsten comparisons** as first-class benchmark/validation targets (public + reproducible), not only NONMEM/Monolix.
2) **Trust artifact distribution**: DOI snapshots (Zenodo) + `CITATION.cff` for benchmark reports, so results can be cited in papers.
3) **“Spec vs Speed” product narrative**: you already have this internally (parity mode vs fast mode). Make it a user-facing contract with a one-page diagram + policies (seeds, determinism).
4) **Server readiness**: fix the “Metal GPU” story in `nextstat-server` (either implement single-fit on Metal or hard reject at startup). Mismatch kills trust instantly.
5) **Input robustness** (ROOT especially): if you accept untrusted data, eliminate panic paths or contain them; this is an adoption blocker for server/agent modes.

## 5) Where this is tracked

- Audit report with the current technical trust blockers: `audit/2026-02-08_18-23_full.md`
- BMCP: epics and tasks created in project `nextstat.io` (see epics list in BMCP UI).

