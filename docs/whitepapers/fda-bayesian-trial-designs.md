# FDA Bayesian Trial Designs for Drugs and Biologics

This page defines the product and documentation plan for a regulatory-first
Bayesian trial design surface in NextStat. It is intentionally grounded in the
public FDA and ICH position as of March 2026, not in generic "Bayesian
platform" aspirations.

The commercial thesis is simple: regulatory guidance creates instant
credibility, but only if the implementation is narrow, inspectable, and
defensible. For this topic, "faster MCMC" is not enough. We need a design
specification, operating-characteristics engine, prior sensitivity workflow,
and audit-ready outputs that line up with the way sponsors actually discuss a
Bayesian design with FDA.

## Why This Is Worth Doing Now

- The FDA published the draft guidance **Use of Bayesian Methodology in Clinical Trials of Drugs and Biological Products** on `2026-01-12`.
- CDER published **Bayesian Statistics in CDER and CBER: Demonstration Project** on `2026-01-13` to encourage more real-world sponsor engagement on simple Bayesian designs.
- FDA's final guidance **Adaptive Designs for Clinical Trials of Drugs and Biologics** remains the core non-Bayesian adaptive baseline for drugs and biologics.
- Complex designs still route through the **CID Meeting Program**, so any
  product claim must distinguish "simple, inspectable Bayesian designs" from
  "complex innovative design platform".
- `ICH E20` reached Step 2a/b in **June 2025** and the official ICH work plan
  says Step 4 is **expected by October 2026**. As of March 2026 it is still not
  final.

This is the right moment to ship a narrow wedge:

- simple enough to be legible to FDA review teams,
- useful enough for protocol planning and design simulation,
- aligned with assets NextStat already has: NUTS, validation packs, trial
  simulation, and provenance capture.

## Regulatory Baseline (March 2026)

The implementation should assume the following review posture.

### 1. FDA expects pre-specification, not post-hoc Bayesian flexibility

The `2026-01-12` draft guidance repeatedly pushes sponsors to pre-specify:

- the prior distribution,
- the decision criteria,
- the full analysis procedure,
- simulation-based operating characteristics under clinically relevant
  scenarios,
- and sensitivity analyses when prior assumptions are uncertain.

Implication for product scope: the core artifact is not "fit a model". The core
artifact is a versioned design package that binds priors, looks, stopping
rules, estimands, and simulation assumptions together.

### 2. FDA is explicitly trying to make simple Bayesian designs easier to discuss

The `2026-01-13` BSA demonstration project is a signal that FDA wants more
sponsor interaction on straightforward designs, especially where Bayesian
borrowing or posterior probability rules can reduce sample size or speed up
learning without turning the design into a black box.

Implication for product scope: v0 should target simple designs that fit this
program, not platform trials, response-adaptive randomization, or fully custom
MCMC-only workflows.

### 3. Adaptive design credibility still depends on frequentist-style operating characteristics

For drugs and biologics, Bayesian designs are still expected to show
frequentist operating characteristics:

- type I error,
- power,
- bias,
- false-go / false-stop behavior,
- stopping probabilities by look,
- and robustness under prior-data conflict.

Implication for product scope: the simulator is first-class. Posterior
computation alone is insufficient.

### 4. Complex designs still need an FDA interaction path

For anything beyond a simple, highly transparent design, the correct external
story is still:

- use the product to define and stress-test the design,
- package the assumptions cleanly,
- then engage FDA via the **CID Meeting Program** when warranted.

Implication for messaging: we should never position v0 as "FDA-ready Bayesian
trial platform". We should position it as "regulatory-aligned design and
simulation tooling for Bayesian trial planning".

## Product Thesis

The most defensible entry point is:

> "A narrow Bayesian design workbench for drugs and biologics that produces a
> design spec, operating-characteristics report, prior sensitivity report, and
> provenance bundle."

Not this:

> "A universal Bayesian clinical trials engine."

The first claim is credible in 2026. The second is not.

## What NextStat Can Reuse Today

We already have several pieces that materially reduce implementation risk.

### Posterior engine

- `nextstat.bayes.sample()` and the Rust NUTS / MAMS stack already provide the
  general posterior engine for non-conjugate extensions.
- Diagnostics infrastructure already exists for divergence rate, ESS, R-hat,
  and related sampler quality gates.

### Design-adjacent statistical primitives

- `crates/ns-inference/src/sequential.rs` already implements group sequential
  boundaries and alpha-spending.
- `crates/ns-inference/src/trial_simulation.rs` already provides Monte Carlo
  clinical trial simulation for PK workflows.

### Regulated-workflow primitives

- `RunBundle` provenance capture already exists in `ns-inference::artifacts`.
- 21 CFR Part 11 mapping already exists in
  `docs/validation/21cfr-part11-compliance.md`.
- IQ/OQ/PQ and validation-pack narratives already exist in
  `docs/validation/iq-oq-pq-protocol.md` and
  `docs/references/validation-report.md`.

### Existing pharma distribution surface

- `docs/personas/biologists.md`
- `docs/tutorials/pharma-pk.md`
- `docs/tutorials/pharma-survival.md`
- `docs/references/python-api.md`

This means we are not starting from zero. We are composing an already credible
regulated-computing story around a new design surface.

## Stable Surface Status (March 2026)

The stable product surface described by this whitepaper is now shipped in-repo.

Accepted stable subset:

- versioned exact design-spec, named-look analysis, operating-characteristics,
  posterior-predictive, and prior-sensitivity artifacts for
  `beta_binomial` and `normal_normal`
- consolidated family-specific `*_design_report_v0` JSON plus deterministic
  Markdown rendering
- deterministic report-bundle packaging with published runtime budget and
  `nextstat-bench` benchmark gate for packaging-affecting changes
- frozen `bayesian_design_regulatory_appendix_v0` JSON plus deterministic
  Markdown and PDF render paths
- validation-pack integration for frozen Bayesian design reports and appendix
  artifacts
- published prior-conflict diagnostic artifacts derived from frozen design
  reports
- published historical-control borrowing policy/review artifacts plus
  deterministic seeded operating-characteristics artifacts
- published robust-mixture prior policy/review artifacts plus deterministic
  seeded operating-characteristics artifacts
- family-specific Python access via `nextstat.bayes_design`
- CLI schema export, acceptance docs, release checklists, and BMCP-backed
  governance for the accepted stable subset

The remaining work is no longer about whether an FDA-aligned Bayesian design
surface exists. It is about explicitly separate expansion tracks beyond the
current stable claim.

## Explicitly Deferred Beyond the Current Stable Claim

The current stable claim is intentionally narrow. The following are explicit
future-expansion items, not hidden debt inside the shipped surface:

- CID / BSA-specific appendix narrative templates for regulated review workflows
- actual robust-mixture posterior engines
- actual historical borrowing posterior engines
- non-conjugate fallback engines and their separate trust labeling
- platform / multi-arm Bayesian design workflows
- survival-design support beyond the current conjugate binary and continuous
  families
- automatic SAP / protocol-language generation beyond the deterministic
  appendix blocks

## Stable Claim Boundary

The shipped stable subset remains deliberately narrow.

### In scope

1. Two endpoint families:
   - beta-binomial for binary response / toxicity
   - normal-normal for continuous endpoint with known or estimated variance
2. Trial structures:
   - exact two-arm parallel-group core designs
   - additive historical-borrowing review and seeded extension-OC artifacts
   - up to 3 planned looks
3. Decision rules:
   - posterior probability of superiority
   - posterior probability that treatment effect exceeds a clinically meaningful
     threshold
   - posterior futility threshold
4. Prior families:
   - flat / weakly informative
   - skeptical
   - additive robust-mixture policy/review and seeded extension-OC artifacts
5. Outputs:
   - machine-readable design spec
   - named-look analysis and seeded operating-characteristics artifacts
   - prior sensitivity and prior-conflict artifacts
   - deterministic design report, bundle, and regulatory appendix artifacts
   - validation-pack appendix outputs and RunBundle-linked provenance manifest

### Explicitly deferred beyond the stable claim

- response-adaptive randomization,
- platform / master-protocol designs,
- basket / umbrella borrowing frameworks,
- full survival-design support,
- missing-data modeling beyond pre-specified scenario hooks,
- automatic SAP generation beyond structured appendix blocks.

This keeps v0 aligned with the FDA demonstration-project tone: simple, clear,
reviewable.

## Proposed Architecture

### 1. Design schema

Extend the shipped `beta_binomial_design_v0` contract into a more general
schema-first design surface, for example:

```yaml
version: 0
trial:
  design_id: simple-binary-2arm-v0
  endpoint:
    family: beta_binomial
    estimand: posterior_response_rate_difference
  looks:
    - n_total: 60
    - n_total: 120
    - n_total: 180
  prior:
    family: robust_mixture_beta
    control:
      components:
        - weight: 0.7
          alpha: 18
          beta: 42
        - weight: 0.3
          alpha: 1
          beta: 1
    treatment:
      alpha: 1
      beta: 1
  decision_rules:
    success:
      posterior_prob_gt: 0.975
      effect_threshold: 0.0
    futility:
      posterior_prob_gt: 0.10
      effect_threshold: 0.0
  simulation:
    n_replicates: 50000
    scenarios:
      - name: null
        p_control: 0.30
        p_treatment: 0.30
      - name: target
        p_control: 0.30
        p_treatment: 0.45
      - name: prior_data_conflict
        p_control: 0.15
        p_treatment: 0.30
```

This schema should be versioned, JSON-schema-backed, and hashable so that the
exact design can be captured in a RunBundle.

### 2. Two execution paths

#### Path A: exact / conjugate engine

Use closed-form posteriors where available:

- beta-binomial,
- normal-normal.

This path should be the default for v0 because it is fast, transparent, and
easy to validate analytically.

#### Path B: posterior engine fallback

Use NUTS / MAMS only when the design leaves the conjugate family.

This path is valuable, but it should not define the first release. For v0 it is
an escape hatch, not the main story.

### 3. Operating-characteristics engine

Add a simulation runner that takes a design spec and emits:

- success rate by scenario,
- futility rate by scenario,
- look-specific stopping probabilities,
- expected sample size,
- posterior bias / interval coverage where relevant,
- prior sensitivity deltas.

### 4. Report layer

Emit:

- `design_report.json`
- `design_report.md`
- optional PDF appendix via existing validation/report patterns

The report should explicitly list:

- all priors,
- all looks,
- all decision criteria,
- all simulation seeds,
- and all software/provenance fields.

## Implementation Order

### Phase 0: schema and exact models

Status in March 2026:

1. shipped: `beta_binomial_design_v0` and `normal_normal_design_v0` schemas and validators
2. shipped: beta-binomial posterior utilities
3. shipped: normal-normal posterior utilities
4. shipped: deterministic simulation harness for the beta-binomial and normal-normal slices
5. shipped: deterministic `design_report.json` / `design_report.md` writers for the beta-binomial and normal-normal slices
6. shipped: deterministic Python RunBundle packaging helpers for frozen beta-binomial and normal-normal design reports
7. shipped: versioned `bayesian_design_report_bundle_v0` summary contract for deterministic report packaging
8. shipped: versioned `bayesian_design_regulatory_appendix_v0` JSON contract for frozen design-report appendix blocks
9. shipped: deterministic `validation-pack/render_validation_pack.sh --bayesian-design-report PATH` integration for frozen Bayesian appendix assembly
10. shipped: deterministic Markdown/PDF appendix rendering from frozen `bayesian_design_regulatory_appendix_v0` artifacts, including validation-pack emission of `.md` and optional `.pdf` appendix outputs
11. shipped: versioned `bayesian_prior_conflict_diagnostic_v0` contract for campaign-based prior-data conflict summaries derived from frozen design reports
12. shipped: versioned historical-control borrowing policy/review contracts for deterministic frozen-report policy gating without hidden borrowed-posterior execution
13. shipped: versioned robust-mixture prior policy/review contracts for deterministic frozen-report policy gating without hidden robust-mixture posterior execution

Release acceptance for the shipped report surface is now formalized in:

- `docs/specs/pharma/bayesian_design_report_acceptance_v0.md`
- `docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md`
- `docs/specs/pharma/bayesian_historical_control_borrowing_review_acceptance_v0.md`
- `docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_acceptance_v0.md`
- `docs/specs/pharma/bayesian_robust_mixture_prior_review_acceptance_v0.md`
- `docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md`
- `docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md`

### Phase 1: sensitivity and borrowing

Campaign-based prior-data conflict diagnostics are already shipped for frozen
design reports. Phase 1 broadens the prior policy surface beyond that baseline.

Then add:

1. actual robust-mixture posterior engines
2. historical-control borrowing posterior engines
3. comparison tables across prior families

### Phase 2: FDA interaction package

Then add:

1. CID / BSA-ready appendix render templates
2. report templates for protocol appendix language
3. example designs mirroring common sponsor use cases

## TDD Plan

This feature should be developed test-first because credibility is the product.

### Unit tests

Add deterministic tests for:

- schema validation and helpful errors,
- posterior updates for beta-binomial,
- posterior updates for normal-normal,
- posterior probability calculations,
- stopping-rule evaluation at each look,
- prior sensitivity grid generation.

### Golden tests

Add golden comparisons against closed-form calculations for:

- posterior mean,
- posterior interval bounds,
- posterior probability of superiority,
- predictive probability where applicable.

For conjugate designs there is no excuse for approximate disagreement.

### Simulation tests

Add seeded simulation tests that verify:

- null-scenario type I error stays within Monte Carlo error,
- target-scenario power exceeds configured minimum,
- expected sample size decreases when early stopping thresholds are more
  aggressive,
- prior-data conflict changes borrowing weight in the expected direction,
- results are bit-stable for fixed seeds.

### Integration tests

Add end-to-end tests that build a design spec, run the simulator, and assert:

- output schemas validate,
- markdown report contains all looks / thresholds / priors,
- RunBundle packaging contains the frozen report copy, deterministic markdown, manifest inventory, and simulation seed provenance,
- report references prior sensitivity and operating characteristics.

## Acceptance Gates

We should not promote this surface publicly until all of the following are
true.

1. Exact conjugate posteriors match analytical reference values to numerical
   precision.
2. Simulation outputs are reproducible for fixed seeds.
3. Every report contains:
   - design version
   - software version
   - design hash
   - seed inventory
   - prior table
   - operating-characteristics table
   - prior sensitivity appendix
4. Validation-pack integration can carry the design report without manual
   patching.
5. At least one example is explicitly framed for the FDA BSA demonstration
   project class of simple designs.

## Non-Goals

This work should not claim any of the following in v0:

- "FDA-cleared" or "FDA-approved" software,
- automatic regulatory acceptance,
- support for every Bayesian adaptive design family,
- replacement of sponsor statistical judgment,
- replacement of formal FDA meeting strategy.

The honest claim is narrower:

> NextStat can help specify, simulate, document, and audit simple Bayesian
> trial designs in a way that is materially closer to the current FDA
> discussion frame.

## Recommended Public Positioning

Use language like:

- "Bayesian design simulation for simple drug/biologic trials"
- "prior sensitivity and operating-characteristics reporting"
- "provenance-ready design packages for FDA discussion"

Avoid language like:

- "one-click FDA submission"
- "autonomous adaptive trial design"
- "platform-trial operating system"

## Sources

- FDA draft guidance: [Use of Bayesian Methodology in Clinical Trials of Drugs
  and Biological Products](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-bayesian-methodology-clinical-trials-drugs-and-biological-products)
- FDA program page: [Bayesian Statistics in CDER and CBER: Demonstration
  Project](https://www.fda.gov/drugs/development-resources/bayesian-statistics-cder-and-cber-demonstration-project)
- FDA final guidance: [Adaptive Designs for Clinical Trials of Drugs and
  Biologics](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/adaptive-designs-clinical-trials-drugs-and-biologics-guidance-industry)
- FDA interaction path: [CID Meeting Program](https://www.fda.gov/drugs/development-resources/complex-innovative-trial-design-meeting-program)
- ICH category page: [Efficacy Guidelines](https://admin.ich.org/page/efficacy-guidelines)
- ICH work plan: [ICH Association Work Plan 2026](https://admin.ich.org/sites/default/files/inline-files/ICHAssociation_WorkPlan_2026_Approved_2025_1118.pdf)
- ICH topic plan: [ICH E20 EWG Work Plan](https://database.ich.org/sites/default/files/ICH_E20_EWG_WorkPlan_2025_0721_FINAL.pdf)
