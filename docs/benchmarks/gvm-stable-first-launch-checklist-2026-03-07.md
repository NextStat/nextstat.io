# GVM Stable-First Launch Checklist

**Date**: 2026-03-07  
**Status**: launch checklist  
**Scope**: stable-first scalar GVM release / announcement execution

## Purpose

This checklist begins after the release PR is ready or merged.

It answers a narrower operational question:

- what must be true before maintainers announce that the stable-first GVM
  subset is ready for real users

## Launch checklist

### Repo / CI

- [ ] release PR is merged
- [ ] dedicated stable-first gate workflow is green:
  - `.github/workflows/gvm-stable-first.yml`
- [ ] release workflow is wired to require the stable-first gate:
  - `.github/workflows/release.yml`
- [ ] no unresolved `P0` issue remains in the external validation tracker

### User-facing docs

- [ ] top-level README points to the stable-first quickstart
- [ ] docs index points to:
  - quickstart
  - tutorial
  - support matrix
  - release notes
  - release candidate memo
- [ ] quickstart points to the committed example bundle
- [ ] example bundle README points to the maintainer validation pack

### Golden path

- [ ] `make gvm-stable-first-example` works on maintainer machine
- [ ] Route D adoption playbook smoke exists and is current
- [ ] stable-first tabular ingress is documented
- [ ] stable-first manifest ingress is documented

### External rollout

- [ ] first external validator wave is prepared
- [ ] invite template is ready to send
- [ ] report template is ready to collect feedback
- [ ] tracker template is ready to record outcomes

### Messaging

- [ ] stable-first subset is described as `stable`
- [ ] advanced scenario/campaign/parity/reporting layers are still described as `research-grade`
- [ ] release version is described consistently as `v0.10.0`
- [ ] no blanket claim is made about the full GVM stack being stable
- [ ] operating envelope is stated explicitly

## Launch message checklist

Any announcement should include:

- what is now stable
- what is still research-grade
- the shortest runnable path:
  - `make gvm-stable-first-example`
- where the evidence lives:
  - benchmark snapshot
  - robustness snapshot
  - support matrix

## Post-launch follow-through

Within the first validation wave:

- [ ] collect at least two external validator reports
- [ ] classify blockers vs docs friction
- [ ] feed actionable issues back into the tracker
- [ ] update quickstart/docs if repeated confusion appears

## Exit condition

The launch is operationally complete when:

- the release PR is merged
- the stable-first gate is green
- user-facing docs are aligned
- the external validation wave is actively tracked
