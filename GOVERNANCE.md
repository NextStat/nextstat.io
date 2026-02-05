# NextStat Governance

This document describes how the NextStat project is governed: roles, responsibilities, and decision-making processes.

## Principles

1. Openness: decisions are made in public via issues and pull requests
2. Merit: influence is based on quality and consistency of contributions
3. Consensus: we aim for agreement, with escalation paths when needed
4. Inclusivity: contributors of all experience levels are welcome

## Roles

### Contributor

Anyone who has contributed at least one pull request (merged or not).

Rights:

- Open issues
- Submit pull requests
- Comment and participate in discussions

Responsibilities:

- Follow the Code of Conduct
- Sign commits with DCO (see `DCO.md`)
- Follow `CONTRIBUTING.md`

### Committer

A contributor who has demonstrated consistent, high-quality contributions.

Criteria (guidelines):

- At least 10 merged PRs
- Active within the last 3 months
- Working knowledge of the architecture and codebase
- Endorsed by at least 2 maintainers

Rights:

- All contributor rights
- Review and approve PRs
- Help triage issues (labels, assignment, milestones)
- Participate in technical votes

Responsibilities:

- Provide timely reviews (aim for first feedback within 48 hours)
- Help with releases
- Support new contributors

Appointment process:

1. A maintainer opens an issue proposing a candidate
2. Maintainers discuss for up to 72 hours
3. At least 2 votes in favor and no votes against

### Maintainer

Core project members with merge rights and release responsibility.

Criteria (guidelines):

- Committer for at least 6 months
- Deep understanding of the codebase and roadmap
- Demonstrated ownership and leadership
- Approved by a majority of existing maintainers

Rights:

- All committer rights
- Merge pull requests
- Manage releases
- Manage roles (appoint committers)
- Final decision in technical disputes (subject to escalation rules below)

Responsibilities:

- Ensure overall code quality
- Maintain long-term direction
- Mentor committers and contributors
- Participate in planning

Current maintainers:

- @andresvlc (Project Lead)

Appointment process:

1. A maintainer proposes a candidate
2. Maintainers discuss for 1 week
3. Vote passes with 2/3 majority

### Project Lead

Project founder / final escalation point for hard conflicts.

Rights:

- All maintainer rights
- Final resolution in conflicts
- Can change governance process (through RFC process)

Responsibilities:

- Long-term vision and strategy
- Resolve blocking conflicts
- Represent the project externally

Current Project Lead: @andresvlc

## Decision-Making

### Day-to-day changes

Examples: bug fixes, documentation updates, small improvements.

Process:

1. A contributor opens a PR
2. At least 1 committer or maintainer reviews
3. After approval, a maintainer merges

Requirements:

- Tests pass
- DCO sign-off on all commits
- Coding standards followed

### Significant changes

Examples: new features, module refactors, API changes.

Process:

1. Open an issue to discuss before implementation
2. Discussion for at least 48 hours
3. If consensus forms, implement and open a PR
4. At least 2 committer/maintainer reviews
5. Merge after approval

### Critical changes

Examples: architectural changes, breaking changes, licensing changes.

Process:

1. Create an RFC in `docs/rfcs/` (folder may be added as the RFC process matures)
2. Public discussion for at least 2 weeks
3. Maintainer vote passes with 2/3 majority
4. Implementation only after approval

RFC format (template):

```markdown
# RFC-0001: Title

## Summary
Brief description.

## Motivation
Why this is needed.

## Detailed Design
Technical details.

## Drawbacks
Known tradeoffs.

## Alternatives
Other options considered.

## Unresolved Questions
Open questions.
```

## Conflict Resolution

1. Discuss in the issue/PR and try to reach consensus
2. Escalate to maintainers if needed
3. Maintainer vote (simple majority)
4. Project Lead resolves blocking conflicts if still unresolved

## Review Process

For reviewers, verify:

- Tests cover changes and pass in CI
- Code follows rustfmt/clippy and style guides
- DCO sign-off on all commits
- Docs updated when behavior changes
- No breaking changes without an RFC (when applicable)
- Performance is not regressed without justification
- Security risks are not introduced

Timeline targets:

- First feedback within 48 hours
- Full review within 1 week for non-critical PRs

## Release Process

Versioning: Semantic Versioning 2.0.0

- MAJOR (X.0.0): breaking changes
- MINOR (0.X.0): new backward-compatible features
- PATCH (0.0.X): bug fixes

Release cadence (guidelines):

- Patch releases as needed
- Minor releases every 6-8 weeks
- Major releases when required

Procedure (high-level):

1. Open an issue "Release vX.Y.Z"
2. Prepare:
   - Update `CHANGELOG.md`
   - Update versions in Cargo/Python packaging
   - Run full test suite
   - Build docs
3. Create release branch `release/vX.Y.Z`
4. Tag `vX.Y.Z`
5. Publish artifacts (crates.io / PyPI)
6. Create GitHub Release notes

## Changes to Governance

This document may be changed via an RFC-like process:

1. Open a proposal issue (or RFC if the RFC folder exists)
2. Public discussion for at least 4 weeks
3. Maintainer vote passes with 3/4 majority
4. Project Lead has veto power for governance changes

## Code of Conduct

We want a friendly, productive community:

- Be respectful
- Be constructive in criticism
- Be patient with newcomers
- No harassment, discrimination, or trolling

Violations may result in:

1. Warning
2. Temporary ban (1-4 weeks)
3. Permanent ban

Report issues to: conduct@nextstat.io

## Contacts

- General: dev@nextstat.io
- Governance: governance@nextstat.io
- Code of Conduct: conduct@nextstat.io

