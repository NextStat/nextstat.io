## Description

<!-- Provide a brief description of the changes in this PR -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test improvement

## Related Issues

<!-- Link to related issues using #issue_number -->

Closes #
Related to #

## Changes Made

<!-- List the main changes in bullet points -->

-
-
-

## Testing

<!-- Describe the tests you ran and how to reproduce them -->

### Test Environment

- OS:
- Rust version:
- Python version (if applicable):

### Test Commands

```bash
# Commands to test this PR
cargo test --workspace
```

### Test Results

<!-- Paste relevant test output or screenshots -->

## Checklist

<!-- Mark completed items with an "x" -->

### Code Quality

- [ ] My code follows the project's style guidelines (ran `cargo fmt`)
- [ ] I have run `cargo clippy` and fixed all warnings
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings

### Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes (`cargo test --workspace`)
- [ ] I have tested on the target platforms (mark applicable):
  - [ ] Linux
  - [ ] macOS
  - [ ] Windows

### Documentation

- [ ] I have updated the documentation (if needed)
- [ ] I have added docstrings to new public functions/methods
- [ ] I have updated CHANGELOG.md (for significant changes)
- [ ] I have updated README.md (if needed)

### Git Hygiene

- [ ] **All commits are signed off with DCO** (`git commit -s`)
- [ ] My commits follow the [Conventional Commits](https://www.conventionalcommits.org/) format
- [ ] I have rebased my branch on the latest main
- [ ] My branch has a descriptive name (e.g., `feature/add-mle`, `bugfix/fix-gradient`)

### Compatibility

- [ ] My changes are backward compatible (or I have created an RFC for breaking changes)
- [ ] I have checked compatibility with pyhf (if applicable)
- [ ] I have verified numerical accuracy (if applicable)
- [ ] If I changed root Rust crates or `Cargo.lock`, I ran `make nsr-vendor-sync` and committed synced files under `bindings/ns-r/src`

### R Bindings Vendoring (if applicable)

- [ ] I ran the vendoring sync/check commands:

```bash
make nsr-vendor-sync
make nsr-vendor-check
```

- [ ] If sync changed files, I committed updates under `bindings/ns-r/src`

## Performance Impact

<!-- If this PR affects performance, describe the impact -->

- [ ] No performance impact
- [ ] Performance improvement (provide benchmarks)
- [ ] Potential performance regression (justified because...)

<details>
<summary>Benchmark Results (if applicable)</summary>

```
# Paste benchmark results here
```

</details>

## Screenshots (if applicable)

<!-- Add screenshots for UI changes or visualizations -->

## Additional Notes

<!-- Any additional information that reviewers should know -->

## Reviewer Guidelines

For reviewers, please check:

- [ ] Code quality and adherence to project standards
- [ ] Test coverage is adequate
- [ ] Documentation is clear and complete
- [ ] DCO sign-off is present on all commits
- [ ] No security vulnerabilities introduced
- [ ] Performance considerations addressed

---

**By submitting this pull request, I confirm that my contribution is made under the terms of the project's license (AGPL-3.0-or-later OR LicenseRef-Commercial).**
