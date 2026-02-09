# Terminology and Style Guide (Public Docs)

This guide keeps public docs readable for non-HEP audiences while still allowing
HEP-specific concepts where needed.

## Rules of Thumb

- First use: expand acronyms in prose.
  - Example: `NLL (negative log-likelihood)`, `POI (parameter of interest)`.
- Prefer plain English in personas/quickstarts.
  - If a HEP term is necessary, add a one-line gloss.
- Avoid jargon in navigation pages.
  - Personas should link out to deeper references rather than teach HEP vocabulary.

## Terminology Lint

We run a lightweight terminology lint over:

- `docs/README.md`
- `docs/personas/*.md`
- `docs/quickstarts/**/*.md`

Lint script: `scripts/docs/terminology_lint.py`

If you intentionally want to bypass the lint for a file, add:

```md
<!-- terminology-lint: disable -->
```

## Recommended Expansions

- `NLL` -> `negative log-likelihood`
- `POI` -> `parameter of interest`
- `NP` / `NPs` -> `nuisance parameter(s)`
- `CLs` -> `modified frequentist confidence level`
- `Asimov dataset` -> `expected dataset` (or define it once)
- `HistFactory workspace` -> `binned-likelihood workspace format (pyhf-compatible)`

