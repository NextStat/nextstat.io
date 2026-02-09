# TREx report golden fixtures (NextStat)

These fixtures are **numbers-first** goldens for `nextstat report` JSON artifacts. They are used to
guard against accidental numeric drift without relying on pixel diffs.

## What is included

- `histfactory_v0/` — goldens for `tests/fixtures/histfactory/workspace.json` + `combination.xml`
  using a deterministic **pseudo-fit** (constraint centers + diagonal covariance).

## How to (re)generate

From the repo root:

```sh
NS_RECORD_GOLDENS=1 cargo test -p ns-cli --test cli_report_golden -- --nocapture
```

Then re-run without `NS_RECORD_GOLDENS` to confirm:

```sh
cargo test -p ns-cli --test cli_report_golden
```

