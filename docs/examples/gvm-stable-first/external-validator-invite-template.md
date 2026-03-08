# Stable-First GVM External Validator Invite Template

Use this as the first email / Slack / Mattermost message to an external
validator.

---

Subject: NextStat stable-first GVM validation request

Hi,

I’d like to ask you to validate a small, stable-first measurement-combination
workflow in NextStat.

The request is intentionally narrow:

1. run the committed example bundle
2. if that works, try one small scalar measurement-combination case from your
   own analysis

Start here:

- quickstart:
  `docs/quickstarts/hep-gvm-stable-first.md`
- committed example bundle:
  `docs/examples/gvm-stable-first/README.md`

For the first pass, please run:

```bash
make gvm-stable-first-example
```

If that succeeds, please try one small real case using the same stable-first
path:

- `combine-measurements-build-spec`
- `combine-measurements`
- `combine-measurements-calibrate`
- `combine-measurements-calibrate-study`

Please return your feedback using:

- `docs/examples/gvm-stable-first/external-validation-report-template.md`

What I care about most:

- whether the committed example runs without edits
- whether the table-to-spec workflow is clear
- whether `requested_solver` / `effective_solver` are understandable
- whether anything feels unexpectedly confusing or fragile

This request is only about the stable-first scalar GVM subset. You do not need
to validate the research-grade scenario/campaign/reporting layers.

Thanks.

---
