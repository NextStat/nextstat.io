# TREx/ROOT Expression Corpus (NTUP)

This folder contains an **offline** corpus of TREx/TtHFitter-style expressions used for compatibility testing.

- We **do not** depend on network during tests.
- Files are small JSON lists of expressions extracted from public repos.

Current sources:
- FCCFitter (public GitHub): selected `ReadFrom: NTUP` configs under `config/`.
- TRExFitter-style public examples:
  - `drkevinbarends/TRexFitter` (`configs/*.trf`)
  - `kyungeonchoi/ServiceXforTRExFitter` (`config/example.config`)
  - `alexander-held/TRExFitter-config-translation` (`input/minimal_example.config`)
