.PHONY: apex2-baseline-record apex2-baseline-compare apex2-pre-release-gate

PY ?= ./.venv/bin/python
PYTHONPATH ?= bindings/ns-py/python
RECORD_ARGS ?=
COMPARE_ARGS ?=

apex2-baseline-record:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/record_baseline.py $(RECORD_ARGS)

apex2-baseline-compare:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/compare_with_latest_baseline.py $(COMPARE_ARGS)

apex2-pre-release-gate:
	bash scripts/apex2/pre_release_gate.sh
