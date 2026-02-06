.PHONY: apex2-baseline-record apex2-baseline-compare

PY ?= ./.venv/bin/python
PYTHONPATH ?= bindings/ns-py/python

apex2-baseline-record:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/record_baseline.py

apex2-baseline-compare:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/compare_with_latest_baseline.py

