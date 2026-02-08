.PHONY: venv install hep validate

PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

venv:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r env/python/requirements.txt

hep: install
	$(PYBIN) suites/hep/run.py --deterministic --out out/hep_simple_nll.json

validate: install
	$(PYBIN) - <<'PY'
	import json
	from pathlib import Path
	import jsonschema
	schema = json.loads(Path("manifests/schema/benchmark_result_v1.schema.json").read_text())
	inst = json.loads(Path("out/hep_simple_nll.json").read_text())
	jsonschema.validate(inst, schema)
	print("benchmark_result_v1: schema ok")
	PY

