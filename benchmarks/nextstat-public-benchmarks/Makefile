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
	$(PYBIN) suites/hep/suite.py --deterministic --out-dir out/hep

validate: install
	$(PYBIN) - <<'PY'
	import json
	from pathlib import Path
	import jsonschema
	schema_case = json.loads(Path("manifests/schema/benchmark_result_v1.schema.json").read_text())
	schema_suite = json.loads(Path("manifests/schema/benchmark_suite_result_v1.schema.json").read_text())
	suite = json.loads(Path("out/hep/hep_suite.json").read_text())
	jsonschema.validate(suite, schema_suite)
	for e in suite["cases"]:
	    inst = json.loads((Path("out/hep") / e["path"]).read_text())
	    jsonschema.validate(inst, schema_case)
	print("benchmark_suite_result_v1: schema ok")
	PY
