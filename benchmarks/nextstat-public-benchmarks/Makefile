.PHONY: venv install hep hep-fit pharma pharma-fit validate snapshot

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

hep-fit: install
	$(PYBIN) suites/hep/suite.py --deterministic --fit --fit-repeat 3 --out-dir out/hep_fit

pharma: install
	$(PYBIN) suites/pharma/suite.py --deterministic --out-dir out/pharma

pharma-fit: install
	$(PYBIN) suites/pharma/suite.py --deterministic --fit --fit-repeat 3 --out-dir out/pharma_fit

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
		schema_pharma_case = json.loads(Path("manifests/schema/pharma_benchmark_result_v1.schema.json").read_text())
		schema_pharma_suite = json.loads(Path("manifests/schema/pharma_benchmark_suite_result_v1.schema.json").read_text())
		psuite = json.loads(Path("out/pharma/pharma_suite.json").read_text())
		jsonschema.validate(psuite, schema_pharma_suite)
		for e in psuite["cases"]:
		    inst = json.loads((Path("out/pharma") / e["path"]).read_text())
		    jsonschema.validate(inst, schema_pharma_case)
		print("pharma_benchmark_suite_result_v1: schema ok")
		PY

snapshot: install
	$(PYBIN) scripts/publish_snapshot.py --deterministic --fit --fit-repeat 3
