.PHONY: \
	apex2-baseline-record \
	apex2-baseline-compare \
	apex2-pre-release-gate \
	apex2-root-prereq \
	apex2-root-baseline-record \
	apex2-root-cases \
	apex2-condor-render-root-array-sub \
	apex2-root-aggregate \
	apex2-root-suite-compare-perf \
	apex2-root-suite-compare-latest

PY ?= ./.venv/bin/python
PYTHONPATH ?= bindings/ns-py/python
RECORD_ARGS ?=
COMPARE_ARGS ?=
ROOT_SEARCH_DIR ?=
ROOT_CASES_OUT ?= tmp/apex2_root_cases.json
ROOT_CASES_ARGS ?=
ROOT_CASES_JSON ?=
ROOT_BASELINE_OUT_DIR ?= tmp/baselines
ROOT_BASELINE_ARGS ?=
ROOT_CONDOR_INITIALDIR ?= $(CURDIR)
ROOT_CONDOR_SUB_OUT ?= apex2_root_suite_array.sub
ROOT_RESULTS_DIR ?=
ROOT_AGG_GLOB ?= apex2_root_case_*.json
ROOT_AGG_OUT ?= tmp/apex2_root_suite_aggregate.json
ROOT_AGG_ARGS ?=
ROOT_BASELINE_SUITE ?=
ROOT_CURRENT_SUITE ?=
ROOT_PERF_OUT ?= tmp/root_suite_perf_compare.json
ROOT_PERF_ARGS ?=
ROOT_ROOT_MANIFEST ?= tmp/baselines/latest_root_manifest.json

apex2-baseline-record:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/record_baseline.py $(RECORD_ARGS)

apex2-baseline-compare:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/compare_with_latest_baseline.py $(COMPARE_ARGS)

apex2-pre-release-gate:
	bash scripts/apex2/pre_release_gate.sh

apex2-root-prereq:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/apex2_root_suite_report.py --prereq-only --out tmp/apex2_root_prereq.json

apex2-root-baseline-record:
	@test -n "$(ROOT_SEARCH_DIR)" || (echo "Set ROOT_SEARCH_DIR=/abs/path/to/trex/output" >&2; exit 2)
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/record_baseline.py \
		--only root \
		--out-dir "$(ROOT_BASELINE_OUT_DIR)" \
		--root-search-dir "$(ROOT_SEARCH_DIR)" \
		$(ROOT_BASELINE_ARGS)

apex2-root-cases:
	@test -n "$(ROOT_SEARCH_DIR)" || (echo "Set ROOT_SEARCH_DIR=/abs/path/to/trex/output" >&2; exit 2)
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/generate_apex2_root_cases.py \
		--search-dir "$(ROOT_SEARCH_DIR)" \
		--out "$(ROOT_CASES_OUT)" \
		$(ROOT_CASES_ARGS)

apex2-condor-render-root-array-sub:
	@test -n "$(ROOT_CASES_JSON)" || (echo "Set ROOT_CASES_JSON=/abs/path/to/apex2_root_cases.json" >&2; exit 2)
	"$(PY)" scripts/condor/render_apex2_root_suite_array_sub.py \
		--cases "$(ROOT_CASES_JSON)" \
		--initialdir "$(ROOT_CONDOR_INITIALDIR)" \
		--out "$(ROOT_CONDOR_SUB_OUT)"

apex2-root-aggregate:
	@test -n "$(ROOT_RESULTS_DIR)" || (echo "Set ROOT_RESULTS_DIR=/abs/path/to/results_dir" >&2; exit 2)
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/aggregate_apex2_root_suite_reports.py \
		--in-dir "$(ROOT_RESULTS_DIR)" \
		--glob "$(ROOT_AGG_GLOB)" \
		--out "$(ROOT_AGG_OUT)" \
		$(ROOT_AGG_ARGS)

apex2-root-suite-compare-perf:
	@test -n "$(ROOT_BASELINE_SUITE)" || (echo "Set ROOT_BASELINE_SUITE=/abs/path/to/root_suite_baseline.json" >&2; exit 2)
	@test -n "$(ROOT_CURRENT_SUITE)" || (echo "Set ROOT_CURRENT_SUITE=/abs/path/to/root_suite_current.json" >&2; exit 2)
	"$(PY)" tests/compare_apex2_root_suite_to_baseline.py \
		--baseline "$(ROOT_BASELINE_SUITE)" \
		--current "$(ROOT_CURRENT_SUITE)" \
		--out "$(ROOT_PERF_OUT)" \
		$(ROOT_PERF_ARGS)

apex2-root-suite-compare-latest:
	@test -n "$(ROOT_CURRENT_SUITE)" || (echo "Set ROOT_CURRENT_SUITE=/abs/path/to/apex2_root_suite_aggregate.json" >&2; exit 2)
	"$(PY)" tests/compare_apex2_root_suite_to_latest_baseline.py \
		--manifest "$(ROOT_ROOT_MANIFEST)" \
		--current "$(ROOT_CURRENT_SUITE)" \
		--out "$(ROOT_PERF_OUT)" \
		$(ROOT_PERF_ARGS)
