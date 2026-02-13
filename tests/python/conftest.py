"""Pytest config for Python regression tests.

We keep helper modules (like `_tolerances.py`) alongside tests and ensure they
are importable regardless of how pytest is invoked.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

# Make `tests/python` importable as a top-level module path.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


@dataclass(frozen=True)
class _TimingRecord:
    nodeid: str
    label: str
    seconds: float


class _TimingRecorder:
    def __init__(self, *, enabled: bool, json_path: str | None):
        self.enabled = enabled
        self.json_path = json_path
        self.records: list[_TimingRecord] = []

    def add(self, nodeid: str, label: str, seconds: float) -> None:
        if not self.enabled:
            return
        self.records.append(_TimingRecord(nodeid=nodeid, label=str(label), seconds=float(seconds)))

    @contextmanager
    def time_block(self, *, nodeid: str, label: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(nodeid, label, time.perf_counter() - t0)

    def dump_json(self) -> None:
        if not (self.enabled and self.json_path):
            return
        out_path = Path(self.json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([asdict(r) for r in self.records], indent=2), encoding="utf-8")


class _TimingFacade:
    def __init__(self, *, recorder: _TimingRecorder, nodeid: str):
        self._recorder = recorder
        self._nodeid = nodeid

    def add(self, label: str, seconds: float) -> None:
        self._recorder.add(self._nodeid, label, seconds)

    @contextmanager
    def time(self, label: str) -> Iterator[None]:
        with self._recorder.time_block(nodeid=self._nodeid, label=label):
            yield

    def call(self, label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self.time(label):
            return fn(*args, **kwargs)


def strip_for_pyhf(workspace: dict) -> dict:
    """Return a deep-copy of *workspace* with NextStat-specific extension fields removed.

    pyhf ≥0.7 validates workspaces with ``additionalProperties: false``, so
    non-standard fields (e.g. ``constraint`` in
    ``measurements[].config.parameters[]``) cause ``InvalidSpecification``.

    This helper strips those fields so the same fixture can be consumed by both
    NextStat (which reads the extensions) and pyhf (which rejects them).
    """
    import copy

    ws = copy.deepcopy(workspace)
    # Known NextStat extensions that live inside the pyhf-validated subtree:
    _PARAM_EXTRA_KEYS = {"constraint"}
    for meas in ws.get("measurements", []):
        for param in meas.get("config", {}).get("parameters", []):
            for key in _PARAM_EXTRA_KEYS:
                param.pop(key, None)
    return ws


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ns-test-timings",
        action="store_true",
        default=False,
        help="Record and print NextStat-vs-reference timing breakdown for parity/validation tests.",
    )
    parser.addoption(
        "--ns-test-timings-json",
        action="store",
        default=None,
        help="Write recorded timing records to a JSON file path.",
    )


def pytest_configure(config: pytest.Config) -> None:
    enabled = bool(config.getoption("--ns-test-timings") or os.environ.get("NS_TEST_TIMINGS") == "1")
    json_path = config.getoption("--ns-test-timings-json") or os.environ.get("NS_TEST_TIMINGS_JSON")
    # Stash on config so fixtures/hooks can access.
    config._ns_timing_recorder = _TimingRecorder(enabled=enabled, json_path=json_path)  # type: ignore[attr-defined]


@pytest.fixture()
def ns_timing(request: pytest.FixtureRequest) -> _TimingFacade:
    recorder: _TimingRecorder = request.config._ns_timing_recorder  # type: ignore[attr-defined]
    return _TimingFacade(recorder=recorder, nodeid=request.node.nodeid)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    recorder: _TimingRecorder = config._ns_timing_recorder  # type: ignore[attr-defined]
    if not recorder.enabled or not recorder.records:
        return

    # nodeid -> label -> seconds
    by_test: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    totals: dict[str, float] = defaultdict(float)
    for r in recorder.records:
        by_test[r.nodeid][r.label] += r.seconds
        totals[r.label] += r.seconds

    terminalreporter.section("NextStat timing breakdown (NS_TEST_TIMINGS=1 / --ns-test-timings)")

    def _fmt(secs: float) -> str:
        if secs >= 60.0:
            return f"{secs/60.0:.2f}m"
        if secs >= 1.0:
            return f"{secs:.2f}s"
        return f"{secs*1000.0:.1f}ms"

    # Prefer showing only tests where we have both nextstat and some reference label.
    shown = 0
    for nodeid in sorted(by_test.keys()):
        labels = by_test[nodeid]
        ns_total = sum(
            v for k, v in labels.items() if k == "nextstat" or k.startswith("nextstat:")
        )
        if ns_total <= 0.0:
            continue
        ref_total = sum(
            v for k, v in labels.items() if not (k == "nextstat" or k.startswith("nextstat:"))
        )
        if ref_total <= 0.0:
            continue
        terminalreporter.write_line(
            f"{nodeid}: nextstat={_fmt(ns_total)} ref={_fmt(ref_total)} ({', '.join(sorted(k for k in labels.keys() if not (k == 'nextstat' or k.startswith('nextstat:'))))})"
        )
        shown += 1
        if shown >= 25:
            terminalreporter.write_line("... (truncated; use --ns-test-timings-json for full records)")
            break

    terminalreporter.write_line(
        "Totals: "
        + ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])))
    )

    recorder.dump_json()
