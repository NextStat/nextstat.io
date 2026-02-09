"""Tests for preprocessing cache (content-hash keyed)."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from nextstat.analysis.preprocess.cache import CachedPreprocessPipeline, cache_key
from nextstat.analysis.preprocess.config import DEFAULT_STEPS


def _toy_workspace() -> dict:
    return {
        "version": "1.0.0",
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "bkg",
                        "data": [100.0, 200.0, 300.0],
                        "modifiers": [
                            {
                                "name": "sys1",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [110.0, 220.0, 330.0],
                                    "lo_data": [90.0, 180.0, 270.0],
                                },
                            }
                        ],
                    }
                ],
            }
        ],
        "observations": [{"name": "SR", "data": [100.0, 200.0, 300.0]}],
        "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    }


class TestCacheKey:
    def test_same_input_same_key(self) -> None:
        ws = _toy_workspace()
        steps = [{"kind": "smooth_histosys"}]
        k1 = cache_key(ws, steps)
        k2 = cache_key(ws, steps)
        assert k1 == k2

    def test_different_workspace_different_key(self) -> None:
        ws1 = _toy_workspace()
        ws2 = _toy_workspace()
        ws2["channels"][0]["samples"][0]["data"][0] = 999.0
        steps = [{"kind": "smooth_histosys"}]
        assert cache_key(ws1, steps) != cache_key(ws2, steps)

    def test_different_steps_different_key(self) -> None:
        ws = _toy_workspace()
        s1 = [{"kind": "smooth_histosys"}]
        s2 = [{"kind": "prune_systematics"}]
        assert cache_key(ws, s1) != cache_key(ws, s2)

    def test_key_is_hex_sha256(self) -> None:
        k = cache_key(_toy_workspace(), [])
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


class TestCachedPreprocessPipeline:
    def test_first_run_computes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = CachedPreprocessPipeline(td)
            ws = _toy_workspace()
            result = pipe.run(ws)
            assert result.workspace is not None
            assert len(result.provenance.steps) > 0

    def test_second_run_hits_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = CachedPreprocessPipeline(td)
            ws = _toy_workspace()
            r1 = pipe.run(copy.deepcopy(ws))
            r2 = pipe.run(copy.deepcopy(ws))
            # Both should produce identical results
            assert r1.workspace == r2.workspace
            assert r1.provenance.output_sha256 == r2.provenance.output_sha256

    def test_cache_miss_on_changed_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = CachedPreprocessPipeline(td)
            ws1 = _toy_workspace()
            r1 = pipe.run(ws1)

            ws2 = _toy_workspace()
            ws2["channels"][0]["samples"][0]["data"][0] = 999.0
            r2 = pipe.run(ws2)
            # Results should differ
            assert r1.provenance.input_sha256 != r2.provenance.input_sha256

    def test_custom_steps_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            steps = [{"kind": "smooth_histosys", "params": {"method": "gaussian"}}]
            pipe = CachedPreprocessPipeline(td, steps_config=steps)
            ws = _toy_workspace()
            result = pipe.run(ws)
            assert len(result.provenance.steps) == 1

    def test_clear_removes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = CachedPreprocessPipeline(td)
            ws = _toy_workspace()
            pipe.run(ws)
            # Cache should have files
            count = pipe.clear()
            assert count >= 1
            # After clear, should recompute
            r2 = pipe.run(ws)
            assert r2.workspace is not None

    def test_cache_files_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = CachedPreprocessPipeline(td)
            ws = _toy_workspace()
            pipe.run(ws)
            # Check cache file exists
            json_files = list(Path(td).rglob("*.json"))
            assert len(json_files) == 1
            # Verify it's valid JSON
            data = json.loads(json_files[0].read_text())
            assert "workspace" in data
            assert "provenance" in data

    def test_corrupt_cache_treated_as_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = CachedPreprocessPipeline(td)
            ws = _toy_workspace()
            pipe.run(ws)
            # Corrupt the cache file
            json_files = list(Path(td).rglob("*.json"))
            assert len(json_files) == 1
            json_files[0].write_text("NOT VALID JSON {{{")
            # Should recompute without error
            result = pipe.run(ws)
            assert result.workspace is not None
