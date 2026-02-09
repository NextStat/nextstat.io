"""Integration tests for preprocessing config builder and workspace helper."""

from __future__ import annotations

import copy
import json

import pytest

from nextstat.analysis.preprocess.config import (
    DEFAULT_STEPS,
    build_pipeline_from_config,
    preprocess_workspace,
)
from nextstat.analysis.preprocess.pipeline import PreprocessResult


def _toy_workspace() -> dict:
    """Workspace with one histosys and one normsys."""
    return {
        "version": "1.0.0",
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "bkg",
                        "data": [100.0, 200.0, 300.0, 400.0, 500.0],
                        "modifiers": [
                            {
                                "name": "shape_sys",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [110.0, 220.0, 330.0, 440.0, 550.0],
                                    "lo_data": [90.0, 180.0, 270.0, 360.0, 450.0],
                                },
                            },
                            {
                                "name": "small_sys",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [100.01, 200.02, 300.03, 400.04, 500.05],
                                    "lo_data": [99.99, 199.98, 299.97, 399.96, 499.95],
                                },
                            },
                            {
                                "name": "norm_sys",
                                "type": "normsys",
                                "data": {"hi": 1.05, "lo": 0.95},
                            },
                            {
                                "name": "tiny_norm",
                                "type": "normsys",
                                "data": {"hi": 1.001, "lo": 0.999},
                            },
                        ],
                    }
                ],
            }
        ],
        "observations": [{"name": "SR", "data": [100.0, 200.0, 300.0, 400.0, 500.0]}],
        "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    }


class TestBuildPipelineFromConfig:
    def test_default_steps(self) -> None:
        pipeline = build_pipeline_from_config()
        assert len(pipeline.steps) == 4

    def test_custom_steps(self) -> None:
        steps = [
            {"kind": "smooth_histosys", "params": {"method": "gaussian", "sigma": 2.0}},
        ]
        pipeline = build_pipeline_from_config(steps)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "smooth_histosys_v0"

    def test_empty_steps(self) -> None:
        pipeline = build_pipeline_from_config([])
        assert len(pipeline.steps) == 0

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown preprocessing step kind"):
            build_pipeline_from_config([{"kind": "bogus"}])

    def test_all_step_kinds(self) -> None:
        steps = [
            {"kind": "negative_bins_hygiene"},
            {"kind": "symmetrize_histosys"},
            {"kind": "smooth_histosys"},
            {"kind": "prune_systematics"},
        ]
        pipeline = build_pipeline_from_config(steps)
        assert len(pipeline.steps) == 4


class TestPreprocessWorkspace:
    def test_default_pipeline(self) -> None:
        """preprocess_workspace with no config runs the default pipeline."""
        ws = _toy_workspace()
        result = preprocess_workspace(ws)
        assert isinstance(result, PreprocessResult)
        assert len(result.provenance.steps) == 4
        # small_sys should be pruned by default (0.01% < 0.5%)
        mods = result.workspace["channels"][0]["samples"][0]["modifiers"]
        mod_names = {m["name"] for m in mods}
        assert "shape_sys" in mod_names
        assert "small_sys" not in mod_names

    def test_disabled_config(self) -> None:
        """config with enabled=False returns workspace unchanged."""
        ws = _toy_workspace()
        ws_copy = copy.deepcopy(ws)
        config = {"enabled": False}
        result = preprocess_workspace(ws, config=config)
        assert result.workspace == ws_copy
        assert len(result.provenance.steps) == 0

    def test_custom_config_steps(self) -> None:
        """Config with custom steps is respected."""
        ws = _toy_workspace()
        config = {
            "enabled": True,
            "steps": [
                {"kind": "prune_systematics", "params": {"shape_threshold": 0.005}},
            ],
        }
        result = preprocess_workspace(ws, config=config)
        assert len(result.provenance.steps) == 1

    def test_provenance_has_sha(self) -> None:
        ws = _toy_workspace()
        result = preprocess_workspace(ws)
        assert len(result.provenance.input_sha256) == 64
        assert len(result.provenance.output_sha256) == 64

    def test_in_place_mutates(self) -> None:
        ws = _toy_workspace()
        original_id = id(ws)
        result = preprocess_workspace(ws, in_place=True)
        # Should be same dict object
        assert id(result.workspace) == original_id

    def test_not_in_place_copies(self) -> None:
        ws = _toy_workspace()
        ws_before = copy.deepcopy(ws)
        result = preprocess_workspace(ws, in_place=False)
        # Input should be unchanged
        assert ws == ws_before

    def test_norm_prune_with_overall(self) -> None:
        """Test pruning normsys via overall method."""
        ws = _toy_workspace()
        config = {
            "enabled": True,
            "steps": [
                {"kind": "prune_systematics", "params": {
                    "prune_method": "norm",
                    "norm_threshold": 0.005,
                }},
            ],
        }
        result = preprocess_workspace(ws, config=config)
        mods = result.workspace["channels"][0]["samples"][0]["modifiers"]
        mod_names = {m["name"] for m in mods}
        # tiny_norm (0.1%) should be pruned, norm_sys (5%) kept
        assert "norm_sys" in mod_names
        assert "tiny_norm" not in mod_names

    def test_full_pipeline_provenance_chain(self) -> None:
        """Full pipeline SHA chain is consistent."""
        ws = _toy_workspace()
        result = preprocess_workspace(ws)
        steps = result.provenance.steps
        for i in range(len(steps) - 1):
            assert steps[i].output_sha256 == steps[i + 1].input_sha256
        assert result.provenance.input_sha256 == steps[0].input_sha256
        assert result.provenance.output_sha256 == steps[-1].output_sha256
