import os
import pytest

from nextstat_nlp._errors import MissingBackendDependency
from nextstat_nlp.backends import get_backend


def test_mlx_backend_missing_cli(monkeypatch):
    monkeypatch.delenv("NEXTSTAT_GLINER2_MLX_CLI", raising=False)
    with pytest.raises(MissingBackendDependency):
        get_backend("mlx")


def test_mlx_backend_bad_cli_path(monkeypatch, tmp_path):
    bad = tmp_path / "no_such_binary"
    monkeypatch.setenv("NEXTSTAT_GLINER2_MLX_CLI", str(bad))
    with pytest.raises(MissingBackendDependency):
        get_backend("mlx")
