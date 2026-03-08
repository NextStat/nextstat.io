import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "bayesian" / "run.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("nextstat_bayesian_run", RUN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_local_cmdstan_install_picks_highest_version(tmp_path: Path) -> None:
    module = _load_module()
    repo_root = tmp_path / "seed_repo"
    vendor = repo_root / "vendor" / "cmdstan"
    (vendor / "cmdstan-2.37.0").mkdir(parents=True)
    (vendor / "cmdstan-2.38.1").mkdir(parents=True)
    (vendor / "cmdstan-2.40.0").mkdir(parents=True)

    found = module._discover_local_cmdstan_install(repo_root)
    assert found == (vendor / "cmdstan-2.40.0").resolve()


def test_discover_local_cmdstan_install_returns_none_when_missing(tmp_path: Path) -> None:
    module = _load_module()
    repo_root = tmp_path / "seed_repo"
    repo_root.mkdir()

    found = module._discover_local_cmdstan_install(repo_root)
    assert found is None


def test_resolve_cmdstan_home_prefers_vendor_over_ambient(tmp_path: Path) -> None:
    module = _load_module()
    repo_root = tmp_path / "seed_repo"
    vendor = repo_root / "vendor" / "cmdstan"
    chosen_vendor = vendor / "cmdstan-2.38.0"
    chosen_vendor.mkdir(parents=True)
    ambient = tmp_path / "ambient" / "cmdstan-2.40.0"
    ambient.mkdir(parents=True)

    found, source = module._resolve_cmdstan_home(repo_root, str(ambient))
    assert found == chosen_vendor.resolve()
    assert source == "vendor"


def test_resolve_cmdstan_home_falls_back_to_ambient(tmp_path: Path) -> None:
    module = _load_module()
    repo_root = tmp_path / "seed_repo"
    repo_root.mkdir()
    ambient = tmp_path / "ambient" / "cmdstan-2.40.0"
    ambient.mkdir(parents=True)

    found, source = module._resolve_cmdstan_home(repo_root, str(ambient))
    assert found == ambient.resolve()
    assert source == "ambient"


def test_choose_pymc_pytensor_flags_prefers_existing_env() -> None:
    module = _load_module()
    flags, source = module._choose_pymc_pytensor_flags(
        "mode=NUMBA,blas__ldflags=-lblas", has_openblas=True, has_blas=True
    )
    assert flags == "mode=NUMBA,blas__ldflags=-lblas"
    assert source == "env"


def test_choose_pymc_pytensor_flags_prefers_openblas_over_blas() -> None:
    module = _load_module()
    flags, source = module._choose_pymc_pytensor_flags(
        None, has_openblas=True, has_blas=True
    )
    assert flags == "blas__ldflags=-lopenblas"
    assert source == "auto_openblas"


def test_choose_pymc_pytensor_flags_falls_back_to_blas() -> None:
    module = _load_module()
    flags, source = module._choose_pymc_pytensor_flags(
        None, has_openblas=False, has_blas=True
    )
    assert flags == "blas__ldflags=-lblas"
    assert source == "auto_blas"
