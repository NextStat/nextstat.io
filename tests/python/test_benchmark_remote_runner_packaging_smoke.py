from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = (
    REPO_ROOT
    / "benchmarks"
    / "nextstat-public-benchmarks"
    / "scripts"
    / "install_local_nextstat_python.sh"
)
REMOTE_RUNNERS = [
    REPO_ROOT / "scripts" / "benchmarks" / "bench_mams_suite_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "bench_mams_multiseed_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "bench_mams_stress_multiseed_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "publish_mams_snapshot_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "bench_bayesian_suite_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "bench_bayesian_multiseed_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "publish_bayesian_snapshot_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "bench_sampler_matrix_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "bench_ads_variance_reduction_matrix_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "bench_ads_timeseries_surface_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "apex2_phase0_remote.sh",
    REPO_ROOT / "scripts" / "benchmarks" / "apex2_phase1_remote.sh",
]


def test_remote_install_helper_builds_local_snapshot_without_pypi_runtime_deps() -> None:
    text = HELPER.read_text(encoding="utf-8")
    assert "maturin build" in text
    assert "--no-deps" in text
    assert "nextstat-*.whl" in text


def test_host_backed_remote_runners_use_shared_local_install_helper() -> None:
    helper_ref = "benchmarks/nextstat-public-benchmarks/scripts/install_local_nextstat_python.sh"
    for path in REMOTE_RUNNERS:
        text = path.read_text(encoding="utf-8")
        assert helper_ref in text, path
        assert "maturin develop --release --pip-path" not in text, path
