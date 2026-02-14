"""Benchmark environment snapshot — collects all reproducibility-critical info.

Every benchmark script MUST call `collect_environment()` and store the result
in the JSON artifact under the `"environment"` key. This is non-negotiable
for reproducibility (see .claude/benchmark-protocol.md §6b).

Usage:
    from scripts.bench_env import collect_environment
    env = collect_environment()
    artifact["environment"] = env
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    """Run a command and return stripped stdout, or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _gpu_info() -> list[dict[str, str]] | None:
    """Query nvidia-smi for GPU info. Returns list of dicts or None."""
    if not shutil.which("nvidia-smi"):
        return None
    raw = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap,memory.total,power.limit,compute_mode",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return None
    gpus = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({
                "name": parts[0],
                "driver": parts[1],
                "compute_capability": parts[2],
                "memory_mb": parts[3],
                "power_limit_w": parts[4],
                "compute_mode": parts[5],
            })
    return gpus if gpus else None


def _cuda_version() -> str | None:
    """Get CUDA toolkit version from nvcc."""
    raw = _run(["nvcc", "--version"])
    if raw:
        for line in raw.splitlines():
            if "release" in line.lower():
                # e.g. "Cuda compilation tools, release 12.2, V12.2.140"
                return line.strip()
    return None


def _jax_info() -> dict[str, str | None]:
    """Collect JAX/JAXlib versions and backend info."""
    info: dict[str, str | None] = {
        "jax": None, "jaxlib": None, "backend": None, "device": None,
    }
    try:
        import jax  # type: ignore
        info["jax"] = str(jax.__version__)
        import jaxlib  # type: ignore
        info["jaxlib"] = str(jaxlib.__version__)
        devices = jax.devices()
        if devices:
            d = devices[0]
            info["backend"] = str(d.platform)
            info["device"] = str(d.device_kind)
    except Exception:
        pass

    # Compilation cache
    info["compilation_cache_dir"] = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    try:
        import jax  # type: ignore
        cc = getattr(jax.config, "jax_compilation_cache_dir", None)
        if cc:
            info["compilation_cache_dir"] = str(cc)
    except Exception:
        pass

    return info


def _competitor_versions() -> dict[str, str | None]:
    """Collect versions of known competitor packages."""
    packages = [
        "blackjax", "numpyro", "cmdstanpy", "pymc", "arviz",
        "statsmodels", "sklearn", "glum", "linearmodels",
        "lifelines", "pyhf", "scipy", "numpy",
    ]
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            pass  # not installed — don't include
    return versions


def collect_environment() -> dict[str, Any]:
    """Collect full environment snapshot for benchmark reproducibility.

    Returns a dict suitable for JSON serialization. Store this under
    `artifact["environment"]`.
    """
    env: dict[str, Any] = {
        # Python
        "python_version": sys.version.split()[0],
        "python_full": sys.version,

        # OS / hardware
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "node": platform.node(),

        # CPU
        "cpu": None,

        # Rust
        "rustc": _run(["rustc", "--version"]),
    }

    # CPU model (Linux: lscpu, macOS: sysctl)
    if platform.system() == "Linux":
        raw = _run(["lscpu"])
        if raw:
            for line in raw.splitlines():
                if "Model name" in line:
                    env["cpu"] = line.split(":", 1)[-1].strip()
                    break
    elif platform.system() == "Darwin":
        env["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])

    # GPU
    gpus = _gpu_info()
    if gpus is not None:
        env["gpus"] = gpus
        env["cuda_toolkit"] = _cuda_version()
        env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    # JAX
    jax_info = _jax_info()
    if jax_info.get("jax") is not None:
        env["jax"] = jax_info

    # XLA flags
    xla_flags = os.environ.get("XLA_FLAGS")
    xla_prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if xla_flags is not None or xla_prealloc is not None:
        env["xla"] = {
            "XLA_FLAGS": xla_flags,
            "XLA_PYTHON_CLIENT_PREALLOCATE": xla_prealloc,
        }

    # NextStat
    try:
        import nextstat  # type: ignore
        env["nextstat_version"] = str(nextstat.__version__)
    except ImportError:
        env["nextstat_version"] = None

    # Competitor packages
    competitors = _competitor_versions()
    if competitors:
        env["packages"] = competitors

    return env


def print_environment(env: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect (if needed) and print environment summary to stdout. Returns the env dict."""
    if env is None:
        env = collect_environment()

    print("=" * 70)
    print("  BENCHMARK ENVIRONMENT")
    print("=" * 70)
    print(f"  Python:   {env.get('python_version', '?')}")
    print(f"  Platform: {env.get('platform', '?')}")
    if env.get("cpu"):
        print(f"  CPU:      {env['cpu']}")
    if env.get("rustc"):
        print(f"  Rust:     {env['rustc']}")
    if env.get("nextstat_version"):
        print(f"  NextStat: {env['nextstat_version']}")

    gpus = env.get("gpus")
    if gpus:
        for i, g in enumerate(gpus):
            print(f"  GPU[{i}]:   {g['name']} ({g['memory_mb']}MB, driver {g['driver']}, CC {g['compute_capability']})")
        if env.get("cuda_toolkit"):
            print(f"  CUDA:     {env['cuda_toolkit']}")

    jax = env.get("jax")
    if jax:
        print(f"  JAX:      {jax.get('jax', '?')} / JAXlib {jax.get('jaxlib', '?')} / {jax.get('backend', '?')} {jax.get('device', '?')}")
        cache = jax.get("compilation_cache_dir")
        print(f"  JAX cache: {cache or 'disabled'}")

    pkgs = env.get("packages", {})
    if pkgs:
        pkg_str = ", ".join(f"{k}={v}" for k, v in sorted(pkgs.items()))
        print(f"  Packages: {pkg_str}")

    xla = env.get("xla")
    if xla:
        print(f"  XLA_FLAGS: {xla.get('XLA_FLAGS', 'unset')}")
        print(f"  XLA_PREALLOC: {xla.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'unset')}")

    print("=" * 70)
    return env


if __name__ == "__main__":
    import json as _json
    env = print_environment()
    print(_json.dumps(env, indent=2))
