from __future__ import annotations

import statistics
import sys
import time
from typing import Callable, List, Tuple


def _timeit(fn: Callable[[], None], *, reps: int = 7) -> List[float]:
    # Warm-up (imports, JIT-like effects, allocator).
    fn()
    times: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _summarize(name: str, times: List[float]) -> str:
    med = statistics.median(times)
    mn = min(times)
    mx = max(times)
    return f"{name}: median={med:.6f}s min={mn:.6f}s max={mx:.6f}s reps={len(times)}"


def bench_fit_batch_gaussian_mean() -> str:
    import nextstat

    models = []
    for k in range(20):
        data = [float(i) + 0.1 * k for i in range(20)]
        models.append(nextstat.GaussianMeanModel(data, 1.0))

    def _run():
        nextstat.fit_batch(models)

    times = _timeit(_run, reps=7)
    return _summarize("fit_batch(GaussianMeanModel) n_models=20 n=20", times)


def bench_kalman_forecast_intervals() -> str:
    import nextstat

    model = nextstat.timeseries.local_level_model(q=0.1, r=0.2, m0=0.0, p0=1.0)
    ys = [[(t * 0.01) for _ in range(1)] for t in range(1_000)]

    def _run():
        nextstat.timeseries.kalman_forecast(model, ys, steps=100, alpha=0.05)

    times = _timeit(_run, reps=9)
    return _summarize("kalman_forecast(alpha=0.05) n=1000 steps=100", times)


def main() -> int:
    import nextstat

    print(f"python={sys.version.split()[0]} nextstat={getattr(nextstat, '__version__', 'unknown')}")
    results: List[Tuple[str, str]] = []

    results.append(("fit_batch", bench_fit_batch_gaussian_mean()))
    results.append(("kalman_forecast", bench_kalman_forecast_intervals()))

    for _, line in results:
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

