"""Benchmark extraction pipelines: cold start, warm, accuracy.

Usage:
    python benchmarks/bench_extraction.py [--backend heuristic|gliner2|onnx|mlx]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from nextstat_nlp.priors import extract_prior_candidates
from nextstat_nlp.regimens import extract_regimens
from nextstat_nlp.survival import extract_survival_records
from nextstat_nlp.backends import get_backend

GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden"
_BACKEND_OBJ = None


def _read(name: str) -> list[str]:
    return [l.strip() for l in (GOLDEN_DIR / name).read_text().splitlines() if l.strip()]


def bench_pipeline(
    name: str,
    func,
    texts: list[str],
    backend: str,
    *,
    model_id: str | None,
    device: str | None,
    providers: list[str] | None,
    num_threads: int | None,
    n_warmup: int = 1,
    n_iter: int = 5,
):
    # Warmup
    for _ in range(n_warmup):
        func(
            texts,
            backend=backend,
            model_id=model_id,
            device=device,
            providers=providers,
            num_threads=num_threads,
            backend_obj=_BACKEND_OBJ,
        )

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        func(
            texts,
            backend=backend,
            model_id=model_id,
            device=device,
            providers=providers,
            num_threads=num_threads,
            backend_obj=_BACKEND_OBJ,
        )
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    print(f"  {name:30s}  median={median*1000:.2f}ms  min={min(times)*1000:.2f}ms  max={max(times)*1000:.2f}ms  (n={n_iter})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="heuristic", choices=["heuristic", "gliner2", "onnx", "mlx"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None, help="Torch device for backend=gliner2 (e.g. cpu, mps).")
    parser.add_argument("--providers", default=None, help="Comma-separated ORT providers for backend=onnx.")
    parser.add_argument("--num-threads", type=int, default=None, help="ORT intra/inter-op threads for backend=onnx.")
    parser.add_argument("--n-iter", type=int, default=10)
    args = parser.parse_args()

    print(f"Backend: {args.backend}")
    if args.model_id:
        print(f"Model: {args.model_id}")
    print(f"Iterations: {args.n_iter}")
    print()

    providers = [p.strip() for p in args.providers.split(",")] if args.providers else None

    global _BACKEND_OBJ
    _BACKEND_OBJ = get_backend(
        args.backend,
        model_id=args.model_id,
        device=args.device,
        providers=providers,
        num_threads=args.num_threads,
    )

    surv_texts = _read("survival_oncology.txt")
    prior_texts = _read("priors_pk_abstract.txt")
    regimen_texts = _read("regimen_protocol.txt")

    bench_pipeline(
        "survival (5 subjects)",
        extract_survival_records,
        surv_texts,
        args.backend,
        model_id=args.model_id,
        device=args.device,
        providers=providers,
        num_threads=args.num_threads,
        n_iter=args.n_iter,
    )
    bench_pipeline(
        "priors (PK abstract)",
        extract_prior_candidates,
        prior_texts,
        args.backend,
        model_id=args.model_id,
        device=args.device,
        providers=providers,
        num_threads=args.num_threads,
        n_iter=args.n_iter,
    )
    bench_pipeline(
        "regimens (3 records)",
        extract_regimens,
        regimen_texts,
        args.backend,
        model_id=args.model_id,
        device=args.device,
        providers=providers,
        num_threads=args.num_threads,
        n_iter=args.n_iter,
    )


if __name__ == "__main__":
    main()
