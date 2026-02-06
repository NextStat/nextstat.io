"""Minimal k-fold cross-validation helpers (Phase 6.4.2).

Design constraints:
- Dependency-free (no numpy required)
- Deterministic shuffling via `seed`
- Small surface area focused on NextStat GLM fits
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from . import linear, logistic, poisson, negbin
from . import metrics as _metrics


IndexSplit = Tuple[List[int], List[int]]


def kfold_indices(
    n: int,
    k: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
) -> List[IndexSplit]:
    if n <= 0:
        raise ValueError("n must be > 0")
    if k <= 1:
        raise ValueError("k must be >= 2")
    if k > n:
        raise ValueError("k must be <= n")

    idx = list(range(n))
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(idx)

    folds: List[List[int]] = [[] for _ in range(k)]
    for i, ix in enumerate(idx):
        folds[i % k].append(ix)

    splits: List[IndexSplit] = []
    for i in range(k):
        test_idx = sorted(folds[i])
        train_idx = sorted(ix for j, f in enumerate(folds) if j != i for ix in f)
        splits.append((train_idx, test_idx))
    return splits


def _take_rows(x: Sequence[Sequence[float]], idx: Sequence[int]) -> List[List[float]]:
    return [list(map(float, x[i])) for i in idx]


def _take_vals(y: Sequence[Any], idx: Sequence[int]) -> List[Any]:
    return [y[i] for i in idx]


@dataclass(frozen=True)
class CvResult:
    model: str
    metric: str
    scores: List[float]
    mean: float
    stdev: float
    k: int
    splits: List[IndexSplit]


ModelName = Literal["linear", "logistic", "poisson", "negbin"]
MetricName = Literal["rmse", "log_loss", "mean_poisson_deviance"]


def cross_val_score(
    model: ModelName,
    x: Sequence[Sequence[float]],
    y: Sequence[Any],
    *,
    k: int = 5,
    shuffle: bool = True,
    seed: int = 0,
    metric: Optional[MetricName] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> CvResult:
    if fit_kwargs is None:
        fit_kwargs = {}

    n = len(x)
    if n != len(y):
        raise ValueError("X and y must have the same length")
    splits = kfold_indices(n, k, shuffle=shuffle, seed=seed)

    if metric is None:
        if model == "linear":
            metric = "rmse"
        elif model == "logistic":
            metric = "log_loss"
        else:
            metric = "mean_poisson_deviance"

    scores: List[float] = []
    for train_idx, test_idx in splits:
        x_train = _take_rows(x, train_idx)
        y_train = _take_vals(y, train_idx)
        x_test = _take_rows(x, test_idx)
        y_test = _take_vals(y, test_idx)

        if model == "linear":
            fit = linear.fit(x_train, y_train, **fit_kwargs)
            y_pred = fit.predict(x_test)
            if metric != "rmse":
                raise ValueError("linear supports only metric='rmse'")
            score = _metrics.rmse(y_test, y_pred)
        elif model == "logistic":
            fit = logistic.fit(x_train, y_train, **fit_kwargs)
            p = fit.predict_proba(x_test)
            if metric != "log_loss":
                raise ValueError("logistic supports only metric='log_loss'")
            score = _metrics.log_loss(y_test, p)
        elif model == "poisson":
            fit = poisson.fit(x_train, y_train, **fit_kwargs)
            mu = fit.predict_mean(x_test)
            if metric != "mean_poisson_deviance":
                raise ValueError("poisson supports only metric='mean_poisson_deviance'")
            score = _metrics.mean_poisson_deviance(y_test, mu)
        elif model == "negbin":
            fit = negbin.fit(x_train, y_train, **fit_kwargs)
            mu = fit.predict_mean(x_test)
            if metric != "mean_poisson_deviance":
                raise ValueError("negbin supports only metric='mean_poisson_deviance'")
            score = _metrics.mean_poisson_deviance(y_test, mu)
        else:
            raise ValueError(f"unknown model '{model}'")

        scores.append(float(score))

    mean = statistics.fmean(scores) if scores else float("nan")
    stdev = statistics.pstdev(scores) if len(scores) >= 2 else 0.0
    return CvResult(
        model=model,
        metric=str(metric),
        scores=scores,
        mean=float(mean),
        stdev=float(stdev),
        k=k,
        splits=splits,
    )


__all__ = ["CvResult", "kfold_indices", "cross_val_score"]

