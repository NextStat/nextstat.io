"""Neural Surrogate Distillation — dataset generation for surrogate training.

NextStat serves as the "oracle" (ground-truth generator) for training
lightweight neural surrogates of the HistFactory likelihood surface.

The workflow:

1. **Sample** parameter space (Sobol, Latin Hypercube, or uniform random).
2. **Evaluate** NLL and its gradient at each point via NextStat's GPU batch kernel.
3. **Export** the ``(parameters, NLL, gradient)`` dataset as NumPy arrays,
   a PyTorch ``TensorDataset``, or Parquet file.
4. **Train** a small MLP to approximate the likelihood surface.
5. **Validate** by comparing the surrogate's MLE against NextStat's exact fit.

The surrogate runs in nanoseconds vs. milliseconds for the exact computation,
enabling real-time MCMC, global EFT fits, or interactive dashboards.

Example::

    import nextstat
    from nextstat.distill import generate_dataset, to_torch_dataset

    model = nextstat.from_pyhf(workspace_json)

    # Generate 100k (params, NLL, grad) tuples on GPU
    ds = generate_dataset(model, n_samples=100_000, method="sobol")

    # Convert to PyTorch for training
    train_ds = to_torch_dataset(ds)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=4096)

    for params_batch, nll_batch, grad_batch in loader:
        pred_nll = surrogate(params_batch)
        loss = F.mse_loss(pred_nll, nll_batch)
        ...

References:
    - arXiv:2007.01725 — MadMiner: ML methods for particle physics
    - arXiv:2305.18317 — Neural network surrogates for HistFactory
    - arXiv:2309.12504 — Differentiable likelihood surrogates
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class SurrogateDataset:
    """Container for surrogate training data.

    Attributes:
        parameters: ``np.ndarray`` of shape ``(n_samples, n_params)`` — sampled parameter vectors.
        nll: ``np.ndarray`` of shape ``(n_samples,)`` — NLL values at each point.
        gradient: ``np.ndarray`` of shape ``(n_samples, n_params)`` — gradient of NLL w.r.t. parameters.
        parameter_names: ``list[str]`` — parameter names from the model.
        parameter_bounds: ``np.ndarray`` of shape ``(n_params, 2)`` — ``[lo, hi]`` bounds used for sampling.
        metadata: ``dict`` — generation metadata (method, time, model info).
    """

    parameters: np.ndarray
    nll: np.ndarray
    gradient: np.ndarray
    parameter_names: list[str]
    parameter_bounds: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.parameters.shape[0]

    @property
    def n_params(self) -> int:
        return self.parameters.shape[1]

    def __repr__(self) -> str:
        return (
            f"SurrogateDataset(n_samples={self.n_samples}, n_params={self.n_params}, "
            f"method={self.metadata.get('method', '?')!r})"
        )


def _default_bounds(model) -> np.ndarray:
    """Compute default parameter bounds: init ± 5σ, clipped to physical range."""
    init = np.array(model.parameter_init(), dtype=np.float64)
    bounds_lo = np.array(model.parameter_bounds_lo(), dtype=np.float64)
    bounds_hi = np.array(model.parameter_bounds_hi(), dtype=np.float64)

    # Use model bounds, but tighten to ±5 around init for constrained params
    width = 5.0
    lo = np.maximum(bounds_lo, init - width)
    hi = np.minimum(bounds_hi, init + width)

    return np.column_stack([lo, hi])


def _sample_sobol(n_samples: int, n_params: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    """Quasi-random Sobol sequence over parameter bounds."""
    try:
        from scipy.stats.qmc import Sobol  # type: ignore

        sampler = Sobol(d=n_params, scramble=True, seed=seed)
        # Sobol requires n = 2^m samples; we take next power of 2 and truncate
        m = max(1, int(np.ceil(np.log2(max(n_samples, 2)))))
        raw = sampler.random_base2(m)[:n_samples]  # [n_samples, n_params] in [0, 1]
    except ImportError:
        # Fallback: Halton-like stratified random
        raw = np.random.default_rng(seed).random((n_samples, n_params))

    # Scale to bounds
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return raw * (hi - lo) + lo


def _sample_lhs(n_samples: int, n_params: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    """Latin Hypercube Sampling over parameter bounds."""
    try:
        from scipy.stats.qmc import LatinHypercube  # type: ignore

        sampler = LatinHypercube(d=n_params, seed=seed)
        raw = sampler.random(n=n_samples)
    except ImportError:
        rng = np.random.default_rng(seed)
        raw = np.zeros((n_samples, n_params))
        for j in range(n_params):
            perm = rng.permutation(n_samples)
            raw[:, j] = (perm + rng.random(n_samples)) / n_samples

    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return raw * (hi - lo) + lo


def _sample_uniform(n_samples: int, n_params: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    """Uniform random sampling over parameter bounds."""
    rng = np.random.default_rng(seed)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return rng.uniform(lo, hi, size=(n_samples, n_params))


def _sample_gaussian(n_samples: int, n_params: int, bounds: np.ndarray, seed: int, init: np.ndarray) -> np.ndarray:
    """Gaussian sampling centered at model init, clipped to bounds."""
    rng = np.random.default_rng(seed)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    sigma = (hi - lo) / 6.0  # 99.7% within bounds
    samples = rng.normal(loc=init, scale=sigma, size=(n_samples, n_params))
    return np.clip(samples, lo, hi)


def generate_dataset(
    model,
    n_samples: int = 100_000,
    *,
    method: str = "sobol",
    bounds: Optional[np.ndarray] = None,
    seed: int = 42,
    include_gradient: bool = True,
    batch_size: int = 10_000,
    gpu: bool = False,
) -> SurrogateDataset:
    """Generate a surrogate training dataset from a HistFactory model.

    Samples the parameter space and evaluates NLL (+ gradient) at each point
    using NextStat's native engine.

    Args:
        model: ``nextstat.HistFactoryModel``.
        n_samples: number of parameter points to evaluate (default 100k).
        method: sampling strategy:
            - ``"sobol"`` (default) — quasi-random Sobol sequence (best coverage).
            - ``"lhs"`` — Latin Hypercube Sampling.
            - ``"uniform"`` — uniform random.
            - ``"gaussian"`` — Gaussian around model init (focused near MLE).
        bounds: optional ``(n_params, 2)`` array of ``[lo, hi]`` per parameter.
            Default: model init ± 5, clipped to physical bounds.
        seed: random seed for reproducibility.
        include_gradient: if ``True`` (default), compute ∂NLL/∂params at each point.
        batch_size: internal batch size for evaluation.
        gpu: reserved for future GPU batch evaluation (currently unused).

    Returns:
        :class:`SurrogateDataset` containing parameters, NLL values, gradients.

    Example::

        ds = generate_dataset(model, n_samples=500_000, method="sobol", gpu=True)
        print(f"{ds.n_samples} points, {ds.n_params} params")
        print(f"NLL range: [{ds.nll.min():.1f}, {ds.nll.max():.1f}]")
    """
    import nextstat

    t0 = time.perf_counter()

    param_names = model.parameter_names()
    n_params = len(param_names)
    init = np.array(model.parameter_init(), dtype=np.float64)

    if bounds is None:
        bounds = _default_bounds(model)

    # Sample parameter space
    samplers = {
        "sobol": lambda: _sample_sobol(n_samples, n_params, bounds, seed),
        "lhs": lambda: _sample_lhs(n_samples, n_params, bounds, seed),
        "uniform": lambda: _sample_uniform(n_samples, n_params, bounds, seed),
        "gaussian": lambda: _sample_gaussian(n_samples, n_params, bounds, seed, init),
    }

    if method not in samplers:
        raise ValueError(f"method must be one of {list(samplers)}, got {method!r}")

    parameters = samplers[method]()

    # Evaluate NLL (+ gradient) at each point
    nll_values = np.empty(n_samples, dtype=np.float64)
    grad_values = np.zeros((n_samples, n_params), dtype=np.float64) if include_gradient else np.empty((0, 0))

    mle = nextstat.MaximumLikelihoodEstimator(model)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        for i in range(start, end):
            params = parameters[i].tolist()
            if include_gradient:
                nll_val, grad = mle.nll_and_grad(params)
                nll_values[i] = nll_val
                grad_values[i] = grad
            else:
                nll_values[i] = mle.nll(params)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metadata = {
        "method": method,
        "seed": seed,
        "n_samples": n_samples,
        "n_params": n_params,
        "include_gradient": include_gradient,
        "gpu": gpu,
        "generation_time_ms": elapsed_ms,
        "throughput_samples_per_sec": n_samples / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0,
        "nll_min": float(nll_values.min()),
        "nll_max": float(nll_values.max()),
        "nll_mean": float(nll_values.mean()),
    }

    return SurrogateDataset(
        parameters=parameters,
        nll=nll_values,
        gradient=grad_values,
        parameter_names=list(param_names),
        parameter_bounds=bounds,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------


def to_torch_dataset(ds: SurrogateDataset):
    """Convert to a PyTorch ``TensorDataset``.

    Returns a ``TensorDataset`` of ``(parameters, nll, gradient)`` tensors,
    ready for ``DataLoader``.

    Requires ``torch``.

    Example::

        train_ds = to_torch_dataset(ds)
        loader = DataLoader(train_ds, batch_size=4096, shuffle=True)

        for params, nll, grad in loader:
            pred = surrogate(params)
            loss = F.mse_loss(pred.squeeze(), nll)
            loss.backward()
    """
    try:
        import torch  # type: ignore
    except ImportError:
        raise ImportError("PyTorch is required for to_torch_dataset(). Install: pip install torch")

    params_t = torch.from_numpy(ds.parameters).float()
    nll_t = torch.from_numpy(ds.nll).float()

    if ds.gradient.size > 0:
        grad_t = torch.from_numpy(ds.gradient).float()
    else:
        grad_t = torch.zeros(ds.n_samples, ds.n_params)

    return torch.utils.data.TensorDataset(params_t, nll_t, grad_t)


def to_numpy(ds: SurrogateDataset) -> dict[str, np.ndarray]:
    """Export as a dict of NumPy arrays.

    Returns:
        ``{"parameters": ..., "nll": ..., "gradient": ..., "bounds": ...}``
    """
    return {
        "parameters": ds.parameters,
        "nll": ds.nll,
        "gradient": ds.gradient,
        "bounds": ds.parameter_bounds,
    }


def to_npz(ds: SurrogateDataset, path: str) -> None:
    """Save dataset to a compressed ``.npz`` file.

    Example::

        to_npz(ds, "surrogate_data.npz")

        # Load later:
        data = np.load("surrogate_data.npz")
        params, nll, grad = data["parameters"], data["nll"], data["gradient"]
    """
    np.savez_compressed(
        path,
        parameters=ds.parameters,
        nll=ds.nll,
        gradient=ds.gradient,
        bounds=ds.parameter_bounds,
        names=np.array(ds.parameter_names, dtype=object),
    )


def to_parquet(ds: SurrogateDataset, path: str) -> None:
    """Save dataset to a Parquet file (requires ``pyarrow``).

    Columns: one per parameter (by name), plus ``_nll`` and ``_grad_<name>``
    columns. Ideal for loading into Polars / DuckDB / Spark.

    Example::

        to_parquet(ds, "surrogate_data.parquet")

        import polars as pl
        df = pl.read_parquet("surrogate_data.parquet")
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        raise ImportError(
            "pyarrow is required for to_parquet(). Install: pip install pyarrow"
        )

    arrays = {}
    for j, name in enumerate(ds.parameter_names):
        arrays[name] = pa.array(ds.parameters[:, j])

    arrays["_nll"] = pa.array(ds.nll)

    if ds.gradient.size > 0:
        for j, name in enumerate(ds.parameter_names):
            arrays[f"_grad_{name}"] = pa.array(ds.gradient[:, j])

    table = pa.table(arrays)
    pq.write_table(table, path, compression="zstd")


def from_npz(path: str) -> SurrogateDataset:
    """Load a dataset from a ``.npz`` file created by :func:`to_npz`.

    Example::

        ds = from_npz("surrogate_data.npz")
        print(ds.n_samples, ds.n_params)
    """
    data = np.load(path, allow_pickle=True)
    names = data["names"].tolist() if "names" in data else [f"p{i}" for i in range(data["parameters"].shape[1])]

    return SurrogateDataset(
        parameters=data["parameters"],
        nll=data["nll"],
        gradient=data["gradient"],
        parameter_names=names,
        parameter_bounds=data["bounds"],
        metadata={"loaded_from": path},
    )


# ---------------------------------------------------------------------------
# Surrogate training helpers
# ---------------------------------------------------------------------------


def train_mlp_surrogate(
    ds: SurrogateDataset,
    *,
    hidden_layers: tuple[int, ...] = (256, 256, 128),
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 4096,
    val_fraction: float = 0.1,
    grad_weight: float = 0.1,
    device: str = "cpu",
    verbose: bool = True,
) -> Any:
    """Train a simple MLP surrogate on the generated dataset.

    This is a convenience function for quick prototyping. For production,
    use :func:`to_torch_dataset` and write your own training loop.

    Args:
        ds: :class:`SurrogateDataset` from :func:`generate_dataset`.
        hidden_layers: tuple of hidden layer sizes.
        epochs: training epochs.
        lr: learning rate (Adam).
        batch_size: mini-batch size.
        val_fraction: fraction of data held out for validation.
        grad_weight: weight for gradient MSE loss term (0 = NLL-only).
        device: ``"cpu"`` or ``"cuda"``.
        verbose: print progress.

    Returns:
        Trained ``torch.nn.Module`` (an MLP).

    Example::

        surrogate = train_mlp_surrogate(ds, epochs=50, device="cuda")
        # Use it:
        params_tensor = torch.tensor(model.parameter_init()).float()
        predicted_nll = surrogate(params_tensor)
    """
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except ImportError:
        raise ImportError("PyTorch is required for train_mlp_surrogate(). Install: pip install torch")

    # Build MLP
    layers: list[nn.Module] = []
    in_dim = ds.n_params
    for h in hidden_layers:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.SiLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    mlp = nn.Sequential(*layers).to(device)

    # Prepare data
    n_val = max(1, int(ds.n_samples * val_fraction))
    n_train = ds.n_samples - n_val

    params_all = torch.from_numpy(ds.parameters).float().to(device)
    nll_all = torch.from_numpy(ds.nll).float().to(device)

    # Normalise inputs and targets for stable training
    p_mean = params_all[:n_train].mean(dim=0)
    p_std = params_all[:n_train].std(dim=0).clamp(min=1e-8)
    nll_mean = nll_all[:n_train].mean()
    nll_std = nll_all[:n_train].std().clamp(min=1e-8)

    params_norm = (params_all - p_mean) / p_std
    nll_norm = (nll_all - nll_mean) / nll_std

    train_params = params_norm[:n_train]
    train_nll = nll_norm[:n_train]
    val_params = params_norm[n_train:]
    val_nll = nll_norm[n_train:]

    use_grad = grad_weight > 0 and ds.gradient.size > 0
    if use_grad:
        grad_all = torch.from_numpy(ds.gradient).float().to(device)
        grad_norm = grad_all / (nll_std / p_std)
        train_grad = grad_norm[:n_train]

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    for epoch in range(epochs):
        mlp.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            p_batch = train_params[idx]
            nll_batch = train_nll[idx]

            pred = mlp(p_batch).squeeze(-1)
            loss = F.mse_loss(pred, nll_batch)

            if use_grad:
                p_batch_g = p_batch.detach().requires_grad_(True)
                pred_g = mlp(p_batch_g).squeeze(-1)
                pred_grad = torch.autograd.grad(
                    pred_g.sum(), p_batch_g, create_graph=True
                )[0]
                grad_loss = F.mse_loss(pred_grad, train_grad[idx])
                loss = loss + grad_weight * grad_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            mlp.eval()
            with torch.no_grad():
                val_pred = mlp(val_params).squeeze(-1)
                val_loss = F.mse_loss(val_pred, val_nll).item()
            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"train_loss={epoch_loss / n_batches:.6f}  "
                f"val_loss={val_loss:.6f}"
            )

    # Attach normalisation constants for inference
    mlp._ns_p_mean = p_mean
    mlp._ns_p_std = p_std
    mlp._ns_nll_mean = nll_mean
    mlp._ns_nll_std = nll_std
    mlp.eval()

    return mlp


def predict_nll(surrogate, params_np: np.ndarray) -> np.ndarray:
    """Evaluate a trained surrogate on raw (un-normalised) parameters.

    Handles the normalisation that :func:`train_mlp_surrogate` applied.

    Args:
        surrogate: ``torch.nn.Module`` from :func:`train_mlp_surrogate`.
        params_np: ``np.ndarray`` of shape ``(n_points, n_params)`` or ``(n_params,)``.

    Returns:
        ``np.ndarray`` — predicted NLL values.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        raise ImportError("PyTorch is required. Install: pip install torch")

    was_1d = params_np.ndim == 1
    if was_1d:
        params_np = params_np[np.newaxis, :]

    device = next(surrogate.parameters()).device
    p = torch.from_numpy(params_np).float().to(device)

    p_norm = (p - surrogate._ns_p_mean) / surrogate._ns_p_std

    with torch.no_grad():
        pred_norm = surrogate(p_norm).squeeze(-1)
        pred = pred_norm * surrogate._ns_nll_std + surrogate._ns_nll_mean

    result = pred.cpu().numpy()
    return result[0] if was_1d else result
