"""Thin HTTP client for a remote NextStat inference server.

Usage::

    import nextstat.remote as remote

    client = remote.connect("http://gpu-server:3742")
    result = client.fit(workspace_json)
    ranking = client.ranking(workspace_json)
    health = client.health()

Requires ``httpx`` (``pip install httpx``).  No native extensions needed —
this module is pure Python so it can run on any machine without Rust/CUDA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def connect(url: str, *, timeout: float = 300.0) -> "NextStatClient":
    """Create a client connected to a NextStat server.

    Args:
        url: Base URL of the server, e.g. ``"http://gpu-server:3742"``.
        timeout: Request timeout in seconds (default 300 s — fits can be slow).

    Returns:
        :class:`NextStatClient`
    """
    return NextStatClient(url.rstrip("/"), timeout=timeout)


@dataclass
class FitResult:
    """Result of a remote ``/v1/fit`` call."""

    parameter_names: list[str]
    poi_index: int | None
    bestfit: list[float]
    uncertainties: list[float]
    nll: float
    twice_nll: float
    converged: bool
    n_iter: int
    n_fev: int
    n_gev: int
    covariance: list[float] | None
    device: str
    wall_time_s: float
    _raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class RankingEntry:
    """Single entry in a ranking result."""

    name: str
    delta_mu_up: float
    delta_mu_down: float
    pull: float
    constraint: float


@dataclass
class RankingResult:
    """Result of a remote ``/v1/ranking`` call."""

    entries: list[RankingEntry]
    device: str
    wall_time_s: float
    _raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class BatchFitResult:
    """Result of a remote ``/v1/batch/fit`` call."""

    results: list[FitResult | None]
    errors: list[str | None]
    device: str
    wall_time_s: float
    _raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class ToyFitItem:
    """Single toy fit result from ``/v1/batch/toys``."""

    bestfit: list[float]
    nll: float
    converged: bool
    n_iter: int


@dataclass
class BatchToysResult:
    """Result of a remote ``/v1/batch/toys`` call."""

    n_toys: int
    n_converged: int
    results: list[ToyFitItem]
    device: str
    wall_time_s: float
    _raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class ModelInfo:
    """Info about a cached model on the server."""

    model_id: str
    name: str
    n_params: int
    n_channels: int
    age_s: float
    last_used_s: float
    hit_count: int


@dataclass
class HealthResult:
    """Result of a remote ``/v1/health`` call."""

    status: str
    version: str
    uptime_s: float
    device: str
    inflight: int
    total_requests: int
    cached_models: int
    _raw: dict[str, Any] = field(repr=False, default_factory=dict)


class NextStatClient:
    """HTTP client for a remote NextStat inference server.

    Do not instantiate directly — use :func:`connect`.
    """

    def __init__(self, base_url: str, *, timeout: float = 300.0) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "nextstat.remote requires httpx. Install it with: pip install httpx"
            ) from exc

        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def __repr__(self) -> str:
        return f"NextStatClient({self._base_url!r})"

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._client.close()

    def __enter__(self) -> "NextStatClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # POST /v1/fit
    # ------------------------------------------------------------------

    def fit(
        self,
        workspace: dict | str | None = None,
        *,
        model_id: str | None = None,
        gpu: bool = True,
    ) -> FitResult:
        """Run a maximum-likelihood fit on the remote server.

        Args:
            workspace: pyhf/HS3 workspace as a ``dict`` or JSON string.
                Can be omitted if *model_id* is given.
            model_id: Cached model ID (from :meth:`upload_model`). Skips
                workspace parsing on the server.
            gpu: Whether to request GPU acceleration (default ``True``).

        Returns:
            :class:`FitResult`
        """
        body: dict[str, Any] = {"gpu": gpu}
        if model_id is not None:
            body["model_id"] = model_id
        elif workspace is not None:
            body["workspace"] = _ensure_dict(workspace)
        else:
            raise ValueError("either workspace or model_id must be provided")
        resp = self._client.post("/v1/fit", json=body)
        _check(resp)
        data = resp.json()
        return FitResult(
            parameter_names=data["parameter_names"],
            poi_index=data.get("poi_index"),
            bestfit=data["bestfit"],
            uncertainties=data["uncertainties"],
            nll=data["nll"],
            twice_nll=data["twice_nll"],
            converged=data["converged"],
            n_iter=data["n_iter"],
            n_fev=data["n_fev"],
            n_gev=data["n_gev"],
            covariance=data.get("covariance"),
            device=data["device"],
            wall_time_s=data["wall_time_s"],
            _raw=data,
        )

    # ------------------------------------------------------------------
    # POST /v1/ranking
    # ------------------------------------------------------------------

    def ranking(
        self,
        workspace: dict | str | None = None,
        *,
        model_id: str | None = None,
        gpu: bool = True,
    ) -> RankingResult:
        """Compute nuisance-parameter ranking on the remote server.

        Args:
            workspace: pyhf/HS3 workspace as a ``dict`` or JSON string.
                Can be omitted if *model_id* is given.
            model_id: Cached model ID (from :meth:`upload_model`).
            gpu: Whether to request GPU acceleration (default ``True``).

        Returns:
            :class:`RankingResult`
        """
        body: dict[str, Any] = {"gpu": gpu}
        if model_id is not None:
            body["model_id"] = model_id
        elif workspace is not None:
            body["workspace"] = _ensure_dict(workspace)
        else:
            raise ValueError("either workspace or model_id must be provided")
        resp = self._client.post("/v1/ranking", json=body)
        _check(resp)
        data = resp.json()
        entries = [
            RankingEntry(
                name=e["name"],
                delta_mu_up=e["delta_mu_up"],
                delta_mu_down=e["delta_mu_down"],
                pull=e["pull"],
                constraint=e["constraint"],
            )
            for e in data["entries"]
        ]
        return RankingResult(
            entries=entries,
            device=data["device"],
            wall_time_s=data["wall_time_s"],
            _raw=data,
        )

    # ------------------------------------------------------------------
    # GET /v1/health
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # POST /v1/batch/fit
    # ------------------------------------------------------------------

    def batch_fit(
        self,
        workspaces: list[dict | str],
        *,
        gpu: bool = True,
    ) -> BatchFitResult:
        """Fit multiple workspaces in one request.

        Args:
            workspaces: List of pyhf/HS3 workspace dicts or JSON strings.
            gpu: Whether to request GPU acceleration (default ``True``).

        Returns:
            :class:`BatchFitResult`
        """
        ws_list = [_ensure_dict(w) for w in workspaces]
        resp = self._client.post(
            "/v1/batch/fit", json={"workspaces": ws_list, "gpu": gpu}
        )
        _check(resp)
        data = resp.json()
        results: list[FitResult | None] = []
        errors: list[str | None] = []
        for item in data["results"]:
            if item.get("error"):
                results.append(None)
                errors.append(item["error"])
            else:
                results.append(
                    FitResult(
                        parameter_names=item["parameter_names"],
                        poi_index=item.get("poi_index"),
                        bestfit=item["bestfit"],
                        uncertainties=item["uncertainties"],
                        nll=item["nll"],
                        twice_nll=item["twice_nll"],
                        converged=item["converged"],
                        n_iter=item["n_iter"],
                        n_fev=item["n_fev"],
                        n_gev=item["n_gev"],
                        covariance=item.get("covariance"),
                        device=item["device"],
                        wall_time_s=item["wall_time_s"],
                        _raw=item,
                    )
                )
                errors.append(None)
        return BatchFitResult(
            results=results,
            errors=errors,
            device=data["device"],
            wall_time_s=data["wall_time_s"],
            _raw=data,
        )

    # ------------------------------------------------------------------
    # POST /v1/batch/toys
    # ------------------------------------------------------------------

    def batch_toys(
        self,
        workspace: dict | str,
        *,
        params: list[float] | None = None,
        n_toys: int = 1000,
        seed: int = 42,
        gpu: bool = True,
    ) -> BatchToysResult:
        """Run batch toy fitting on the remote server.

        Args:
            workspace: pyhf/HS3 workspace as a ``dict`` or JSON string.
            params: Parameters to generate toys at. If ``None``, uses the
                model's default initial parameters.
            n_toys: Number of pseudo-experiments (default 1000).
            seed: Random seed (default 42).
            gpu: Whether to request GPU acceleration (default ``True``).

        Returns:
            :class:`BatchToysResult`
        """
        body: dict[str, Any] = {
            "workspace": _ensure_dict(workspace),
            "n_toys": n_toys,
            "seed": seed,
            "gpu": gpu,
        }
        if params is not None:
            body["params"] = params
        resp = self._client.post("/v1/batch/toys", json=body)
        _check(resp)
        data = resp.json()
        items = [
            ToyFitItem(
                bestfit=t["bestfit"],
                nll=t["nll"],
                converged=t["converged"],
                n_iter=t["n_iter"],
            )
            for t in data["results"]
        ]
        return BatchToysResult(
            n_toys=data["n_toys"],
            n_converged=data["n_converged"],
            results=items,
            device=data["device"],
            wall_time_s=data["wall_time_s"],
            _raw=data,
        )

    # ------------------------------------------------------------------
    # Model pool: POST / GET / DELETE /v1/models
    # ------------------------------------------------------------------

    def upload_model(
        self,
        workspace: dict | str,
        *,
        name: str | None = None,
    ) -> str:
        """Upload a workspace to the server's model cache.

        Returns the ``model_id`` (SHA-256 hash) that can be passed to
        :meth:`fit` and :meth:`ranking` to skip re-parsing.

        Args:
            workspace: pyhf/HS3 workspace as a ``dict`` or JSON string.
            name: Optional human-readable name for the cached model.

        Returns:
            Model ID string.
        """
        body: dict[str, Any] = {"workspace": _ensure_dict(workspace)}
        if name is not None:
            body["name"] = name
        resp = self._client.post("/v1/models", json=body)
        _check(resp)
        return resp.json()["model_id"]

    def list_models(self) -> list[ModelInfo]:
        """List all models in the server's cache.

        Returns:
            List of :class:`ModelInfo`.
        """
        resp = self._client.get("/v1/models")
        _check(resp)
        return [
            ModelInfo(
                model_id=m["model_id"],
                name=m["name"],
                n_params=m["n_params"],
                n_channels=m["n_channels"],
                age_s=m["age_s"],
                last_used_s=m["last_used_s"],
                hit_count=m["hit_count"],
            )
            for m in resp.json()
        ]

    def delete_model(self, model_id: str) -> bool:
        """Remove a model from the server's cache.

        Args:
            model_id: Model ID returned by :meth:`upload_model`.

        Returns:
            ``True`` if the model was deleted.
        """
        resp = self._client.delete(f"/v1/models/{model_id}")
        _check(resp)
        return resp.json().get("deleted", False)

    # ------------------------------------------------------------------
    # GET /v1/health
    # ------------------------------------------------------------------

    def health(self) -> HealthResult:
        """Check server health.

        Returns:
            :class:`HealthResult`
        """
        resp = self._client.get("/v1/health")
        _check(resp)
        data = resp.json()
        return HealthResult(
            status=data["status"],
            version=data["version"],
            uptime_s=data["uptime_s"],
            device=data["device"],
            inflight=data["inflight"],
            total_requests=data["total_requests"],
            cached_models=data.get("cached_models", 0),
            _raw=data,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _ensure_dict(workspace: dict | str) -> dict:
    """Accept dict or JSON string, return dict."""
    if isinstance(workspace, str):
        return json.loads(workspace)
    return workspace


class NextStatServerError(Exception):
    """Raised when the server returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


def _check(resp: Any) -> None:
    """Raise :class:`NextStatServerError` on non-2xx responses."""
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        raise NextStatServerError(resp.status_code, detail)
