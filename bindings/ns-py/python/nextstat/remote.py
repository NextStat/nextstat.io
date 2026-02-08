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
class HealthResult:
    """Result of a remote ``/v1/health`` call."""

    status: str
    version: str
    uptime_s: float
    device: str
    inflight: int
    total_requests: int
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
        workspace: dict | str,
        *,
        gpu: bool = True,
    ) -> FitResult:
        """Run a maximum-likelihood fit on the remote server.

        Args:
            workspace: pyhf/HS3 workspace as a ``dict`` or JSON string.
            gpu: Whether to request GPU acceleration (default ``True``).

        Returns:
            :class:`FitResult`
        """
        ws = _ensure_dict(workspace)
        resp = self._client.post("/v1/fit", json={"workspace": ws, "gpu": gpu})
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
        workspace: dict | str,
        *,
        gpu: bool = True,
    ) -> RankingResult:
        """Compute nuisance-parameter ranking on the remote server.

        Args:
            workspace: pyhf/HS3 workspace as a ``dict`` or JSON string.
            gpu: Whether to request GPU acceleration (default ``True``).

        Returns:
            :class:`RankingResult`
        """
        ws = _ensure_dict(workspace)
        resp = self._client.post("/v1/ranking", json={"workspace": ws, "gpu": gpu})
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
