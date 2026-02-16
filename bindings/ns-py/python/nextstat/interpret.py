"""ML interpretability helpers for NextStat.

Provides Feature-Importance-style APIs that translate physics ranking
plots into ML-familiar formats (sorted impact tables, DataFrames,
matplotlib bar charts).

For ML engineers: the "ranking plot" is the HEP equivalent of
Feature Importance — it shows how much each systematic uncertainty
(latent variable) affects the parameter of interest (POI / signal
strength).

Example::

    import nextstat
    from nextstat.interpret import rank_impact

    model = nextstat.from_pyhf(workspace_json)
    table = rank_impact(model)
    print(table)  # sorted by total impact, highest first
"""

from __future__ import annotations

from typing import Any, Optional


def rank_impact(
    model,
    *,
    gpu: bool = False,
    sort_by: str = "total",
    top_n: Optional[int] = None,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """Compute systematic-impact ranking (Feature Importance for HEP).

    Calls NextStat's native ``ranking()`` (with ``device="cuda"`` for GPU) and
    returns a sorted list of dicts augmented with ``total_impact``.

    Args:
        model: ``nextstat.HistFactoryModel``.
        gpu: if ``True``, use GPU-accelerated ranking (requires CUDA build).
        sort_by: key to sort by. One of:
            - ``"total"`` (default) — ``|delta_mu_up| + |delta_mu_down|``
            - ``"up"`` — ``|delta_mu_up|``
            - ``"down"`` — ``|delta_mu_down|``
            - ``"pull"`` — ``|pull|``
            - ``"name"`` — alphabetical
        top_n: if set, return only the top N entries.
        ascending: if ``True``, sort ascending (default ``False`` = highest
            impact first).

    Returns:
        ``list[dict]`` — each dict has keys:

        - ``name`` — parameter name
        - ``delta_mu_up`` — shift in POI when NP is shifted +1σ
        - ``delta_mu_down`` — shift in POI when NP is shifted −1σ
        - ``total_impact`` — ``|delta_mu_up| + |delta_mu_down|``
        - ``pull`` — post-fit pull value
        - ``constraint`` — constraint sigma
        - ``rank`` — 1-based rank (1 = highest impact)

    ML translation:
        - **name** = systematic / latent variable name
        - **total_impact** = feature importance score
        - **pull** = how far the latent variable moved from its prior
        - **constraint** = prior width (analogous to regularisation strength)

    Example::

        table = rank_impact(model, top_n=10)
        for row in table:
            print(f"{row['rank']:2d}. {row['name']:30s}  impact={row['total_impact']:.4f}")
    """
    import nextstat as ns  # type: ignore

    if gpu:
        raw = ns.ranking(model, device="cuda")
    else:
        raw = ns.ranking(model)

    entries = []
    for r in raw:
        up = r.get("delta_mu_up", 0.0)
        down = r.get("delta_mu_down", 0.0)
        entries.append({
            "name": r.get("name", ""),
            "delta_mu_up": up,
            "delta_mu_down": down,
            "total_impact": abs(up) + abs(down),
            "pull": r.get("pull", 0.0),
            "constraint": r.get("constraint", 0.0),
        })

    sort_keys = {
        "total": lambda e: e["total_impact"],
        "up": lambda e: abs(e["delta_mu_up"]),
        "down": lambda e: abs(e["delta_mu_down"]),
        "pull": lambda e: abs(e["pull"]),
        "name": lambda e: e["name"],
    }
    key_fn = sort_keys.get(sort_by)
    if key_fn is None:
        raise ValueError(
            f"sort_by must be one of {list(sort_keys)}, got {sort_by!r}"
        )

    entries.sort(key=key_fn, reverse=not ascending)

    if top_n is not None and top_n > 0:
        entries = entries[:top_n]

    for i, e in enumerate(entries, 1):
        e["rank"] = i

    return entries


def rank_impact_df(model, **kwargs):
    """Like :func:`rank_impact` but returns a ``pandas.DataFrame``.

    Requires ``pandas`` (optional dependency). All keyword arguments are
    forwarded to :func:`rank_impact`.

    Example::

        df = rank_impact_df(model, top_n=15)
        df.plot.barh(x="name", y="total_impact")
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        raise ImportError(
            "pandas is required for rank_impact_df(). "
            "Install it with: pip install pandas"
        )

    entries = rank_impact(model, **kwargs)
    return pd.DataFrame(entries)


def plot_rank_impact(
    model,
    *,
    top_n: int = 20,
    gpu: bool = False,
    figsize: tuple[float, float] = (8, 6),
    title: str = "Systematic Impact on Signal Strength (μ)",
    ax=None,
    **kwargs,
):
    """Horizontal bar chart of systematic impact (ranking plot).

    Requires ``matplotlib`` (optional dependency).

    Args:
        model: ``nextstat.HistFactoryModel``.
        top_n: show top N systematics (default 20).
        gpu: use GPU-accelerated ranking.
        figsize: figure size if creating a new figure.
        title: plot title.
        ax: optional ``matplotlib.axes.Axes`` to draw on.
        **kwargs: forwarded to ``rank_impact()``.

    Returns:
        ``matplotlib.figure.Figure``

    Example::

        fig = plot_rank_impact(model, top_n=15)
        fig.savefig("ranking.png", dpi=150)
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_rank_impact(). "
            "Install it with: pip install matplotlib"
        )

    entries = rank_impact(model, top_n=top_n, gpu=gpu, **kwargs)
    entries.reverse()

    names = [e["name"] for e in entries]
    ups = [e["delta_mu_up"] for e in entries]
    downs = [e["delta_mu_down"] for e in entries]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_pos = range(len(names))
    ax.barh(y_pos, ups, align="center", color="#2196F3", alpha=0.8, label="+1σ")
    ax.barh(y_pos, downs, align="center", color="#FF9800", alpha=0.8, label="−1σ")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Δμ")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    fig.tight_layout()

    return fig
