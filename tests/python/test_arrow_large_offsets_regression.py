from __future__ import annotations

from pathlib import Path

import pytest

nextstat = pytest.importorskip("nextstat")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pl = pytest.importorskip("polars")
duckdb = pytest.importorskip("duckdb")


def _fit_mu_hat(model) -> float:
    fit = nextstat.fit(model)
    return float(fit.bestfit[0])


def _make_large_table():
    return pa.table(
        {
            "channel": pa.array(["SR", "SR"], type=pa.large_string()),
            "sample": pa.array(["signal", "background"], type=pa.large_string()),
            "yields": pa.array(
                [[5.0, 10.0], [50.0, 60.0]],
                type=pa.large_list(pa.field("item", pa.float64())),
            ),
            "stat_error": pa.array(
                [[1.0, 1.5], [5.0, 7.0]],
                type=pa.large_list(pa.field("item", pa.float64())),
            ),
        }
    )


def test_from_arrow_accepts_large_offsets_for_polars_and_duckdb(tmp_path: Path) -> None:
    table = _make_large_table()
    parquet_path = tmp_path / "histograms.parquet"
    pq.write_table(table, parquet_path, compression="snappy")

    obs = {"SR": [55.0, 65.0]}

    # Baseline path
    mu_parquet = _fit_mu_hat(nextstat.from_parquet(str(parquet_path), poi="mu", observations=obs))

    # Direct large-offset Arrow table path
    mu_large_arrow = _fit_mu_hat(nextstat.from_arrow(table, poi="mu", observations=obs))

    # Polars -> Arrow (typically LargeUtf8/LargeList)
    pl_df = pl.read_parquet(str(parquet_path))
    mu_polars = _fit_mu_hat(nextstat.from_arrow(pl_df.to_arrow(), poi="mu", observations=obs))

    # DuckDB -> Arrow reader -> Table
    con = duckdb.connect()
    reader = con.execute(f"SELECT * FROM '{parquet_path}'").arrow()
    duck_arrow = reader.read_all() if hasattr(reader, "read_all") else reader
    mu_duckdb = _fit_mu_hat(nextstat.from_arrow(duck_arrow, poi="mu", observations=obs))

    assert mu_large_arrow == pytest.approx(mu_parquet, rel=1e-10, abs=1e-10)
    assert mu_polars == pytest.approx(mu_parquet, rel=1e-10, abs=1e-10)
    assert mu_duckdb == pytest.approx(mu_parquet, rel=1e-10, abs=1e-10)
