"""DuckDB-powered in-process analytics for large CSV/Parquet files."""

from pathlib import Path
from typing import Any

import duckdb

DATA_DIR = Path("/data")


def query_file(sql: str) -> list[dict[str, Any]]:
    """Run SQL against files using DuckDB. Supports read_csv_auto, read_parquet, etc."""
    with duckdb.connect() as con:
        return con.execute(sql).df().to_dict("records")


def file_summary(file_path: str | Path, value_col: str = "value") -> dict[str, Any]:
    """Quick descriptive stats for a CSV/Parquet file."""
    with duckdb.connect() as con:
        row = con.execute(f"""
            SELECT
                count(*)        AS total,
                min({value_col}) AS min_val,
                max({value_col}) AS max_val,
                avg({value_col}) AS avg_val,
                stddev({value_col}) AS std_val
            FROM read_csv_auto('{file_path}')
        """).fetchone()
    keys = ("total", "min_val", "max_val", "avg_val", "std_val")
    return dict(zip(keys, row, strict=False)) if row else {}


def group_by_column(file_path: str | Path, group_col: str, value_col: str = "value") -> list[dict]:
    """Aggregate a file by a categorical column."""
    with duckdb.connect() as con:
        return (
            con.execute(f"""
            SELECT {group_col}, count(*) AS n, sum({value_col}) AS total
            FROM read_csv_auto('{file_path}')
            GROUP BY {group_col}
            ORDER BY total DESC
        """)
            .df()
            .to_dict("records")
        )
