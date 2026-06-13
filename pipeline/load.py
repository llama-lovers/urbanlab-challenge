"""
Load clean_*.parquet files into Postgres.

Expected schema for the default 'events' table:
  lat (float), lon (float), category (str), value (float), [attributes (dict)]

Any extra columns are ignored unless you change INSERT_COLS below.

Usage:
    python load.py
    python load.py --table my_table --truncate
"""

import os
from pathlib import Path

import click
import polars as pl
import psycopg2
import psycopg2.extras

PROCESSED = Path("/data/processed")
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/hackathon")

INSERT_COLS = ["lat", "lon", "category", "value"]


def _build_geom(lon: float, lat: float) -> str:
    return f"SRID=4326;POINT({lon} {lat})"


@click.command()
@click.option("--table", default="events", show_default=True)
@click.option("--truncate", is_flag=True, help="Truncate table before loading")
def load(table: str, truncate: bool) -> None:
    files = sorted(PROCESSED.glob("clean_*.parquet"))
    if not files:
        click.echo("No clean_*.parquet files. Run transform.py first.")
        return

    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            if truncate:
                cur.execute(f"TRUNCATE TABLE {table}")
                click.echo(f"Truncated {table}")

            for path in files:
                df = pl.read_parquet(path)

                available = [c for c in INSERT_COLS if c in df.columns]
                rows = [
                    {
                        "geom": _build_geom(r["lon"], r["lat"]),
                        "category": r.get("category", "default"),
                        "value": float(r.get("value", 1.0)),
                    }
                    for r in df.select(available).to_dicts()
                ]

                sql = f"""
                    INSERT INTO {table} (geom, category, value)
                    VALUES (%(geom)s, %(category)s, %(value)s)
                """
                psycopg2.extras.execute_batch(cur, sql, rows, page_size=1000)
                conn.commit()
                click.echo(f"  {path.name}: {len(rows)} rows → {table}")
    finally:
        conn.close()


if __name__ == "__main__":
    load()
