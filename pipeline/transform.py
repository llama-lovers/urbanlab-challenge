"""
Clean and enrich ingested Parquet files.

Edit the TRANSFORMS block below to match your dataset's schema.
Output is saved as clean_<original>.parquet in /data/processed/.

Usage:
    python transform.py
"""

from pathlib import Path

import click
import polars as pl

PROCESSED = Path("/data/processed")


def _transform(lf: pl.LazyFrame) -> pl.LazyFrame:
    # ---- Adapt this section to your schema ----
    #
    # Example: rename raw columns, drop nulls, clip coordinates, fill defaults
    #
    # lf = lf.rename({"latitude": "lat", "longitude": "lon", "type": "category"})
    # lf = lf.filter(
    #     pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
    # )
    # lf = lf.filter(
    #     pl.col("lat").is_between(-90, 90) & pl.col("lon").is_between(-180, 180)
    # )
    # lf = lf.with_columns(
    #     pl.col("value").fill_null(1.0),
    #     pl.col("category").fill_null("default"),
    # )
    #
    # -------------------------------------------
    return lf


@click.command()
def transform() -> None:
    parquets = [p for p in PROCESSED.glob("*.parquet") if not p.name.startswith("clean_")]

    if not parquets:
        click.echo("No raw Parquet files found. Run ingest.py first.")
        return

    for path in parquets:
        click.echo(f"Transforming {path.name}…")
        lf = _transform(pl.scan_parquet(path))
        out = PROCESSED / f"clean_{path.name}"
        lf.sink_parquet(out)
        click.echo(f"  → {out}")


if __name__ == "__main__":
    transform()
