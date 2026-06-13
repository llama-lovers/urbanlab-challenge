"""
Ingest raw files from /data/raw into Parquet in /data/processed.

Usage:
    python ingest.py                   # process all CSV/JSON/Parquet in data/raw/
    python ingest.py -f data/raw/x.csv # process one file
"""

from pathlib import Path

import click
import polars as pl

RAW = Path("/data/raw")
PROCESSED = Path("/data/processed")


def _scan(path: Path) -> pl.LazyFrame:
    match path.suffix.lower():
        case ".csv":
            return pl.scan_csv(path, infer_schema_length=10_000)
        case ".json" | ".ndjson":
            return pl.scan_ndjson(path)
        case ".parquet":
            return pl.scan_parquet(path)
        case _:
            raise ValueError(f"Unsupported file type: {path.suffix}")


@click.command()
@click.option("--file", "-f", default=None, type=click.Path(), help="Single file to ingest")
def ingest(file: str | None) -> None:
    if file:
        paths = [Path(file)]
    else:
        paths = [
            p for p in RAW.iterdir() if p.suffix.lower() in {".csv", ".json", ".ndjson", ".parquet"}
        ]

    if not paths:
        click.echo("No files found in /data/raw/")
        return

    PROCESSED.mkdir(parents=True, exist_ok=True)

    for path in paths:
        click.echo(f"Ingesting {path.name}…")
        out = PROCESSED / path.with_suffix(".parquet").name
        _scan(path).sink_parquet(out)
        click.echo(f"  → {out}")


if __name__ == "__main__":
    ingest()
