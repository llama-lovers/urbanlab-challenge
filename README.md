# Hackathon Template

Urban data analytics starter for UrbanLab Lublin.
Stack: **FastAPI + PostgreSQL/PostGIS + React + deck.gl**.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- [go-task](https://taskfile.dev/installation/) (`task` command)

## Quickstart (< 5 min)

```bash
task setup   # copy .env and build images
task run     # start all services
```

| Service | URL | Notes |
|---------|-----|-------|
| Frontend dashboard | http://localhost:5173 | React + heatmaps |
| Backend API docs | http://localhost:8000/docs | FastAPI Swagger |
| JupyterLab | http://localhost:8888 | token: `hackathon` |

## Common commands

| Command | Action |
|---------|--------|
| `task run` | Start all services |
| `task stop` | Stop all services |
| `task pipeline` | Run ETL pipeline |
| `task logs` | Follow all logs |
| `task logs -- backend` | Follow one service |
| `task db` | Open psql shell |
| `task notebook` | Open JupyterLab in browser |
| `task clean` | Destroy everything (prompts) |

## Adding your data

1. Drop CSV / JSON / Parquet files into `data/raw/`
2. Edit `pipeline/ingest.py` to match your schema
3. Edit `pipeline/transform.py` for cleaning/enrichment
4. `task pipeline` — loads data into Postgres

## Architecture

```
backend/      FastAPI (port 8000)  — REST API, PostGIS queries, DuckDB analytics
frontend/     React + Vite (5173)  — deck.gl heatmaps, Recharts, Tailwind
pipeline/     Click CLI            — polars ingest → transform → load to Postgres
notebooks/    JupyterLab (8888)    — exploration with polars + geopandas
data/         raw / processed / exports  (gitignored, mounted into containers)
scripts/      db_init.sql          — PostGIS setup + example schema
```

## Key patterns

- **Heatmap**: `GET /api/geo/heatmap?category=X` → `[{lat, lon, weight}]` → `HeatmapLayer` component
- **Choropleth**: `GET /api/geo/choropleth` → GeoJSON FeatureCollection → `ChoroplethMap` component
- **DuckDB**: query large CSV/Parquet files in-process without loading into Postgres
- **Polars LazyFrame**: scan large files without loading them fully into RAM
