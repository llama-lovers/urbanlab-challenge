#!/bin/bash
set -e
echo "=== ETL Pipeline ==="
echo "--- ingest ---"
python ingest.py
echo "--- transform ---"
python transform.py
echo "--- load ---"
python load.py
echo "=== Done ==="
