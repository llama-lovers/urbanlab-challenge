#!/usr/bin/env python3
"""
Dump the FastAPI OpenAPI schema to backend/openapi.json.

Importing the app is safe offline — the engine is lazy and lifespan/create_all
only run under uvicorn, so no Postgres connection is needed.

Usage:
    cd backend && python scripts/export_openapi.py
"""

import json
from pathlib import Path

from app.main import app

out = Path(__file__).parent.parent / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=2) + "\n")
print(f"Wrote {out}")
