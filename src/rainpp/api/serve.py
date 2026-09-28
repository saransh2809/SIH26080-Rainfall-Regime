"""Single-origin server for deployment: the built dashboard at /, the API at /api.

These are the same paths the Vite dev server proxies, so the frontend needs no configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rainpp.api.app import create_app
from rainpp.config import PROJECT_ROOT


def create_server(static_dir: Path) -> FastAPI:
    if not (static_dir / "index.html").is_file():
        raise FileNotFoundError(f"no built dashboard in {static_dir}; run `npm run build` in frontend/")
    server = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    server.mount("/api", create_app())
    server.mount("/", StaticFiles(directory=static_dir, html=True), name="dashboard")
    return server


def __getattr__(name: str):
    # Built lazily so importing this module (e.g. in tests) does not require a built frontend.
    if name == "app":
        return create_server(Path(os.environ.get("RAINPP_STATIC_DIR", PROJECT_ROOT / "frontend" / "dist")))
    raise AttributeError(name)
