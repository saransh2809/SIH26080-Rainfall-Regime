"""FastAPI backend. Serves configuration, model metadata, validation reports and (Phase 9b)
precomputed forecast products. Clients can never pass file paths: every file is resolved from a
fixed directory plus a validated identifier.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rainpp import __version__
from rainpp.config import PROJECT_ROOT, Settings, load_settings, load_yaml

REPORT_PHASES = Literal["phase4", "phase5", "phase6", "phase6b", "phase7"]
MODEL_NAMES = ("quantile_mapping", "global_lgbm", "regime_classifier", "regime_bc/C1", "regime_bc/B2s")


@lru_cache
def settings() -> Settings:
    return load_settings()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not available — has the corresponding phase been run?")
    return json.loads(path.read_text(encoding="utf-8"))


BOUNDARY_FILES = {"states": "India-States.shp", "districts": "India-Districts-2011Census.shp"}
SIMPLIFY_DEG = 0.01  # ~1 km; keeps the GeoJSON small enough for the browser


@lru_cache
def _boundaries(layer: str) -> dict:
    """Simplified GeoJSON of Survey-of-India-based boundaries (DataMeet), cached per process."""
    import geopandas as gpd

    from rainpp.spatial.districts import load_districts

    path = settings().paths.data_dir / "raw" / "boundaries" / BOUNDARY_FILES[layer]
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{layer} boundaries not downloaded")
    gdf = load_districts(path) if layer == "districts" else gpd.read_file(path).to_crs("EPSG:4326")
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True)
    return json.loads(gdf.to_json())


def create_app() -> FastAPI:
    s = settings()
    app = FastAPI(title="Regime-Aware Rainfall Post-Processing API (SIH 26080)", version=__version__,
                  description="Post-processing of NWP rainfall forecasts; not a weather model.")
    app.add_middleware(CORSMiddleware, allow_origins=s.api.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "mode": s.mode}

    @app.get("/metadata")
    def metadata() -> dict:
        data_meta = s.paths.data_dir / "processed" / "dataset_metadata.json"
        return {
            "mode": s.mode,
            "data_kind": "real",
            "system": "post-processing of GEFSv12 reforecast rainfall; does not forecast weather from scratch",
            "domain": s.domain.model_dump(),
            "season_months": s.season.months,
            "lead_days": s.forecast.lead_days,
            "split": s.split.model_dump(),
            "dataset": json.loads(data_meta.read_text(encoding="utf-8")) if data_meta.is_file() else None,
            "thresholds_mm": load_yaml("thresholds.yaml")["thresholds_mm"],
        }

    @app.get("/regimes")
    def regimes() -> dict:
        cfg = load_yaml("regimes.yaml")
        return {"synoptic": cfg["synoptic"], "local": cfg["local"],
                "evidence_levels": {"published": "from peer-reviewed or operational literature",
                                    "derived": "published criterion applied outside its original scope",
                                    "heuristic": "project rule; parameters need sensitivity testing"}}

    @app.get("/districts")
    def districts() -> list[dict]:
        import pandas as pd

        path = s.paths.data_dir / "interim" / "static" / "district_weights.districts.parquet"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="district weights not built")
        table = pd.read_parquet(path)
        table["low_coverage"] = table["coverage_fraction"] < 0.5
        return table.to_dict(orient="records")

    @app.get("/geo/{layer}")
    def geo(layer: Literal["states", "districts"]) -> dict:
        return _boundaries(layer)

    @app.get("/model-info")
    def model_info() -> dict:
        out = {}
        for name in MODEL_NAMES:
            path = s.paths.model_dir / name / "model_metadata.json"
            out[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else "not trained"
        return out

    @app.get("/verification/{phase}")
    def verification(phase: REPORT_PHASES) -> dict:
        return _read_json(PROJECT_ROOT / "reports" / f"{phase}_validation.json")

    return app


app = create_app()
