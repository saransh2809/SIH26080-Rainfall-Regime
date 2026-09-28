from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rainpp.api.app import create_app

client = TestClient(create_app())


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metadata_states_post_processing_and_split() -> None:
    body = client.get("/metadata").json()
    assert "post-processing" in body["system"]
    assert body["split"]["train"][1] < body["split"]["test"][0]
    assert body["thresholds_mm"]["heavy"] == 64.5


def test_regimes_expose_evidence_levels() -> None:
    body = client.get("/regimes").json()
    assert set(body["evidence_levels"]) == {"published", "derived", "heuristic"}


def test_unknown_report_phase_rejected() -> None:
    assert client.get("/verification/phase99").status_code == 422


def test_path_traversal_is_impossible() -> None:
    r = client.get("/verification/..%2F..%2Fconfig%2Fsettings")
    assert r.status_code in (404, 422)


def test_geo_layers_are_whitelisted() -> None:
    assert client.get("/geo/rivers").status_code == 422
    body = client.get("/geo/states").json()
    assert body["type"] == "FeatureCollection" and len(body["features"]) > 20


def test_district_ids_are_integers() -> None:
    response = client.get("/districts")
    if response.status_code == 404:
        pytest.skip("district weights not built")
    assert all(isinstance(d["district_id"], int) for d in response.json())


def test_model_info_lists_every_model() -> None:
    body = client.get("/model-info").json()
    assert "global_lgbm" in body and "regime_classifier" in body


def test_forecast_rejects_malformed_date_and_unknown_variable() -> None:
    assert client.get("/forecast/not-a-date").status_code == 422
    assert client.get("/forecast/2017-08-28/grid", params={"var": "secret", "lead": 1}).status_code == 422
    assert client.get("/forecast/2017-08-28/grid", params={"var": "raw_mm", "lead": 99}).status_code == 422


def test_missing_product_is_404_not_invented() -> None:
    assert client.get("/forecast/1999-01-01").status_code == 404


def test_products_endpoint_returns_list() -> None:
    assert isinstance(client.get("/products").json(), list)
