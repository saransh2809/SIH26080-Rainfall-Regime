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


def test_district_csv_is_sorted_and_labelled() -> None:
    products = client.get("/products").json()
    if not products:
        pytest.skip("no forecast products built")
    r = client.get(f"/forecast/{products[0]}/districts.csv", params={"lead": 1})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    lines = r.text.splitlines()
    header = lines[0].split(",")
    assert "observed_imd_mm_verification_only" in header and "product_source" in header
    p = header.index("p_heavy_ge_64_5mm_max_cell")
    values = [float(v.split(",")[p]) for v in lines[1:6] if v.split(",")[p]]
    assert values == sorted(values, reverse=True)


def test_district_csv_rejects_bad_lead() -> None:
    assert client.get("/forecast/2017-08-28/districts.csv", params={"lead": 99}).status_code == 422


def test_events_only_list_built_products() -> None:
    built = set(client.get("/products").json())
    events = client.get("/events").json()
    assert all(e["init"] in built for e in events)
    assert all({"label", "district", "observed_max_mm", "period"} <= set(e) for e in events)


def test_products_endpoint_returns_list() -> None:
    assert isinstance(client.get("/products").json(), list)
