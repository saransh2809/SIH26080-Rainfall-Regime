from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rainpp.api.serve import create_server


def test_dashboard_and_api_share_one_origin(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>dashboard</title>", encoding="utf-8")
    client = TestClient(create_server(tmp_path))
    page = client.get("/", params={"init": "2017-08-29", "lang": "hi"})
    assert page.status_code == 200 and "dashboard" in page.text
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/verification/phase99").status_code == 422


def test_static_files_cannot_escape_the_build_directory(tmp_path) -> None:
    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    client = TestClient(create_server(tmp_path))
    assert client.get("/..%2Fpyproject.toml").status_code == 404


def test_missing_build_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="npm run build"):
        create_server(tmp_path)
