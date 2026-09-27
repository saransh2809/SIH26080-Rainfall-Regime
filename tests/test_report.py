from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainpp.config import Domain
from rainpp.verification.report import compare, to_fields, to_markdown

DOMAIN = Domain(lat_min=10.0, lat_max=10.5, lon_min=76.0, lon_max=76.5, resolution_deg=0.25)


def _table() -> pd.DataFrame:
    """SYNTHETIC: 2 days x 3x3 cells, lead 1; 'perfect' equals obs, 'wet' adds 10 mm."""
    rows = []
    for d in pd.date_range("2016-07-01", periods=2):
        for lat in (10.0, 10.25, 10.5):
            for lon in (76.0, 76.25, 76.5):
                obs = 80.0 if (lat, lon) == (10.25, 76.25) else 1.0
                rows.append({"valid_date": d, "lead_day": 1, "lat": lat, "lon": lon, "obs_precip_mm": obs})
    t = pd.DataFrame(rows)
    t["perfect"] = t["obs_precip_mm"]
    t["wet"] = t["obs_precip_mm"] + 10.0
    return t


def test_to_fields_places_values_on_grid() -> None:
    fields = to_fields(_table(), "obs_precip_mm", DOMAIN)
    assert fields.shape == (2, 3, 3)
    assert fields[0, 1, 1] == 80.0


def test_compare_scores_models_on_identical_rows() -> None:
    res = compare(_table(), {"perfect": "perfect", "wet": "wet"}, DOMAIN, [64.5], [64.5], [1, 3])
    perfect, wet = res[1]["perfect"], res[1]["wet"]
    assert perfect["rmse"] == 0.0 and wet["bias"] == pytest.approx(10.0)
    assert perfect["categorical"]["64.5"]["csi"] == 1.0
    assert perfect["fss"]["64.5"]["1"] == 1.0
    assert perfect["n"] == wet["n"] == 18


def test_markdown_renders_undefined_as_na() -> None:
    table = _table()
    table["obs_precip_mm"] = 0.0
    table["dry"] = 0.0
    md = to_markdown(compare(table, {"dry": "dry"}, DOMAIN, [64.5], [], []), "t")
    assert "| POD ≥64.5 mm | n/a |" in md


def test_missing_column_rejected() -> None:
    with pytest.raises(KeyError):
        compare(_table(), {"x": "not_there"}, DOMAIN, [64.5], [], [])


def test_no_nan_rows_for_numpy_types() -> None:
    fields = to_fields(_table(), "wet", DOMAIN)
    assert not np.isnan(fields).any()
