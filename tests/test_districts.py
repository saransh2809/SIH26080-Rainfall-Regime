"""District aggregation tests with SYNTHETIC square districts on a 0.25° grid."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely import box

from rainpp.spatial.districts import DistrictWeights, build_weights


def _setup():
    lats, lons = np.meshgrid([10.0, 10.25], [76.0, 76.25], indexing="ij")
    lats, lons = lats.ravel(), lons.ravel()          # 4 cells: (10,76) (10,76.25) (10.25,76) (10.25,76.25)
    districts = gpd.GeoDataFrame(
        {"district_id": [1, 2, 3], "district_name": ["West", "Whole", "Offshore"], "state_name": ["S"] * 3},
        geometry=[box(75.875, 9.875, 76.125, 10.375),   # exactly the western column of cells
                  box(75.875, 9.875, 76.375, 10.375),   # all four cells
                  box(80.0, 15.0, 80.5, 15.5)],         # no cells
        crs="EPSG:4326")
    return build_weights(districts, lats, lons, 0.25)


def test_weights_normalised_and_coverage_reported() -> None:
    w = _setup()
    np.testing.assert_allclose(np.asarray(w.weights.sum(axis=1)).ravel(), [1.0, 1.0, 0.0])
    assert w.districts.coverage_fraction.tolist()[:2] == pytest.approx([1.0, 1.0], abs=1e-6)
    assert w.districts.coverage_fraction.iloc[2] == 0.0


def test_mean_uses_only_overlapping_cells() -> None:
    w = _setup()
    values = np.array([10.0, 30.0, 20.0, 40.0])   # western column = 10 and 20
    out = w.aggregate_mean(values)
    assert out[0] == pytest.approx(15.0, rel=1e-3)
    assert out[1] == pytest.approx(25.0, rel=1e-3)
    assert math.isnan(out[2])                      # uncovered district: NaN, not a borrowed value


def test_max_and_batched_shapes() -> None:
    w = _setup()
    batch = np.array([[10.0, 30.0, 20.0, 40.0], [0.0, 0.0, 5.0, 0.0]])
    assert w.aggregate_mean(batch).shape == (2, 3)
    mx = w.aggregate_max(batch)
    assert mx[0, 0] == 20.0 and mx[0, 1] == 40.0 and mx[1, 0] == 5.0
    assert math.isnan(mx[0, 2])


def test_nan_cells_rejected() -> None:
    with pytest.raises(ValueError, match="NaN"):
        _setup().aggregate_mean(np.array([1.0, np.nan, 1.0, 1.0]))


def test_save_load_round_trip(tmp_path: Path) -> None:
    w = _setup()
    w.save(tmp_path / "dw")
    loaded = DistrictWeights.load(tmp_path / "dw")
    values = np.array([10.0, 30.0, 20.0, 40.0])
    np.testing.assert_allclose(loaded.aggregate_mean(values)[:2], w.aggregate_mean(values)[:2])
