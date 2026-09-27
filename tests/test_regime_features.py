"""Classifier feature tests on tiny SYNTHETIC fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rainpp.regimes.features import (
    add_forecast_anomaly,
    box_mean,
    build_classifier_table,
    fit_forecast_core_stats,
)

BOX = {"lat_min": 10.0, "lat_max": 11.0, "lon_min": 70.0, "lon_max": 71.0}
CFG = {k: BOX for k in ("llj_box", "central_box", "trough_box", "south_box", "west_coast_box")}
CFG["persistence_lags_days"] = [1, 2]


def _grid(values: np.ndarray, extra: dict | None = None) -> xr.DataArray:
    lat, lon = np.array([10.0, 11.0]), np.array([70.0, 71.0])
    dims = (*(extra or {}), "lat", "lon")
    return xr.DataArray(values, dims=dims, coords={**(extra or {}), "lat": lat, "lon": lon})


def test_box_mean_respects_mask() -> None:
    field = _grid(np.array([[1.0, 100.0], [1.0, 1.0]]))
    mask = _grid(np.array([[True, False], [True, True]]))
    assert float(box_mean(field, BOX, mask)) == pytest.approx(1.0)


def test_table_rows_and_persistence_use_only_past_observations() -> None:
    init = pd.to_datetime(["2012-07-10", "2012-07-11"])
    coords = {"init_time": init, "lead_day": [1, 2]}
    shape = (2, 2, 2, 2)
    fields = xr.Dataset({name: _grid(np.full(shape, 5.0), coords)
                         for name in ("u850", "vort850_1e5", "mslp_hpa", "pwat_mm")})
    rain = _grid(np.full((2, 2, 1, 2, 2), 10.0), {**coords, "member": ["c00"]}).to_dataset(name="precip_mm")
    mask = _grid(np.ones((2, 2), dtype=bool))
    obs_anom = pd.Series([0.5, 1.5, 9.9], index=pd.to_datetime(["2012-07-08", "2012-07-09", "2012-07-10"]))

    table = build_classifier_table(fields, rain, mask, CFG, BOX, obs_anom)
    assert len(table) == 4
    row = table[(table.init_time == "2012-07-10") & (table.lead_day == 2)].iloc[0]
    assert row.valid_date == pd.Timestamp("2012-07-12")
    assert row.obs_core_anom_lag1 == 1.5 and row.obs_core_anom_lag2 == 0.5  # never the 9.9 of D0 itself
    assert row.fc_core_rain_mm == pytest.approx(10.0)


def test_forecast_anomaly_uses_supplied_training_stats() -> None:
    table = pd.DataFrame({"lead_day": [1, 1], "month": [7, 7], "fc_core_rain_mm": [10.0, 16.0]})
    stats = pd.DataFrame({"lead_day": [1], "month": [7], "fc_core_mean": [10.0], "fc_core_sd": [3.0]})
    assert add_forecast_anomaly(table, stats)["fc_core_anom"].tolist() == [0.0, 2.0]


def test_stats_refuse_tiny_samples() -> None:
    core = pd.DataFrame({"lead_day": [1] * 5, "month": [7] * 5, "fc_core_rain_mm": np.arange(5.0)})
    with pytest.raises(ValueError, match="fewer than 30"):
        fit_forecast_core_stats(core)
