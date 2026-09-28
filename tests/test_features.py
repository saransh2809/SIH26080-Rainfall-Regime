from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rainpp.data.align import pair_forecast_observation
from rainpp.data.features import (
    build_forecast_table,
    build_table,
    fit_climatology,
    forecast_features,
    neighbourhood_stats,
)
from rainpp.data.schema import PRECIP_VAR
from tests.conftest import make_forecast, make_observation


def test_neighbourhood_mean_and_max_see_nearby_rain() -> None:
    field = xr.DataArray(np.zeros((5, 5)), dims=("lat", "lon"))
    field[2, 2] = 9.0
    mean, maximum = neighbourhood_stats(field, 3)
    assert float(mean[1, 1]) == pytest.approx(1.0)  # 9 / 9 cells
    assert float(maximum[1, 1]) == 9.0
    assert float(maximum[0, 0]) == 0.0  # 2 cells away: outside the 3x3 box


def test_forecast_features_keep_forecast_dims(forecast) -> None:
    feats = forecast_features(forecast)
    assert feats["nwp_mean_3x3"].dims == forecast["precip_mm"].dims
    assert float(feats["nwp_log1p"].max()) == pytest.approx(np.log1p(3.0))


def test_climatology_is_smoothed_and_covers_every_day() -> None:
    time = pd.date_range("2010-01-01", "2011-12-31")
    values = np.zeros((time.size, 1, 1))
    values[time.dayofyear == 200] = 31.0  # a single wet day of year
    obs = xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time})
    clim = fit_climatology(obs)
    assert clim.sizes["dayofyear"] == 366
    assert clim.sel(dayofyear=200).item() == pytest.approx(1.0)  # 31 spread over a 31-day window
    assert clim.sel(dayofyear=250).item() == 0.0


def test_climatology_wraps_across_new_year() -> None:
    time = pd.date_range("2010-01-01", "2010-12-31")
    values = np.zeros((time.size, 1, 1))
    values[time.dayofyear == 1] = 31.0
    clim = fit_climatology(xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time}))
    assert clim.sel(dayofyear=360).item() > 0.0


def test_table_has_one_row_per_observed_cell() -> None:
    fc = make_forecast()
    values = np.full((3, 3, 2), 4.0)
    values[:, 0, 0] = np.nan  # one cell outside India
    obs = make_observation(values=values)
    table = build_table(fc, pair_forecast_observation(fc, obs))
    assert len(table) == 2 * (3 * 2 - 1)  # 2 leads x 5 observed cells
    assert table["obs_precip_mm"].notna().all()
    assert {"valid_date", "lead_day", "nwp_mean_7x7", "doy_sin"} <= set(table.columns)


def test_forecast_table_matches_training_columns_without_observations() -> None:
    fc = make_forecast()
    values = np.full((3, 3, 2), 4.0)
    values[:, 0, 0] = np.nan
    obs = make_observation(values=values)
    climatology = fit_climatology(obs[PRECIP_VAR])
    live = build_forecast_table(fc, climatology, obs[PRECIP_VAR].isel(time=0).notnull().drop_vars("time"))
    trained = build_table(fc, pair_forecast_observation(fc, obs), climatology)
    assert set(live.columns) == set(trained.columns)
    assert len(live) == len(trained)
    assert live["obs_precip_mm"].isna().all()
    np.testing.assert_allclose(live["obs_climatology_mm"], trained["obs_climatology_mm"])
