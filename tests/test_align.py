from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainpp.data.align import pair_forecast_observation, valid_dates
from rainpp.data.schema import ProvenanceError, SchemaError
from tests.conftest import make_forecast, make_observation


def test_valid_date_is_init_date_plus_lead(forecast) -> None:
    vd = valid_dates(forecast)
    assert pd.Timestamp(vd.sel(lead_day=1).values[0]) == pd.Timestamp("2018-08-02")
    assert pd.Timestamp(vd.sel(lead_day=2).values[0]) == pd.Timestamp("2018-08-03")


def _obs_with_daily_values() -> tuple:
    """Observations 1-3 Aug where every cell equals the day of month; one cell outside 'India'."""
    time = pd.date_range("2018-08-01", periods=3)
    values = np.stack([np.full((3, 2), float(t.day)) for t in time])
    values[:, 0, 0] = np.nan
    return make_observation(values=values)


def test_pairs_each_lead_with_its_valid_day(forecast) -> None:
    paired = pair_forecast_observation(forecast, _obs_with_daily_values())
    obs = paired.obs_precip_mm.isel(init_time=0, member=0)
    assert float(obs.sel(lead_day=1).values[1, 1]) == 2.0
    assert float(obs.sel(lead_day=2).values[1, 1]) == 3.0


def test_cells_without_observation_are_masked_in_both(forecast) -> None:
    paired = pair_forecast_observation(forecast, _obs_with_daily_values())
    assert paired.nwp_precip_mm.isel(lat=0, lon=0).isnull().all()
    assert paired.nwp_precip_mm.isel(lat=1, lon=1).notnull().all()


def test_missing_observation_day_becomes_nan_and_is_counted(forecast) -> None:
    obs = _obs_with_daily_values().isel(time=slice(0, 2))  # 3 Aug missing
    paired = pair_forecast_observation(forecast, obs)
    assert paired.attrs["missing_observation_dates"] == 1
    assert paired.obs_precip_mm.sel(lead_day=2).isnull().all()


def test_real_and_synthetic_refused() -> None:
    with pytest.raises(ProvenanceError):
        pair_forecast_observation(make_forecast("real"), make_observation("synthetic"))


def test_grid_mismatch_refused(forecast) -> None:
    obs = _obs_with_daily_values().assign_coords(lat=[10.1, 10.35, 10.6])
    with pytest.raises(SchemaError, match="grids differ"):
        pair_forecast_observation(forecast, obs)
