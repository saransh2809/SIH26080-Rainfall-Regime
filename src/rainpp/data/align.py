"""Pair forecasts with the observations valid for the same rain day.

Canonical convention (verified in Phase 3): observation date D = 24 h ending 03 UTC on D, and a
00 UTC forecast's lead day N ends 03 UTC on init_date + N. Hence valid_date = init_date + lead_day.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.data.schema import (
    DATA_KIND_ATTR,
    PRECIP_VAR,
    SOURCE_ATTR,
    SchemaError,
    assert_same_kind,
    validate_forecast,
    validate_observation,
)

log = logging.getLogger(__name__)

GRID_TOLERANCE_DEG = 1e-6


def valid_dates(fc: xr.Dataset) -> xr.DataArray:
    """valid_date(init_time, lead_day) = calendar date of init_time + lead_day days."""
    init_days = fc.init_time.dt.floor("D")
    offsets = xr.DataArray(pd.to_timedelta(fc.lead_day.values, unit="D"), dims="lead_day",
                           coords={"lead_day": fc.lead_day})
    return init_days + offsets


def check_same_grid(fc: xr.Dataset, obs: xr.Dataset) -> None:
    for coord in ("lat", "lon"):
        a, b = fc[coord].values, obs[coord].values
        if a.shape != b.shape or not np.allclose(a, b, atol=GRID_TOLERANCE_DEG, rtol=0):
            raise SchemaError(f"forecast and observation '{coord}' grids differ; regrid explicitly first")


def pair_forecast_observation(fc: xr.Dataset, obs: xr.Dataset) -> xr.Dataset:
    """Return nwp_precip_mm and obs_precip_mm on dims (init_time, lead_day, member, lat, lon).

    Cells where the observation is missing (outside India, or missing days) are NaN in both
    variables so that they can never enter training or verification unnoticed.
    """
    validate_forecast(fc)
    validate_observation(obs)
    kind = assert_same_kind(fc, obs)
    check_same_grid(fc, obs)

    vd = valid_dates(fc)
    wanted = pd.DatetimeIndex(np.unique(vd.values))
    missing = wanted.difference(pd.DatetimeIndex(obs.time.values))
    if len(missing):
        log.warning("%d valid dates have no observation, e.g. %s", len(missing), missing[:3].date.tolist())

    obs_on_fc = obs[PRECIP_VAR].reindex(time=wanted).sel(time=vd).drop_vars("time")
    obs_on_fc = obs_on_fc.broadcast_like(fc[PRECIP_VAR]).transpose(*fc[PRECIP_VAR].dims)
    nwp = fc[PRECIP_VAR].where(obs_on_fc.notnull())

    return xr.Dataset(
        {"nwp_precip_mm": nwp, "obs_precip_mm": obs_on_fc},
        coords={"valid_date": vd},
        attrs={
            DATA_KIND_ATTR: kind.value,
            SOURCE_ATTR: f"forecast: {fc.attrs[SOURCE_ATTR]} | observation: {obs.attrs[SOURCE_ATTR]}",
            "missing_observation_dates": len(missing),
        },
    )
