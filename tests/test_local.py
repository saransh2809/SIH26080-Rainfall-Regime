"""Local regime map tests on a SYNTHETIC coastal ridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.regimes.local import LOCAL_CODES, local_regime_codes, static_terrain

CFG = {"coastal": {"max_distance_to_coast_km": 60},
       "orographic": {"min_forced_ascent_ms": 0.05}}


def _static() -> xr.Dataset:
    lat, lon = np.arange(10.0, 12.01, 0.25), np.arange(70.0, 74.01, 0.25)
    land = np.ones((lat.size, lon.size))
    land[:, :2] = 0.0                                                   # sea on the west
    elevation = np.clip((lon - 72.0) / 0.25, 0, None)[None, :] * 400.0 * np.ones((lat.size, 1))  # ridge rises east of 72E
    return xr.Dataset({"land_fraction": (("lat", "lon"), land), "elevation_m": (("lat", "lon"), elevation)},
                      coords={"lat": lat, "lon": lon})


def _fields(u: float) -> xr.Dataset:
    lat, lon = np.arange(9.5, 13.0, 1.0), np.arange(69.5, 75.0, 1.0)
    coords = {"init_time": pd.to_datetime(["2012-07-10"]), "lead_day": [1], "lat": lat, "lon": lon}
    shape = (1, 1, lat.size, lon.size)
    return xr.Dataset({"u850": (tuple(coords), np.full(shape, u)), "v850": (tuple(coords), np.zeros(shape))},
                      coords=coords)


def test_westerly_onto_ridge_is_orographic_and_coast_is_coastal() -> None:
    terrain = static_terrain(_static(), 0.25)
    codes = local_regime_codes(_fields(10.0), terrain, CFG).isel(init_time=0, lead_day=0)
    assert int(codes.sel(lat=11.0, lon=73.0)) == LOCAL_CODES["OROGRAPHIC"]
    assert int(codes.sel(lat=11.0, lon=70.5)) == LOCAL_CODES["COASTAL"]


def test_easterly_is_downslope_so_not_orographic() -> None:
    terrain = static_terrain(_static(), 0.25)
    codes = local_regime_codes(_fields(-10.0), terrain, CFG)
    assert not (codes == LOCAL_CODES["OROGRAPHIC"]).any()
