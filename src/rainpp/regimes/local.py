"""Local rainfall regime maps (OROGRAPHIC / COASTAL / INLAND) on the 0.25° rainfall grid.

Inputs are all available at forecast time: GEFS model terrain and land mask (static) and the
forecast 850 hPa wind at mid rain-day, bilinearly interpolated from the 1° synoptic fields.
Known limitation: over terrain higher than ~1.5 km the 850 hPa level is below ground and the
model wind there is an extrapolation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from rainpp.regimes.labels import LOCAL_CLASSES, coast_distance_km, local_regime, terrain_gradient

LOCAL_CODES = {name: code for code, name in enumerate(LOCAL_CLASSES)}


def static_terrain(static: xr.Dataset, resolution_deg: float) -> xr.Dataset:
    """Distance to coast (km) and terrain gradient (m/km) on the static grid."""
    lats = static.lat.values
    coast = coast_distance_km(static.land_fraction.values, lats, resolution_deg)
    dh_dx, dh_dy = terrain_gradient(static.elevation_m.values, lats, resolution_deg)
    dims = ("lat", "lon")
    return xr.Dataset({"coast_km": (dims, coast), "dh_dx": (dims, dh_dx), "dh_dy": (dims, dh_dy),
                       "elevation_m": static.elevation_m},
                      coords={"lat": static.lat, "lon": static.lon})


def local_regime_codes(fields: xr.Dataset, terrain: xr.Dataset, cfg: dict) -> xr.DataArray:
    """(init_time, lead_day, lat, lon) int8 codes, see LOCAL_CODES."""
    wind = fields[["u850", "v850"]].interp(lat=terrain.lat, lon=terrain.lon, method="linear")
    if wind.u850.isnull().any():
        raise ValueError("synoptic fields do not cover the rainfall grid")
    codes = np.empty(wind.u850.shape, dtype=np.int8)
    for i in range(wind.sizes["init_time"]):
        for j in range(wind.sizes["lead_day"]):
            names = local_regime(terrain.coast_km.values, terrain.dh_dx.values, terrain.dh_dy.values,
                                 wind.u850.values[i, j], wind.v850.values[i, j], cfg)
            codes[i, j] = np.vectorize(LOCAL_CODES.get)(names)
    return xr.DataArray(codes, dims=("init_time", "lead_day", "lat", "lon"),
                        coords={"init_time": wind.init_time, "lead_day": wind.lead_day,
                                "lat": terrain.lat, "lon": terrain.lon}, name="local_regime")
