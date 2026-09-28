"""Features available at forecast time, and the flat training table.

Every feature here is derived from the forecast itself, the calendar, location, or a climatology
fitted on TRAINING years only. Observations of the valid day are never used as features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.data.align import valid_dates
from rainpp.data.schema import PRECIP_VAR

# Neighbourhood windows in grid cells (0.25°): 3 ≈ 0.75°, 7 ≈ 1.75°. They let the corrector see
# rain the model placed nearby, which matters because NWP rain is often displaced.
NEIGHBOURHOOD_SIZES = (3, 7)
CLIMATOLOGY_HALF_WINDOW_DAYS = 15


def neighbourhood_stats(precip: xr.DataArray, size: int) -> tuple[xr.DataArray, xr.DataArray]:
    """Centred box mean and max over size x size cells; edges use the cells available."""
    window = precip.rolling(lat=size, lon=size, center=True, min_periods=1)
    return window.mean(), window.max()


def forecast_features(fc: xr.Dataset) -> xr.Dataset:
    """Per-cell features from the raw (unmasked) forecast, same dims as fc.precip_mm."""
    precip = fc[PRECIP_VAR]
    feats = {"nwp_precip_mm": precip, "nwp_log1p": np.log1p(precip)}
    for size in NEIGHBOURHOOD_SIZES:
        mean, maximum = neighbourhood_stats(precip, size)
        feats[f"nwp_mean_{size}x{size}"] = mean
        feats[f"nwp_max_{size}x{size}"] = maximum
    doy = valid_dates(fc).dt.dayofyear
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return xr.Dataset(feats)


def fit_climatology(obs_train: xr.DataArray) -> xr.DataArray:
    """Mean observed rainfall per cell and day of year, smoothed over ±15 days.

    Must be given training-period observations only; using later years would leak the target.
    """
    daily = obs_train.groupby("time.dayofyear").mean("time")
    daily = daily.reindex(dayofyear=np.arange(1, 367))
    pad = CLIMATOLOGY_HALF_WINDOW_DAYS
    wrapped = xr.concat([daily.isel(dayofyear=slice(-pad, None)), daily, daily.isel(dayofyear=slice(0, pad))],
                        dim="dayofyear")
    smooth = wrapped.rolling(dayofyear=2 * pad + 1, center=True, min_periods=1).mean()
    smooth = smooth.isel(dayofyear=slice(pad, pad + 366))
    return smooth.assign_coords(dayofyear=np.arange(1, 367)).rename("obs_climatology_mm")


def build_forecast_table(fc: xr.Dataset, climatology: xr.DataArray, land_mask: xr.DataArray) -> pd.DataFrame:
    """Same columns as build_table for a forecast with no observation yet, over the observed land cells."""
    feats = forecast_features(fc)
    vd = valid_dates(fc)
    feats["obs_climatology_mm"] = climatology.sel(dayofyear=vd.dt.dayofyear).drop_vars("dayofyear")
    feats["obs_precip_mm"] = xr.full_like(fc[PRECIP_VAR], np.nan)
    feats = feats.assign_coords(valid_date=vd).where(land_mask)
    df = feats.to_dataframe().reset_index()
    df = df[df["nwp_precip_mm"].notna()].reset_index(drop=True)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def build_table(fc: xr.Dataset, paired: xr.Dataset, climatology: xr.DataArray | None = None) -> pd.DataFrame:
    """One row per (init_time, lead_day, member, lat, lon) with an observation."""
    feats = forecast_features(fc)
    vd = valid_dates(fc)
    if climatology is not None:
        feats["obs_climatology_mm"] = climatology.sel(dayofyear=vd.dt.dayofyear).drop_vars("dayofyear")
    feats["obs_precip_mm"] = paired["obs_precip_mm"]
    feats = feats.assign_coords(valid_date=vd)
    df = feats.to_dataframe().reset_index()
    df = df[df["obs_precip_mm"].notna()].reset_index(drop=True)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df
