"""Regime-classifier inputs: one row per (init_time, lead_day), all available at forecast time.

  * forecast synoptic fields (GEFS 850 hPa wind and vorticity, MSLP, PWAT) summarised over boxes
  * forecast rainfall over the core monsoon zone and the west coast
  * forecast core-zone rainfall anomaly, standardised by the MODEL's own training-year statistics
    per lead day and month (removes the model's mean bias before comparing with ±1)
  * observed core-zone anomaly on the days before initialisation (persistence), which IMD has
    already published when the forecast is issued
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.data.schema import PRECIP_VAR


def _sel(da: xr.DataArray, box: dict) -> xr.DataArray:
    return da.sel(lat=slice(box["lat_min"], box["lat_max"]), lon=slice(box["lon_min"], box["lon_max"]))


def box_mean(da: xr.DataArray, box: dict, mask: xr.DataArray | None = None) -> xr.DataArray:
    """cos(latitude)-weighted mean over the box, optionally only where mask is True."""
    sub = _sel(da, box)
    weights = np.cos(np.deg2rad(sub.lat)) * xr.ones_like(sub.lat * sub.lon)
    if mask is not None:
        weights = weights.where(_sel(mask, box), 0.0)
    return sub.weighted(weights).mean(("lat", "lon"))


def forecast_rain_box(fc_rain: xr.Dataset, box: dict, mask: xr.DataArray) -> xr.DataArray:
    """(init_time, lead_day) mean forecast rainfall over observed land cells in the box, control member."""
    return box_mean(fc_rain[PRECIP_VAR].isel(member=0), box, mask)


def fit_forecast_core_stats(core: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of forecast core rain per (lead_day, month). Pass TRAINING years only."""
    grouped = core.groupby(["lead_day", "month"])["fc_core_rain_mm"]
    stats = grouped.agg(["mean", "std", "size"]).rename(columns={"mean": "fc_core_mean", "std": "fc_core_sd"})
    if (stats["size"] < 30).any():
        raise ValueError("fewer than 30 samples for some lead/month; statistics would be unstable")
    return stats[["fc_core_mean", "fc_core_sd"]].reset_index()


def build_classifier_table(fields: xr.Dataset, fc_rain: xr.Dataset, land_mask: xr.DataArray, cfg: dict,
                           core_box: dict, obs_core_anomaly: pd.Series,
                           season_months: tuple[int, int] = (6, 9)) -> pd.DataFrame:
    """Raw feature table; the forecast core-anomaly column is added later by add_forecast_anomaly.

    `month` (used only to group forecast statistics) is clipped to the season, so the few forecasts
    valid just after it (e.g. 1-3 October from late-September runs) share September's statistics.
    """
    rows = {
        "fc_core_rain_mm": forecast_rain_box(fc_rain, core_box, land_mask),
        "fc_west_coast_rain_mm": forecast_rain_box(fc_rain, cfg["west_coast_box"], land_mask),
        "llj_u850": box_mean(fields["u850"], cfg["llj_box"]),
        "central_u850": box_mean(fields["u850"], cfg["central_box"]),
        "trough_vort_max": _sel(fields["vort850_1e5"], cfg["trough_box"]).max(("lat", "lon")),
        "trough_vort_mean": box_mean(fields["vort850_1e5"], cfg["trough_box"]),
        "trough_mslp_min": _sel(fields["mslp_hpa"], cfg["trough_box"]).min(("lat", "lon")),
        "mslp_south_minus_trough": box_mean(fields["mslp_hpa"], cfg["south_box"])
        - box_mean(fields["mslp_hpa"], cfg["trough_box"]),
        "core_pwat_mm": box_mean(fields["pwat_mm"], core_box),
    }
    table = xr.Dataset(rows).to_dataframe().reset_index()
    table["init_time"] = pd.to_datetime(table["init_time"]).dt.floor("D")
    table["valid_date"] = table["init_time"] + pd.to_timedelta(table["lead_day"], unit="D")
    table["month"] = table["valid_date"].dt.month.clip(*season_months)
    doy = table["valid_date"].dt.dayofyear
    table["doy_sin"], table["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    for lag in cfg["persistence_lags_days"]:
        known_day = table["init_time"] - pd.Timedelta(days=lag)
        table[f"obs_core_anom_lag{lag}"] = obs_core_anomaly.reindex(known_day).to_numpy()
    return table


def add_forecast_anomaly(table: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Standardise forecast core rain with the model's own training statistics."""
    merged = table.merge(stats, on=["lead_day", "month"], how="left", validate="many_to_one")
    if merged["fc_core_mean"].isna().any():
        raise ValueError("no training statistics for some lead_day/month")
    merged["fc_core_anom"] = (merged["fc_core_rain_mm"] - merged["fc_core_mean"]) / merged["fc_core_sd"]
    return merged.drop(columns=["fc_core_mean", "fc_core_sd"])
