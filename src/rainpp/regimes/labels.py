"""Regime labels. Every label records its evidence level (see config/regimes.yaml).

Synoptic (one per rain day, country scale), priority MONSOON_DEPRESSION > ACTIVE > BREAK > NORMAL:
  * MONSOON_DEPRESSION — IMD RSMC best track has a system of depression grade or stronger inside the
    configured box during the rain day (official IMD grading). Evidence: heuristic — the track is
    official but the box is our choice.
  * ACTIVE / BREAK — normalised core-monsoon-zone rainfall anomaly > +1 / < −1 for ≥ 3 consecutive
    days (Rajeevan et al. 2010; IITM operational description of the zone as 18–28°N, 65–88°E).
    Published for July–August; applying it in June and September is flagged "derived".
  * NORMAL — none of the above.
Local (one per grid cell per forecast), priority OROGRAPHIC > COASTAL > INLAND — heuristic rules on
model terrain, distance to coast and the FORECAST 850 hPa upslope wind.

Synoptic labels are built from observations and are used only as training targets and for scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

from rainpp.data.schema import PRECIP_VAR

SYNOPTIC_CLASSES = ("NORMAL", "ACTIVE", "BREAK", "MONSOON_DEPRESSION")
LOCAL_CLASSES = ("INLAND", "COASTAL", "OROGRAPHIC")
KM_PER_DEG_LAT = 111.2


# ---- synoptic: active / break ---------------------------------------------------------------

def core_zone_series(obs: xr.Dataset, box: dict) -> pd.Series:
    """cos(latitude)-weighted mean rainfall over the core zone's observed cells, per day."""
    sub = obs[PRECIP_VAR].sel(lat=slice(box["lat_min"], box["lat_max"]), lon=slice(box["lon_min"], box["lon_max"]))
    weights = np.cos(np.deg2rad(sub.lat)) * xr.ones_like(sub.isel(time=0))
    weights = weights.where(sub.isel(time=0).notnull())
    series = sub.weighted(weights.fillna(0)).mean(("lat", "lon"))
    return series.to_series()


def normalised_anomaly(series: pd.Series, climatology_years: tuple[int, int], half_window_days: int = 15) -> pd.Series:
    """(x − mean_doy) / sd_doy with day-of-year mean and SD from the base years, ±15-day smoothed."""
    base = series[(series.index.year >= climatology_years[0]) & (series.index.year <= climatology_years[1])]
    if base.index.year.nunique() < 10:
        raise ValueError("climatology needs at least 10 base years")
    by_doy = base.groupby(base.index.dayofyear)
    mean, sd = (_circular_smooth(s.reindex(range(1, 367)), half_window_days) for s in (by_doy.mean(), by_doy.std()))
    doy = series.index.dayofyear
    return pd.Series((series.to_numpy() - mean[doy].to_numpy()) / sd[doy].to_numpy(), index=series.index)


def _circular_smooth(s: pd.Series, half: int) -> pd.Series:
    values = s.to_numpy(dtype=float)
    padded = np.r_[values[-half:], values, values[:half]]
    smoothed = pd.Series(padded).rolling(2 * half + 1, center=True, min_periods=1).mean().to_numpy()[half:-half]
    return pd.Series(smoothed, index=s.index)


def spell_mask(condition: pd.Series, min_days: int) -> pd.Series:
    """True on days belonging to a run of at least min_days consecutive True values (daily index)."""
    if not condition.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("spell detection needs a continuous daily series")
    run_id = (condition != condition.shift()).cumsum()
    run_len = condition.groupby(run_id).transform("size")
    return condition & (run_len >= min_days)


# ---- synoptic: depressions ------------------------------------------------------------------

def rain_day_of(times: pd.Series, rain_day_end_hour_utc: int = 3) -> pd.Series:
    """IMD rain-day label for UTC timestamps: the date on which the 24 h window ends."""
    t = pd.to_datetime(times)
    return (t - pd.Timedelta(hours=rain_day_end_hour_utc)).dt.floor("D") + pd.Timedelta(days=1)


def depression_days(track: pd.DataFrame, box: dict) -> set[pd.Timestamp]:
    """Rain days with an IMD best-track fix of depression grade or stronger inside the box."""
    graded = track[track["grade"].notna()]
    inside = graded["lat"].between(box["lat_min"], box["lat_max"]) & graded["lon"].between(box["lon_min"], box["lon_max"])
    return set(rain_day_of(graded.loc[inside, "time"]))


# ---- synoptic: combined ---------------------------------------------------------------------

def synoptic_labels(core_series: pd.Series, depressions: set[pd.Timestamp], cfg: dict,
                    months: list[int]) -> pd.DataFrame:
    """Daily labels for the given months; core_series must be continuous daily and span the base years."""
    ab = cfg["active_break"]
    anom = normalised_anomaly(core_series, tuple(ab["climatology_years"]))
    active = spell_mask(anom > ab["anomaly_threshold"], ab["min_consecutive_days"])
    brk = spell_mask(anom < -ab["anomaly_threshold"], ab["min_consecutive_days"])

    df = pd.DataFrame({"core_anomaly": anom, "active": active, "break": brk})
    df = df[df.index.month.isin(months)]
    df["depression"] = df.index.isin(list(depressions))
    df["synoptic_regime"] = np.select(
        [df["depression"], df["active"], df["break"]], ["MONSOON_DEPRESSION", "ACTIVE", "BREAK"], "NORMAL")
    original = df.index.month.isin(ab["original_months"])
    df["evidence"] = np.where(df["synoptic_regime"] == "MONSOON_DEPRESSION", "heuristic",
                              np.where(original, "published", "derived"))
    df.index.name = "valid_date"
    return df[["synoptic_regime", "evidence", "core_anomaly"]]


# ---- local ----------------------------------------------------------------------------------

def coast_distance_km(land_fraction: np.ndarray, lats: np.ndarray, resolution_deg: float) -> np.ndarray:
    """Distance from each land cell to the nearest sea cell (0 over sea); grid-scale approximation."""
    land = land_fraction > 0.5
    mean_lat = float(np.mean(lats))
    sampling = (resolution_deg * KM_PER_DEG_LAT, resolution_deg * KM_PER_DEG_LAT * np.cos(np.deg2rad(mean_lat)))
    return distance_transform_edt(land, sampling=sampling)


def terrain_gradient(elevation_m: np.ndarray, lats: np.ndarray, resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(dh/dx, dh/dy) in metres per km, eastward and northward."""
    dy_km = resolution_deg * KM_PER_DEG_LAT
    dx_km = dy_km * np.cos(np.deg2rad(lats))[:, None]
    dh_dy, dh_dx = np.gradient(elevation_m, dy_km, axis=0), np.gradient(elevation_m, axis=1) / dx_km
    return dh_dx, dh_dy


def local_regime(coast_km: np.ndarray, dh_dx: np.ndarray, dh_dy: np.ndarray, u850: np.ndarray,
                 v850: np.ndarray, cfg: dict) -> np.ndarray:
    """OROGRAPHIC / COASTAL / INLAND per cell from terrain and the forecast 850 hPa wind.

    Orographic when the terrain-forced ascent w = V·∇h (m/s; gradient in m/km, hence /1000)
    reaches the configured threshold — one physical quantity instead of separate slope and wind cuts.
    """
    forced_ascent = (u850 * dh_dx + v850 * dh_dy) / 1000.0
    orographic = forced_ascent >= cfg["orographic"]["min_forced_ascent_ms"]
    coastal = coast_km <= cfg["coastal"]["max_distance_to_coast_km"]
    return np.select([orographic, coastal], ["OROGRAPHIC", "COASTAL"], "INLAND")
