"""NOAA GEFSv12 operational forecasts (AWS bucket ``noaa-gefs-pds``) for live products.

The models were trained on the GEFSv12 *reforecast*. The operational runs use the same model version
(GEFSv12, operational since 2020-09-23) but a different analysis for initial conditions, so their
biases can differ; live products say so.

Verified file facts (2026-09-28 00 UTC, control member ``gec00``):
  * one GRIB2 file per forecast hour, each with a text ``.idx``:
      atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.fHHH  (0.25°): APCP, PWAT, PRMSL
      atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.fHHH   (0.5°):  UGRD/VGRD 850 mb
      atmos/pgrb2bp5/gec00.t00z.pgrb2b.0p50.fHHH   (0.5°):  "PRES:mean sea level"
  * APCP buckets match the reforecast: fHHH holds "(6k)-(HHH) hour acc" with 6k the last multiple of 6
  * sea-level pressure: the reforecast stores "PRES:mean sea level"; the operational PRMSL differs from it
    by up to 9 hPa locally, so the identically named PRES message is used
  * 850 hPa winds and PRES exist only at 0.5°; they are bilinearly interpolated to the 0.25° grid and then
    processed exactly like the reforecast fields (see scripts/check_operational_fields.py for the effect)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.config import Domain
from rainpp.data.schema import DATA_KIND_ATTR, PRECIP_VAR, SOURCE_ATTR, DataKind
from rainpp.data.sources.base import ForecastSource
from rainpp.data.sources.gefs import (
    ECCODES_LOCK,
    crop_global_field,
    packing_step,
    rain_day_window,
    three_hour_increments,
    window_total,
)
from rainpp.data.sources.gefs_fields import (
    SynopticBox,
    block_mean,
    crop_box,
    find_message,
    mid_window_hour,
    relative_vorticity,
)

log = logging.getLogger(__name__)

BUCKET = "noaa-gefs-pds"
FIRST_V12_INIT = date(2020, 9, 23)
MEMBER = "c00"
PRODUCTS = {
    "s": ("pgrb2sp25", "pgrb2s.0p25"),
    "a": ("pgrb2ap5", "pgrb2a.0p50"),
    "b": ("pgrb2bp5", "pgrb2b.0p50"),
}


def file_path(init: date, product: str, hour: int, member: str = MEMBER) -> str:
    folder, stem = PRODUCTS[product]
    return f"{BUCKET}/gefs.{init:%Y%m%d}/00/atmos/{folder}/ge{member}.t00z.{stem}.f{hour:03d}"


def upsample_half_degree(values: np.ndarray) -> np.ndarray:
    """Bilinear 0.5° → 0.25° on the global grid (361x720 → 721x1440); longitude wraps around."""
    if values.shape != (361, 720):
        raise ValueError(f"expected a 361x720 field, got {values.shape}")
    wrapped = np.concatenate([values, values[:, :1]], axis=1)          # 360° column = 0° column
    rows = np.empty((721, 721))
    rows[0::2] = wrapped
    rows[1::2] = 0.5 * (wrapped[:-1] + wrapped[1:])
    out = np.empty((721, 1441))
    out[:, 0::2] = rows
    out[:, 1::2] = 0.5 * (rows[:, :-1] + rows[:, 1:])
    return out[:, :1440]


class _Fetcher:
    def __init__(self, fs=None) -> None:
        self._fs = fs

    @property
    def fs(self):
        if self._fs is None:
            import s3fs

            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    def message(self, init: date, product: str, hour: int, variable: str, level: str,
                label: str | None = None) -> tuple[np.ndarray, float]:
        """Decode one message; returns (values, packing step). `label` matches the idx time text exactly."""
        import eccodes

        path = file_path(init, product, hour)
        idx = self.fs.cat(path + ".idx").decode()
        if label is None:
            start, end = find_message(idx, variable, level, hour)
        else:
            start, end = _find_labelled(idx, variable, level, label)
        data = self.fs.cat_file(path, start=start, end=end)
        with ECCODES_LOCK:
            gid = eccodes.codes_new_from_message(data)
            try:
                shape = (eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni"))
                values = eccodes.codes_get_values(gid).reshape(shape)
                step = packing_step(eccodes.codes_get(gid, "binaryScaleFactor"),
                                    eccodes.codes_get(gid, "decimalScaleFactor"))
            finally:
                eccodes.codes_release(gid)
        return values, step


def _find_labelled(idx_text: str, variable: str, level: str, label: str) -> tuple[int, int | None]:
    lines = idx_text.strip().splitlines()
    for k, line in enumerate(lines):
        parts = line.split(":")
        if parts[3] == variable and parts[4] == level and parts[5] == label:
            end = int(lines[k + 1].split(":")[1]) if k + 1 < len(lines) else None
            return int(parts[1]), end
    raise KeyError(f"{variable} {level} '{label}' not in index")


def is_available(init: date, max_hour: int = 75, fs=None) -> bool:
    """True when the run has been published up to max_hour (files appear hour by hour)."""
    fetcher = _Fetcher(fs)
    return fetcher.fs.exists(file_path(init, "a", max_hour) + ".idx") and \
        fetcher.fs.exists(file_path(init, "s", max_hour) + ".idx")


def latest_available(today: date, max_hour: int = 75, lookback_days: int = 5, fs=None) -> date:
    for back in range(lookback_days + 1):
        init = today - timedelta(days=back)
        if is_available(init, max_hour, fs):
            return init
    raise LookupError(f"no complete GEFS 00 UTC run in the last {lookback_days} days")


class GEFSOperational(ForecastSource):
    """Rain-day totals from operational GEFSv12 (control member), cropped to the domain."""

    name: ClassVar[str] = "gefs_operational"
    data_kind: ClassVar[DataKind] = DataKind.REAL

    def __init__(self, domain: Domain, rain_day_end_hour_utc: int = 3, max_workers: int = 8, fs=None) -> None:
        self.domain = domain
        self.rain_day_end_hour_utc = rain_day_end_hour_utc
        self.max_workers = max_workers
        self.fetch = _Fetcher(fs)

    def _one(self, init: date, lead_days: Sequence[int]) -> np.ndarray:
        if init < FIRST_V12_INIT:
            raise ValueError(f"operational GEFSv12 starts {FIRST_V12_INIT}; use the reforecast before that")
        windows = [rain_day_window(ld, 0, self.rain_day_end_hour_utc) for ld in lead_days]
        max_hour = max(end for _, end in windows)
        keys = [(hour - 3 if hour % 6 == 3 else hour - 6, hour) for hour in range(3, max_hour + 1, 3)]
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda k: self.fetch.message(init, "s", k[1], "APCP", "surface",
                                                             f"{k[0]}-{k[1]} hour acc fcst"), keys))
        fields = {k: g[0] for k, g in zip(keys, got, strict=True)}
        steps = {k: g[1] for k, g in zip(keys, got, strict=True)}
        try:
            increments = three_hour_increments(fields, max_hour, steps)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"GEFS operational {init}: {exc}") from exc
        return np.stack([crop_global_field(window_total(increments, s, e), self.domain)[0] for s, e in windows])

    def _load(self, init_dates: Sequence[date], lead_days: Sequence[int], members: Sequence[str]) -> xr.Dataset:
        if list(members) != [MEMBER]:
            raise ValueError("only the control member c00 is supported")
        _, lats, lons = crop_global_field(np.zeros((721, 1440)), self.domain)
        with ThreadPoolExecutor(self.max_workers) as pool:
            results = list(pool.map(lambda d: self._one(d, lead_days), init_dates))
        arr = np.stack(results)[:, :, None].astype(np.float32)  # (init, lead, member, lat, lon)
        return xr.Dataset(
            {PRECIP_VAR: (("init_time", "lead_day", "member", "lat", "lon"), arr,
                          {"units": "mm", "long_name": "rain-day total precipitation forecast"})},
            coords={"init_time": pd.to_datetime(list(init_dates)), "lead_day": list(lead_days),
                    "member": [MEMBER], "lat": lats, "lon": lons},
            attrs={DATA_KIND_ATTR: self.data_kind.value,
                   SOURCE_ATTR: "NOAA GEFSv12 operational (s3://noaa-gefs-pds), control member",
                   "rain_day_end_hour_utc": self.rain_day_end_hour_utc},
        )


class GEFSOperationalFields:
    """Same output as GEFSFields (1° synoptic fields at mid rain-day) from the operational runs."""

    def __init__(self, box: SynopticBox | None = None, max_workers: int = 8, fs=None) -> None:
        self.box = box or SynopticBox()
        self.max_workers = max_workers
        self.fetch = _Fetcher(fs)

    def _global(self, init: date, hour: int) -> dict[str, np.ndarray]:
        get = self.fetch.message
        return {
            "u850": upsample_half_degree(get(init, "a", hour, "UGRD", "850 mb")[0]),
            "v850": upsample_half_degree(get(init, "a", hour, "VGRD", "850 mb")[0]),
            "mslp_hpa": upsample_half_degree(get(init, "b", hour, "PRES", "mean sea level")[0]) * 0.01,
            "pwat_mm": get(init, "s", hour, "PWAT", "entire atmosphere (considered as a single layer)")[0],
        }

    def _one(self, init: date, lead_days: Sequence[int]) -> dict[str, np.ndarray]:
        _, lats, _ = crop_box(np.zeros((721, 1440)), self.box)
        with ThreadPoolExecutor(len(lead_days)) as pool:
            globals_ = list(pool.map(lambda ld: self._global(init, mid_window_hour(ld)), lead_days))
        per_lead = [{k: crop_box(v, self.box)[0] for k, v in g.items()} for g in globals_]
        out = {name: np.stack([block_mean(p[name]) for p in per_lead]) for name in ("u850", "v850", "mslp_hpa", "pwat_mm")}
        out["vort850_1e5"] = np.stack([block_mean(relative_vorticity(p["u850"], p["v850"], lats, 0.25)) * 1e5
                                       for p in per_lead])
        return out

    def load(self, init_dates: Sequence[date], lead_days: Sequence[int]) -> xr.Dataset:
        with ThreadPoolExecutor(self.max_workers) as pool:
            results = list(pool.map(lambda d: self._one(d, lead_days), init_dates))
        _, lats, lons = crop_box(np.zeros((721, 1440)), self.box)
        lat1, lon1 = block_mean(lats[:, None].repeat(4, 1))[:, 0], block_mean(lons[None, :].repeat(4, 0))[0]
        data = {name: (("init_time", "lead_day", "lat", "lon"), np.stack([r[name] for r in results]).astype(np.float32))
                for name in results[0]}
        return xr.Dataset(
            data,
            coords={"init_time": pd.to_datetime(list(init_dates)), "lead_day": list(lead_days), "lat": lat1, "lon": lon1},
            attrs={DATA_KIND_ATTR: DataKind.REAL.value,
                   SOURCE_ATTR: "NOAA GEFSv12 operational c00; 850 hPa wind and MSL pressure interpolated 0.5°→0.25°, "
                                "then 1° block means as for the reforecast",
                   "valid_hours": str([mid_window_hour(ld) for ld in lead_days])},
        )
