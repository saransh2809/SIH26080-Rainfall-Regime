"""GEFSv12 reforecast synoptic fields for the regime classifier (forecast-time predictors only).

For each lead day the fields are taken at the middle of the IMD rain-day window
(forecast hour 24N - 9: 15, 39, 63 h). Verified message names (2012-07-10 index):
  ugrd_pres/vgrd_pres "UGRD|VGRD:850 mb:<h> hour fcst", pres_msl "PRES:mean sea level:<h> hour fcst",
  pwat_eatm "PWAT:entire atmosphere (considered as a single layer):<h> hour fcst".
850 hPa relative vorticity is computed on the native 0.25° grid, then every field is block-averaged
to 1° over the synoptic box, which also covers the Arabian Sea low-level jet outside the rainfall grid.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.data.schema import DATA_KIND_ATTR, SOURCE_ATTR, DataKind
from rainpp.data.sources.gefs import BUCKET, ECCODES_LOCK

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6.371e6
BLOCK = 4  # 4 x 0.25° = 1°


@dataclass(frozen=True)
class FieldSpec:
    name: str
    file: str
    variable: str
    level: str
    scale: float = 1.0


FIELDS = (
    FieldSpec("u850", "ugrd_pres", "UGRD", "850 mb"),
    FieldSpec("v850", "vgrd_pres", "VGRD", "850 mb"),
    FieldSpec("mslp_hpa", "pres_msl", "PRES", "mean sea level", 0.01),
    FieldSpec("pwat_mm", "pwat_eatm", "PWAT", "entire atmosphere (considered as a single layer)"),
)


@dataclass(frozen=True)
class SynopticBox:
    lat_min: float = 0.0
    lat_max: float = 40.0
    lon_min: float = 40.0
    lon_max: float = 110.0


def mid_window_hour(lead_day: int, init_hour_utc: int = 0, rain_day_end_hour_utc: int = 3) -> int:
    from rainpp.data.sources.gefs import rain_day_window

    start, end = rain_day_window(lead_day, init_hour_utc, rain_day_end_hour_utc)
    return (start + end) // 2


def find_message(idx_text: str, variable: str, level: str, hour: int) -> tuple[int, int | None]:
    """Byte range [start, end) of the instantaneous message; end is None for the last message."""
    lines = idx_text.strip().splitlines()
    for k, line in enumerate(lines):
        parts = line.split(":")
        if parts[3] == variable and parts[4] == level and parts[5] == f"{hour} hour fcst":
            end = int(lines[k + 1].split(":")[1]) if k + 1 < len(lines) else None
            return int(parts[1]), end
    raise KeyError(f"{variable} {level} {hour}h not in index")


def relative_vorticity(u: np.ndarray, v: np.ndarray, lats: np.ndarray, dlon_deg: float) -> np.ndarray:
    """ζ = ∂v/∂x − ∂u/∂y on a regular lat-lon grid (s⁻¹); lats ascending, one row per latitude."""
    dlat = np.deg2rad(np.gradient(lats))[:, None]
    coslat = np.cos(np.deg2rad(lats))[:, None]
    dx = EARTH_RADIUS_M * coslat * np.deg2rad(dlon_deg)
    dv_dx = np.gradient(v, axis=1) / dx
    du_dy = np.gradient(u * coslat, axis=0) / (EARTH_RADIUS_M * dlat) / coslat
    return dv_dx - du_dy


def crop_box(values: np.ndarray, box: SynopticBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a 721x1440 north-to-south global field to [min, max) so it tiles into 1° blocks."""
    lats = np.linspace(90.0, -90.0, 721)
    lons = np.arange(1440) * 0.25
    lat_mask = (lats >= box.lat_min) & (lats < box.lat_max)
    lon_mask = (lons >= box.lon_min) & (lons < box.lon_max)
    sub = values[np.ix_(lat_mask, lon_mask)][::-1]
    return sub, lats[lat_mask][::-1], lons[lon_mask]


def block_mean(field: np.ndarray, block: int = BLOCK) -> np.ndarray:
    ny, nx = field.shape
    if ny % block or nx % block:
        raise ValueError(f"field {field.shape} does not tile into {block}x{block} blocks")
    return field.reshape(ny // block, block, nx // block, block).mean(axis=(1, 3))


def _decode(data: bytes) -> np.ndarray:
    import eccodes

    with ECCODES_LOCK:
        gid = eccodes.codes_new_from_message(data)
        try:
            nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
            if (nj, ni) != (721, 1440):
                raise ValueError(f"unexpected grid {nj}x{ni}")
            return eccodes.codes_get_values(gid).reshape(nj, ni)
        finally:
            eccodes.codes_release(gid)


class GEFSFields:
    """Loads (init_time, lead_day, lat, lon) synoptic fields at 1° for the classifier."""

    def __init__(self, box: SynopticBox | None = None, max_workers: int = 8, fs=None) -> None:
        self.box = box or SynopticBox()
        self.max_workers = max_workers
        self._fs = fs

    @property
    def fs(self):
        if self._fs is None:
            import s3fs

            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    @staticmethod
    def file_path(init: date, file: str, member: str = "c00") -> str:
        stamp = f"{init:%Y%m%d}00"
        return f"{BUCKET}/GEFSv12/reforecast/{init:%Y}/{stamp}/{member}/Days:1-10/{file}_{stamp}_{member}.grib2"

    def _one(self, init: date, lead_days: Sequence[int]) -> dict[str, np.ndarray]:
        hours = [mid_window_hour(ld) for ld in lead_days]
        raw: dict[str, list[np.ndarray]] = {}
        for spec in FIELDS:
            path = self.file_path(init, spec.file)
            idx = self.fs.cat(path + ".idx").decode()
            fields = []
            for hour in hours:
                start, end = find_message(idx, spec.variable, spec.level, hour)
                fields.append(crop_box(_decode(self.fs.cat_file(path, start=start, end=end)), self.box)[0] * spec.scale)
            raw[spec.name] = fields
        _, lats, _ = crop_box(np.zeros((721, 1440)), self.box)
        out = {name: np.stack([block_mean(f) for f in fs]) for name, fs in raw.items()}
        out["vort850_1e5"] = np.stack([
            block_mean(relative_vorticity(u, v, lats, 0.25)) * 1e5 for u, v in zip(raw["u850"], raw["v850"], strict=True)
        ])
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
                   SOURCE_ATTR: "NOAA GEFSv12 reforecast c00, fields at mid rain-day, 1° block means",
                   "valid_hours": str([mid_window_hour(ld) for ld in lead_days])},
        )
