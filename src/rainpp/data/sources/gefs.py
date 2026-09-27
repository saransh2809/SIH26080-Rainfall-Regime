"""NOAA GEFSv12 reforecast adapter (AWS bucket ``noaa-gefs-retrospective``).

Verified file facts (Phase 3 inspection, 2018-08-01 init):
  * path: GEFSv12/reforecast/{YYYY}/{YYYYMMDD}00/{member}/Days:1-10/apcp_sfc_{YYYYMMDD}00_{member}.grib2
  * each file has a text ``.idx`` listing message byte offsets, so only needed hours are downloaded
  * APCP is stored in 6-hour buckets: messages alternate "(6k)-(6k+3) hour acc" and "(6k)-(6k+6) hour acc"
  * global 0.25° grid, latitude 90 → -90, longitude 0 → 359.75 (nodes coincide with the IMD grid)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.config import Domain
from rainpp.data.schema import DATA_KIND_ATTR, PRECIP_VAR, SOURCE_ATTR, DataKind
from rainpp.data.sources.base import ForecastSource

log = logging.getLogger(__name__)

BUCKET = "noaa-gefs-retrospective"
# GRIB2 packing stores Y = (R + X * 2**E) / 10**D, so each value is quantised to 2**E / 10**D.
# Differencing two messages can therefore go negative by up to the sum of their steps; the step
# varies per message (0.01-0.2 mm seen), so the tolerance is taken from the messages themselves.
# This fallback applies only when packing steps are unknown (e.g. tests).
DEFAULT_NEGATIVE_TOLERANCE_MM = 0.15


@dataclass(frozen=True)
class IdxEntry:
    number: int
    offset: int
    variable: str
    level: str
    start_hour: int
    end_hour: int


def parse_idx(text: str) -> list[IdxEntry]:
    """Parse a GRIB .idx file restricted to accumulation messages ("a-b hour acc fcst")."""
    entries = []
    for line in text.strip().splitlines():
        parts = line.split(":")
        if len(parts) < 6 or " hour acc" not in parts[5]:
            continue
        start, end = parts[5].split(" ")[0].split("-")
        entries.append(IdxEntry(int(parts[0]), int(parts[1]), parts[3], parts[4], int(start), int(end)))
    return entries


def rain_day_window(lead_day: int, init_hour_utc: int, rain_day_end_hour_utc: int) -> tuple[int, int]:
    """Forecast-hour window of the observation rain day for a lead day.

    Lead day N covers the N-th observation day ending after the initialisation, e.g. from a
    00 UTC run with a 03 UTC rain-day end, lead 1 = hours 3→27.
    """
    if lead_day < 1:
        raise ValueError("lead_day must be >= 1")
    # End hour of the first rain day that lies entirely after the initialisation.
    first_full_end = (rain_day_end_hour_utc - init_hour_utc) % 24 + 24
    end = first_full_end + 24 * (lead_day - 1)
    return end - 24, end


def three_hour_increments(
    accumulations: dict[tuple[int, int], np.ndarray],
    max_hour: int,
    packing_steps: dict[tuple[int, int], float] | None = None,
) -> dict[int, np.ndarray]:
    """Convert bucket accumulations {(start, end): field} into {end_hour: 3-hour amount}.

    Negatives within the combined quantisation step of the two differenced messages are packing
    noise and clipped to 0; anything larger raises, because it means messages were mis-paired.
    """
    increments: dict[int, np.ndarray] = {}
    for end in range(3, max_hour + 1, 3):
        candidates = [(s, e) for (s, e) in accumulations if e == end]
        if not candidates:
            raise KeyError(f"no accumulation message ends at hour {end}")
        start, _ = max(candidates)  # the shortest window ending here
        field = accumulations[(start, end)]
        if start < end - 3:
            earlier = (start, end - 3)
            field = field - accumulations[earlier]
            if packing_steps is None:
                tolerance = DEFAULT_NEGATIVE_TOLERANCE_MM
            else:
                tolerance = packing_steps[(start, end)] + packing_steps[earlier] + 1e-6
            worst = float(np.nanmin(field))
            if worst < -tolerance:
                raise ValueError(
                    f"accumulation difference produced {worst:.3f} mm at hour {end} "
                    f"(packing tolerance {tolerance:.3f} mm)"
                )
            field = np.clip(field, 0.0, None)
        increments[end] = field
    return increments


def window_total(increments: dict[int, np.ndarray], start_hour: int, end_hour: int) -> np.ndarray:
    """Sum non-negative 3-hour increments over (start_hour, end_hour]."""
    if (end_hour - start_hour) % 3 or start_hour % 3:
        raise ValueError("window must align with 3-hour increments")
    return sum(increments[h] for h in range(start_hour + 3, end_hour + 1, 3))


def crop_global_field(values: np.ndarray, domain: Domain) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a 721x1440 north-to-south global 0.25° field to the domain; return ascending lat."""
    lats = np.linspace(90.0, -90.0, 721)
    lons = np.arange(1440) * 0.25
    lat_mask = (lats >= domain.lat_min - 1e-6) & (lats <= domain.lat_max + 1e-6)
    lon_mask = (lons >= domain.lon_min - 1e-6) & (lons <= domain.lon_max + 1e-6)
    sub = values[np.ix_(lat_mask, lon_mask)]
    return sub[::-1], lats[lat_mask][::-1], lons[lon_mask]


def packing_step(binary_scale_factor: int, decimal_scale_factor: int) -> float:
    """Quantisation step of GRIB2 simple/complex packing."""
    return 2.0**binary_scale_factor / 10.0**decimal_scale_factor


def _decode_messages(data: bytes) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float]]:
    import eccodes

    fields: dict[tuple[int, int], np.ndarray] = {}
    steps: dict[tuple[int, int], float] = {}
    offset = 0
    while offset < len(data):
        gid = eccodes.codes_new_from_message(data[offset:])
        try:
            length = eccodes.codes_get(gid, "totalLength")
            nj, ni = eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni")
            if (nj, ni) != (721, 1440):
                raise ValueError(f"unexpected grid {nj}x{ni}")
            key = (eccodes.codes_get(gid, "startStep"), eccodes.codes_get(gid, "endStep"))
            fields[key] = eccodes.codes_get_values(gid).reshape(nj, ni)
            steps[key] = packing_step(eccodes.codes_get(gid, "binaryScaleFactor"),
                                      eccodes.codes_get(gid, "decimalScaleFactor"))
        finally:
            eccodes.codes_release(gid)
        offset += length
    return fields, steps


def open_interim(path) -> xr.Dataset:
    """Open a cropped GEFS file written by scripts/download_data.py and validate it."""
    from rainpp.data.schema import validate_forecast

    ds = xr.open_dataset(path).load()
    validate_forecast(ds)
    return ds


class GEFSReforecast(ForecastSource):
    """Daily rain-day totals from the GEFSv12 reforecast, cropped to the configured domain."""

    name: ClassVar[str] = "gefs_reforecast"
    data_kind: ClassVar[DataKind] = DataKind.REAL

    def __init__(self, domain: Domain, init_hour_utc: int = 0, rain_day_end_hour_utc: int = 3,
                 max_workers: int = 8, fs=None) -> None:
        if init_hour_utc != 0:
            raise ValueError("GEFSv12 reforecasts are initialised at 00 UTC only")
        self.domain = domain
        self.init_hour_utc = init_hour_utc
        self.rain_day_end_hour_utc = rain_day_end_hour_utc
        self.max_workers = max_workers
        self._fs = fs

    @property
    def fs(self):
        if self._fs is None:
            import s3fs

            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    @staticmethod
    def file_path(init: date, member: str, variable: str = "apcp_sfc") -> str:
        stamp = f"{init:%Y%m%d}00"
        return f"{BUCKET}/GEFSv12/reforecast/{init:%Y}/{stamp}/{member}/Days:1-10/{variable}_{stamp}_{member}.grib2"

    def _fetch_accumulations(self, init: date, member: str, max_hour: int):
        path = self.file_path(init, member)
        entries = parse_idx(self.fs.cat(path + ".idx").decode())
        stop = next((e.offset for e in entries if e.end_hour > max_hour), None)
        data = self.fs.cat_file(path, start=0, end=stop)
        return _decode_messages(data)

    def _one(self, init: date, member: str, lead_days: Sequence[int]) -> np.ndarray:
        windows = [rain_day_window(ld, self.init_hour_utc, self.rain_day_end_hour_utc) for ld in lead_days]
        max_hour = max(end for _, end in windows)
        fields, steps = self._fetch_accumulations(init, member, max_hour)
        try:
            increments = three_hour_increments(fields, max_hour, steps)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"GEFS {init} {member}: {exc}") from exc
        cropped = [crop_global_field(window_total(increments, s, e), self.domain)[0] for s, e in windows]
        log.debug("GEFS %s %s done", init, member)
        return np.stack(cropped)

    def _load(self, init_dates: Sequence[date], lead_days: Sequence[int], members: Sequence[str]) -> xr.Dataset:
        _, lats, lons = crop_global_field(np.zeros((721, 1440)), self.domain)
        jobs = [(init, member) for init in init_dates for member in members]
        with ThreadPoolExecutor(self.max_workers) as pool:
            results = list(pool.map(lambda job: self._one(job[0], job[1], lead_days), jobs))
        arr = np.stack(results).reshape(len(init_dates), len(members), len(lead_days), lats.size, lons.size)
        arr = arr.transpose(0, 2, 1, 3, 4).astype(np.float32)
        return xr.Dataset(
            {PRECIP_VAR: (("init_time", "lead_day", "member", "lat", "lon"), arr,
                          {"units": "mm", "long_name": "rain-day total precipitation forecast"})},
            coords={
                "init_time": pd.to_datetime(list(init_dates)) + pd.Timedelta(hours=self.init_hour_utc),
                "lead_day": list(lead_days),
                "member": list(members),
                "lat": lats,
                "lon": lons,
            },
            attrs={
                DATA_KIND_ATTR: self.data_kind.value,
                SOURCE_ATTR: "NOAA GEFSv12 reforecast (s3://noaa-gefs-retrospective)",
                "rain_day_end_hour_utc": self.rain_day_end_hour_utc,
            },
        )
