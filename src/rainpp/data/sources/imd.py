"""IMD 0.25° gridded daily rainfall adapter (Pai et al. 2014, MAUSAM).

Verified file facts (Phase 3, 2018 file):
  * yearly NetCDF from POST https://imdpune.gov.in/cmpg/Griddata/RF25.php with form field RF25=<year>
  * variable RAINFALL (mm) on TIME, LATITUDE (6.5→38.5), LONGITUDE (66.5→100.0); non-India cells are NaN
  * TIME label D = the 24 h window ENDING 03 UTC on D. Established empirically: GEFS rainfall for
    03 UTC D → 03 UTC D+1 correlates best with IMD date D+1 (mean spatial r 0.43 vs 0.34 for D,
    30/40 JJAS-2018 cases). This matches the canonical valid-date convention used by forecasts.
  * the server drops connections intermittently, so downloads retry with backoff
"""

from __future__ import annotations

import logging
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import ClassVar

import xarray as xr

from rainpp.data.schema import DATA_KIND_ATTR, PRECIP_VAR, SOURCE_ATTR, DataKind
from rainpp.data.sources.base import ObservationSource

log = logging.getLogger(__name__)

DOWNLOAD_URL = "https://imdpune.gov.in/cmpg/Griddata/RF25.php"
NETCDF_MAGIC = (b"CDF", b"\x89HDF")


def year_file(raw_dir: Path, year: int) -> Path:
    return raw_dir / f"imd_rf25_{year}.nc"


def download_year(year: int, raw_dir: Path, attempts: int = 5, timeout_s: int = 600) -> Path:
    """Download one year of IMD 0.25° rainfall, retrying on connection failures."""
    if year < 1901:
        raise ValueError(f"year {year} outside the IMD record")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = year_file(raw_dir, year)
    if target.exists():
        return target
    body = urllib.parse.urlencode({"RF25": str(year)}).encode()
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(DOWNLOAD_URL, data=body), timeout=timeout_s) as resp:
                data = resp.read()
            if not data.startswith(NETCDF_MAGIC):
                raise OSError(f"response for {year} is not NetCDF ({len(data)} bytes)")
            tmp = target.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(target)
            log.info("IMD %s downloaded (%.1f MB)", year, len(data) / 1e6)
            return target
        except OSError as exc:
            if attempt == attempts:
                raise
            wait = 2**attempt
            log.warning("IMD %s attempt %d failed (%s); retrying in %ds", year, attempt, exc, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


class IMDGridded(ObservationSource):
    """Reads downloaded yearly IMD files into the canonical observation layout."""

    name: ClassVar[str] = "imd_gridded"
    data_kind: ClassVar[DataKind] = DataKind.REAL

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

    def _load(self, start: date, end: date) -> xr.Dataset:
        paths = [year_file(self.raw_dir, y) for y in range(start.year, end.year + 1)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"IMD files not downloaded: {missing}")
        years = []
        for path in paths:
            with xr.open_dataset(path) as f:
                years.append(f[["RAINFALL"]].sel(TIME=slice(str(start), str(end))).load())
        ds = xr.concat(years, dim="TIME") if len(years) > 1 else years[0]
        ds = ds.rename(TIME="time", LATITUDE="lat", LONGITUDE="lon", RAINFALL=PRECIP_VAR)
        ds[PRECIP_VAR].attrs = {"units": "mm", "long_name": "observed rainfall, 24 h ending 03 UTC on date"}
        ds.attrs = {
            DATA_KIND_ATTR: self.data_kind.value,
            SOURCE_ATTR: "IMD 0.25 deg gridded rainfall (Pai et al. 2014), imdpune.gov.in",
        }
        return ds
