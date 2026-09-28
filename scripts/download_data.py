"""Download real data: IMD rainfall, GEFSv12 rain-day totals, synoptic fields, and static fields.

GEFS is streamed from AWS and cropped before storage (data/interim/...). Files already on disk
are skipped, so the script can be re-run after a failure.

Usage:
    python scripts/download_data.py                   # everything, all years in the split
    python scripts/download_data.py --years 2018      # selected years
    python scripts/download_data.py --source fields   # one source: imd|gefs|fields|static
    python scripts/download_data.py --source gefs_op fields_op --years 2021 2022  # operational GEFSv12 (live inputs)
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd

from rainpp.config import load_settings
from rainpp.data.sources.gefs import GEFSReforecast
from rainpp.data.sources.gefs_fields import GEFSFields
from rainpp.data.sources.gefs_operational import GEFSOperational, GEFSOperationalFields
from rainpp.data.sources.imd import download_year

log = logging.getLogger("download")


def season_dates(year: int, months: list[int]) -> list:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return [d.date() for d in days if d.month in months]


def gefs_year(src: GEFSReforecast | GEFSOperational, year: int, months: list[int], leads: list[int], members: list[str],
              out_dir: Path) -> None:
    target = out_dir / f"gefs_apcp_{year}.nc"
    if target.exists():
        log.info("GEFS %s already present, skipping", year)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ds = src.load(season_dates(year, months), leads, members)
    tmp = target.with_suffix(".part")
    ds.to_netcdf(tmp, encoding={"precip_mm": {"zlib": True, "complevel": 4}})
    tmp.replace(target)
    log.info("GEFS %s: %d inits in %.0fs -> %s (%.1f MB)", year, ds.sizes["init_time"],
             time.time() - start, target.name, target.stat().st_size / 1e6)


def fields_year(src: GEFSFields | GEFSOperationalFields, year: int, months: list[int], leads: list[int], out_dir: Path) -> None:
    target = out_dir / f"fields_{year}.nc"
    if target.exists():
        log.info("fields %s already present, skipping", year)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ds = src.load(season_dates(year, months), leads)
    tmp = target.with_suffix(".part")
    ds.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    tmp.replace(target)
    log.info("fields %s: %d inits in %.0fs -> %s (%.1f MB)", year, ds.sizes["init_time"],
             time.time() - start, target.name, target.stat().st_size / 1e6)


def static_fields(domain, out_dir: Path) -> None:
    """GEFS land fraction and model terrain height on the 0.25° India grid (time-invariant)."""
    import s3fs
    import xarray as xr

    from rainpp.data.schema import DATA_KIND_ATTR, SOURCE_ATTR
    from rainpp.data.sources.gefs import crop_global_field
    from rainpp.data.sources.gefs_fields import _decode, find_message

    target = out_dir / "static.nc"
    if target.exists():
        log.info("static fields already present, skipping")
        return
    fs = s3fs.S3FileSystem(anon=True)
    land = _decode(fs.cat("noaa-gefs-retrospective/landsfc.pgrb2.0p25"))
    hgt_path = GEFSFields.file_path(date(2010, 6, 1), "hgt_sfc")
    start, end = find_message(fs.cat(hgt_path + ".idx").decode(), "HGT", "surface", 3)
    hgt = _decode(fs.cat_file(hgt_path, start=start, end=end))
    land_c, lats, lons = crop_global_field(land, domain)
    hgt_c = crop_global_field(hgt, domain)[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"land_fraction": (("lat", "lon"), land_c.astype("float32")),
         "elevation_m": (("lat", "lon"), hgt_c.astype("float32"), {"note": "GEFS model terrain, not a survey DEM"})},
        coords={"lat": lats, "lon": lons},
        attrs={DATA_KIND_ATTR: "real", SOURCE_ATTR: "GEFSv12 landsfc.pgrb2.0p25 and hgt_sfc (2010-06-01 00Z +3h)"},
    ).to_netcdf(target)
    log.info("static fields -> %s", target)


def main() -> None:
    settings = load_settings()
    first, last = settings.split.train[0], settings.split.test[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(first, last + 1)))
    parser.add_argument("--source", nargs="+", default=["all"],
                        choices=["imd", "gefs", "fields", "static", "gefs_op", "fields_op", "all"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_dir = settings.paths.data_dir
    wanted = set(args.source)
    if wanted & {"imd", "all"}:
        for year in args.years:
            download_year(year, data_dir / "raw" / "imd")
    if wanted & {"gefs", "all"}:
        src = GEFSReforecast(settings.domain, settings.forecast.init_hour_utc,
                             settings.forecast.rain_day_end_hour_utc, max_workers=args.workers)
        for year in args.years:
            gefs_year(src, year, settings.season.months, settings.forecast.lead_days,
                      settings.forecast.members, data_dir / "interim" / "gefs")
    if wanted & {"static", "all"}:
        static_fields(settings.domain, data_dir / "interim" / "static")
    if wanted & {"fields", "all"}:
        fsrc = GEFSFields(max_workers=args.workers)
        for year in args.years:
            fields_year(fsrc, year, settings.season.months, settings.forecast.lead_days,
                        data_dir / "interim" / "gefs_fields")
    op_dir = data_dir / "interim" / "gefs_operational"
    if "gefs_op" in wanted:
        osrc = GEFSOperational(settings.domain, settings.forecast.rain_day_end_hour_utc, max_workers=args.workers)
        for year in args.years:
            gefs_year(osrc, year, settings.season.months, settings.forecast.lead_days, settings.forecast.members, op_dir)
    if "fields_op" in wanted:
        ofsrc = GEFSOperationalFields(max_workers=args.workers)
        for year in args.years:
            fields_year(ofsrc, year, settings.season.months, settings.forecast.lead_days, op_dir)


if __name__ == "__main__":
    main()
