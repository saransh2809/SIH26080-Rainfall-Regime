"""Download real training data: IMD gridded rainfall and GEFSv12 reforecast rain-day totals.

GEFS is streamed from AWS, cropped to the India domain, and stored as one NetCDF per year in
data/interim/gefs/. Years already on disk are skipped, so the script can be re-run after a failure.

Usage:
    python scripts/download_data.py                 # all years in the configured split
    python scripts/download_data.py --years 2018    # selected years
    python scripts/download_data.py --source imd    # one source only
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from rainpp.config import load_settings
from rainpp.data.sources.gefs import GEFSReforecast
from rainpp.data.sources.imd import download_year

log = logging.getLogger("download")


def season_dates(year: int, months: list[int]) -> list:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return [d.date() for d in days if d.month in months]


def gefs_year(src: GEFSReforecast, year: int, months: list[int], leads: list[int], members: list[str],
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


def main() -> None:
    settings = load_settings()
    first, last = settings.split.train[0], settings.split.test[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(first, last + 1)))
    parser.add_argument("--source", choices=["imd", "gefs", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_dir = settings.paths.data_dir
    if args.source in ("imd", "all"):
        for year in args.years:
            download_year(year, data_dir / "raw" / "imd")
    if args.source in ("gefs", "all"):
        src = GEFSReforecast(settings.domain, settings.forecast.init_hour_utc,
                             settings.forecast.rain_day_end_hour_utc, max_workers=args.workers)
        for year in args.years:
            gefs_year(src, year, settings.season.months, settings.forecast.lead_days,
                      settings.forecast.members, data_dir / "interim" / "gefs")


if __name__ == "__main__":
    main()
