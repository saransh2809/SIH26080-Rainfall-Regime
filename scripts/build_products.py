"""Build dashboard forecast products for a range of 00 UTC initialisation dates.

  --source archive      GEFSv12 reforecast tables (default)
  --source operational  saved GEFSv12 operational runs (data/interim/gefs_operational), through the live pipeline,
                        with IMD observations attached where published

Dates in the held-out periods (test years, operational test years) are refused unless --final-test is given, so
they are opened deliberately, once, after the one-time evaluation of the frozen models.

Usage:
    python scripts/build_products.py --start 2017-08-24 --end 2017-08-31
    python scripts/build_products.py --source operational --start 2022-06-14 --end 2022-06-16 --final-test
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import xarray as xr

from rainpp.config import load_settings
from rainpp.data.sources.gefs import open_interim
from rainpp.data.sources.imd import IMDGridded, year_file
from rainpp.products import ProductBuilder

log = logging.getLogger("build_products")


def held_out(settings, year: int) -> bool:
    t0, t1 = settings.split.test
    op = settings.split.operational_test
    return t0 <= year <= t1 or (op is not None and op[0] <= year <= op[1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, type=pd.Timestamp)
    parser.add_argument("--end", required=True, type=pd.Timestamp)
    parser.add_argument("--source", choices=["archive", "operational"], default="archive")
    parser.add_argument("--final-test", action="store_true", help="allow held-out dates (after the final evaluation)")
    args = parser.parse_args()

    dates = pd.date_range(args.start, args.end, freq="D")
    if any(held_out(settings, d.year) for d in dates) and not args.final_test:
        raise SystemExit("refusing held-out (test / operational-test) dates without --final-test")

    builder = ProductBuilder(settings)
    out_dir = settings.paths.data_dir / "products"
    op_dir = settings.paths.data_dir / "interim" / "gefs_operational"
    raw_imd = settings.paths.data_dir / "raw" / "imd"
    for d in dates:
        start = time.time()
        init = d.date()
        if args.source == "archive":
            product = builder.build(init)
        else:
            fc = open_interim(op_dir / f"gefs_apcp_{d.year}.nc").sel(init_time=[d])
            fields = xr.open_dataset(op_dir / f"fields_{d.year}.nc").sel(init_time=[d]).load()
            valid = [d + pd.Timedelta(days=int(ld)) for ld in fc.lead_day.values]
            observed = (IMDGridded(raw_imd).load(valid[0].date(), valid[-1].date())
                        if all(year_file(raw_imd, v.year).exists() for v in valid) else None)
            product = builder.build_live(init, fc, fields, observed)
        product.save(out_dir, init)
        log.info("%s built in %.1fs", init, time.time() - start)


if __name__ == "__main__":
    main()
