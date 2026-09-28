"""Build a live forecast product from the newest operational GEFS run (or a given date).

The product is marked `source: live`. Verification is pending until IMD publishes observations; with
--attach-observed, IMD grids already on disk are attached for dates that have them (verification only).
Forecasts valid outside the monsoon season (JJAS) are refused unless --allow-out-of-season, because the
models were trained on JJAS only.

Schedule it once a day after ~06 UTC (11:30 IST), when the 00 UTC run has been published, e.g. with
Windows Task Scheduler: .venv\\Scripts\\python scripts\\run_live.py --latest

Usage:
    python scripts/run_live.py --latest
    python scripts/run_live.py --date 2025-08-01 --attach-observed
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, date, datetime

import pandas as pd

from rainpp.config import load_settings
from rainpp.data.sources.gefs_operational import (
    FIRST_V12_INIT,
    MEMBER,
    GEFSOperational,
    GEFSOperationalFields,
    latest_available,
)
from rainpp.data.sources.imd import IMDGridded, year_file
from rainpp.products import ProductBuilder

log = logging.getLogger("run_live")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--latest", action="store_true", help="newest complete 00 UTC run")
    which.add_argument("--date", type=date.fromisoformat, help="00 UTC run of this date (YYYY-MM-DD)")
    parser.add_argument("--attach-observed", action="store_true", help="attach IMD observations already on disk")
    parser.add_argument("--allow-out-of-season", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    leads = list(settings.forecast.lead_days)

    init = latest_available(datetime.now(UTC).date(), max_hour=24 * max(leads) + 3) if args.latest else args.date
    if init < FIRST_V12_INIT:
        raise SystemExit(f"operational GEFSv12 starts {FIRST_V12_INIT}")
    valid = [pd.Timestamp(init) + pd.Timedelta(days=ld) for ld in leads]
    if any(v.month not in settings.season.months for v in valid) and not args.allow_out_of_season:
        raise SystemExit(f"{init}: forecasts valid {valid[0].date()}..{valid[-1].date()} fall outside the monsoon "
                         "season the models were trained on; use --allow-out-of-season to build anyway")

    start = time.time()
    fc = GEFSOperational(settings.domain).load([init], leads, [MEMBER])
    fields = GEFSOperationalFields().load([init], leads)
    log.info("downloaded %s in %.0fs", init, time.time() - start)

    observed = None
    raw_imd = settings.paths.data_dir / "raw" / "imd"
    if args.attach_observed and all(year_file(raw_imd, v.year).exists() for v in valid):
        observed = IMDGridded(raw_imd).load(valid[0].date(), valid[-1].date())

    product = ProductBuilder(settings).build_live(init, fc, fields, observed)
    product.save(settings.paths.data_dir / "products", init)
    lead1 = product.summary["leads"][str(leads[0])]
    log.info("%s saved (%.0fs total): day-1 regime %s (%.0f%%), verification: %s", init, time.time() - start,
             lead1["predicted_regime"], 100 * lead1["regime_confidence"], product.summary["verification"])


if __name__ == "__main__":
    main()
