"""Build dashboard forecast products for a range of 00 UTC initialisation dates.

Test-year dates are refused unless --final-test is given, so the test period is opened deliberately,
once, after model choices are locked.

Usage:
    python scripts/build_products.py --start 2017-08-24 --end 2017-08-31
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from rainpp.config import load_settings
from rainpp.products import ProductBuilder

log = logging.getLogger("build_products")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, type=pd.Timestamp)
    parser.add_argument("--end", required=True, type=pd.Timestamp)
    parser.add_argument("--final-test", action="store_true", help="allow test-year dates (final evaluation only)")
    args = parser.parse_args()

    test0 = settings.split.test[0]
    dates = pd.date_range(args.start, args.end, freq="D")
    if (dates.year >= test0).any() and not args.final_test:
        raise SystemExit(f"refusing test-period dates (>= {test0}) without --final-test")

    builder = ProductBuilder(settings)
    out_dir = settings.paths.data_dir / "products"
    for d in dates:
        start = time.time()
        builder.build(d.date()).save(out_dir, d.date())
        log.info("%s built in %.1fs", d.date(), time.time() - start)


if __name__ == "__main__":
    main()
