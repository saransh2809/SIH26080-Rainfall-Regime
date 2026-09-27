"""Build the per-year training tables from downloaded GEFS and IMD data.

Steps: validate → pair each forecast with its valid-day observation → features → Parquet.
The observed climatology feature is fitted on TRAINING years only and reused for all years.

Usage:
    python scripts/build_dataset.py                    # all years in the split
    python scripts/build_dataset.py --years 2010 2011  # selected years
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from rainpp.config import load_settings
from rainpp.data.align import pair_forecast_observation
from rainpp.data.features import build_table, fit_climatology
from rainpp.data.sources.gefs import open_interim
from rainpp.data.sources.imd import IMDGridded

log = logging.getLogger("build_dataset")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    data_dir = settings.paths.data_dir
    imd = IMDGridded(data_dir / "raw" / "imd")
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_start, train_end = settings.split.train
    obs_train = imd.load(date(train_start, 1, 1), date(train_end, 12, 31))["precip_mm"]
    climatology = fit_climatology(obs_train)
    log.info("climatology fitted on %d-%d", train_start, train_end)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+",
                        default=list(range(settings.split.train[0], settings.split.test[1] + 1)))
    args = parser.parse_args()

    meta_path = out_dir / "dataset_metadata.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    summary = {int(k): v for k, v in previous.get("years", {}).items()}
    for year in args.years:
        fc = open_interim(data_dir / "interim" / "gefs" / f"gefs_apcp_{year}.nc")
        obs = imd.load(date(year, 1, 1), date(year, 12, 31))
        paired = pair_forecast_observation(fc, obs)
        table = build_table(fc, paired, climatology)
        table.to_parquet(out_dir / f"table_{year}.parquet", index=False)
        summary[year] = {"rows": len(table), "missing_observation_dates": paired.attrs["missing_observation_dates"]}
        log.info("%d: %d rows", year, len(table))

    meta = {
        "data_kind": "real",
        "forecast_source": "NOAA GEFSv12 reforecast, control member",
        "observation_source": "IMD 0.25 deg gridded rainfall (Pai et al. 2014)",
        "climatology_fit_years": [train_start, train_end],
        "split": settings.split.model_dump(),
        "years": dict(sorted(summary.items())),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
