"""Build synoptic regime labels (observations) and the classifier feature table (forecast-time).

Outputs in data/processed/:
  regime_labels.parquet     one row per JJAS rain day, with evidence level and core anomaly
  regime_features.parquet   one row per (init_time, lead_day) with forecast-time features + label
  regime_label_summary.json class counts per year

Usage:
    python scripts/build_regimes.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

import pandas as pd
import xarray as xr

from rainpp.config import load_settings, load_yaml
from rainpp.data.sources.gefs import open_interim
from rainpp.data.sources.imd import IMDGridded
from rainpp.regimes.features import (
    add_forecast_anomaly,
    build_classifier_table,
    fit_forecast_core_stats,
)
from rainpp.regimes.labels import (
    core_zone_series,
    depression_days,
    normalised_anomaly,
    synoptic_labels,
)

log = logging.getLogger("build_regimes")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    cfg = load_yaml("regimes.yaml")
    syn = cfg["synoptic"]
    data_dir = settings.paths.data_dir
    processed = data_dir / "processed"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--last-year", type=int, default=settings.split.test[1],
                        help="build up to this year (e.g. before test-year data is downloaded)")
    args = parser.parse_args()
    first, last = settings.split.train[0], args.last_year
    if last < settings.split.validation[1]:
        raise SystemExit("--last-year must include the validation years")
    clim0, _ = syn["active_break"]["climatology_years"]

    imd = IMDGridded(data_dir / "raw" / "imd")
    obs = imd.load(date(clim0, 1, 1), date(last, 12, 31))
    core_box = syn["active_break"]["core_zone_box"]
    core = core_zone_series(obs, core_box)
    log.info("core-zone series %s..%s (%d days)", core.index[0].date(), core.index[-1].date(), len(core))

    track = pd.read_parquet(data_dir / "raw" / "tracks" / "imdtrack_observations.parquet")
    labels = synoptic_labels(core, depression_days(track, syn["depression"]["box"]), syn, settings.season.months)
    labels = labels[(labels.index.year >= first) & (labels.index.year <= last)]
    labels.to_parquet(processed / "regime_labels.parquet")

    summary = labels.groupby([labels.index.year, "synoptic_regime"]).size().unstack(fill_value=0)
    evidence = labels["evidence"].value_counts().to_dict()
    (processed / "regime_label_summary.json").write_text(
        json.dumps({"counts_by_year": summary.to_dict(orient="index"), "evidence": evidence}, indent=2, default=int),
        encoding="utf-8")
    log.info("labels:\n%s\nevidence: %s", summary, evidence)

    anomaly = normalised_anomaly(core, tuple(syn["active_break"]["climatology_years"]))
    land_mask = obs["precip_mm"].isel(time=0).notnull()
    tables = []
    for year in range(first, last + 1):
        fields = xr.open_dataset(data_dir / "interim" / "gefs_fields" / f"fields_{year}.nc").load()
        rain = open_interim(data_dir / "interim" / "gefs" / f"gefs_apcp_{year}.nc")
        tables.append(build_classifier_table(fields, rain, land_mask, cfg["classifier_features"], core_box, anomaly,
                                             (min(settings.season.months), max(settings.season.months))))
    table = pd.concat(tables, ignore_index=True)

    tr0, tr1 = settings.split.train
    train_rows = table["valid_date"].dt.year.between(tr0, tr1)
    stats = fit_forecast_core_stats(table[train_rows])
    table = add_forecast_anomaly(table, stats)
    table = table.merge(labels[["synoptic_regime", "evidence"]], left_on="valid_date", right_index=True, how="left")
    unlabeled = int(table["synoptic_regime"].isna().sum())
    log.info("%d rows are valid outside the labelled months: kept for prediction, never for training/scoring",
             unlabeled)
    table.to_parquet(processed / "regime_features.parquet", index=False)
    log.info("classifier table: %d rows, %d columns", len(table), table.shape[1])


if __name__ == "__main__":
    main()
