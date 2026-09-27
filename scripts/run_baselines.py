"""Phase 4: fit baselines on training years and compare them on VALIDATION years.

  A  raw NWP            (no correction)
  B1 quantile mapping   (per cell, per lead)
  B2 global LightGBM    (regime-blind, Tweedie)

LightGBM's number of boosting rounds is chosen by early stopping on the LAST training year,
then the model is refit on all training years. Validation years are therefore untouched by any
fitting or tuning, and test years are never read.

Usage:
    python scripts/run_baselines.py
"""

from __future__ import annotations

import json
import logging
import time

import pandas as pd

from rainpp.config import PROJECT_ROOT, load_settings, load_yaml
from rainpp.models.global_bc import LGBMCorrector
from rainpp.models.quantile_mapping import QuantileMapping
from rainpp.models.registry import write_metadata
from rainpp.verification.report import compare, to_markdown

log = logging.getLogger("run_baselines")

SOURCES = {"forecast": "NOAA GEFSv12 reforecast, control member",
           "observation": "IMD 0.25 deg gridded rainfall (Pai et al. 2014)"}


def load_years(processed, years: range) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(processed / f"table_{y}.parquet") for y in years], ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    cfg = load_yaml("models.yaml")
    thresholds = load_yaml("thresholds.yaml")["verification_thresholds_mm"]
    processed = settings.paths.data_dir / "processed"
    model_dir = settings.paths.model_dir
    (tr0, tr1), (va0, va1), (te0, _) = settings.split.train, settings.split.validation, settings.split.test
    assert va1 < te0, "validation must precede test"

    train = load_years(processed, range(tr0, tr1 + 1))
    valid = load_years(processed, range(va0, va1 + 1))
    log.info("train %d-%d: %d rows | validation %d-%d: %d rows", tr0, tr1, len(train), va0, va1, len(valid))

    t = time.time()
    qm = QuantileMapping().fit(train)
    valid["pred_qm"] = qm.predict(valid)
    log.info("quantile mapping fitted in %.0fs", time.time() - t)

    t = time.time()
    g = cfg["global_lgbm"]
    inner_train, inner_stop = train[train.valid_date.dt.year < tr1], train[train.valid_date.dt.year == tr1]
    probe = LGBMCorrector(cfg["features"], g["params"], g["num_boost_round"], g["early_stopping_rounds"])
    probe.fit(inner_train, inner_stop)
    rounds = probe.booster.best_iteration
    log.info("early stopping on %d chose %d rounds", tr1, rounds)
    lgbm = LGBMCorrector(cfg["features"], g["params"]).fit_fixed_rounds(train, rounds)
    valid["pred_lgbm"] = lgbm.predict(valid)
    log.info("global LightGBM refit on %d-%d in %.0fs", tr0, tr1, time.time() - t)

    ev = cfg["evaluation"]
    models = {"A_raw_nwp": "nwp_precip_mm", "B1_quantile_mapping": "pred_qm", "B2_global_lgbm": "pred_lgbm"}
    results = compare(valid, models, settings.domain, thresholds, ev["fss_threshold_mm"], ev["fss_window_cells"])

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    header = {"scope": f"VALIDATION years {va0}-{va1}; test years {te0}+ not used", "data_kind": "real",
              "trained_on": [tr0, tr1], "sources": SOURCES}
    (reports / "phase4_validation.json").write_text(json.dumps({**header, "results": results}, indent=2, default=float),
                                                    encoding="utf-8")
    md = to_markdown(results, f"Phase 4 baselines — validation {va0}–{va1} (REAL DATA)")
    (reports / "phase4_validation.md").write_text(md, encoding="utf-8")
    log.info("report written to %s", reports / "phase4_validation.md")

    summary = {str(lead): {m: {"rmse": r[m]["rmse"], "bias": r[m]["bias"]} for m in r} for lead, r in results.items()}
    qm.save(model_dir / "quantile_mapping")
    write_metadata(model_dir / "quantile_mapping", name="quantile_mapping", model_type="empirical quantile mapping",
                   features=["nwp_precip_mm"], training_period=(tr0, tr1), validation_period=(va0, va1),
                   data_sources=SOURCES, validation_metrics={k: v["B1_quantile_mapping"] for k, v in summary.items()})
    lgbm.save(model_dir / "global_lgbm")
    write_metadata(model_dir / "global_lgbm", name="global_lgbm", model_type="LightGBM Tweedie regressor",
                   features=cfg["features"], training_period=(tr0, tr1), validation_period=(va0, va1),
                   data_sources=SOURCES, validation_metrics={k: v["B2_global_lgbm"] for k, v in summary.items()},
                   extra={"boosting_rounds": rounds, "rounds_chosen_on_year": tr1, "params": g["params"],
                          "feature_importance_gain_share": lgbm.feature_importance()})


if __name__ == "__main__":
    main()
