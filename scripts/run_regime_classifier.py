"""Phase 5: train synoptic regime classifiers and compare them on VALIDATION years.

  R0   rule baseline (threshold on forecast core anomaly + fitted vorticity threshold)
  RF   RandomForest, balanced class weights
  LGBM LightGBM multiclass; rounds chosen by early stopping on the last training year, then refit

Test years are never read. Scores are reported overall, per lead day, and on July-August only
(where the active/break definition is published rather than extended).

Usage:
    python scripts/run_regime_classifier.py
"""

from __future__ import annotations

import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xarray as xr

from rainpp.config import PROJECT_ROOT, load_settings, load_yaml
from rainpp.models.registry import write_metadata
from rainpp.regimes.classifier import (
    CLASSES,
    ForestClassifier,
    LGBMRegimeClassifier,
    RuleClassifier,
    balanced_weights,
    classification_scores,
    predict_with_confidence,
)
from rainpp.regimes.cnn import CNNRegimeClassifier

log = logging.getLogger("run_regime_classifier")
SOURCES = {"features": "NOAA GEFSv12 reforecast c00 (forecast fields) + IMD core-zone anomaly before init",
           "labels": "IMD gridded rainfall (active/break) + IMD RSMC best track (depressions)"}


def choose_rounds(features: list[str], params: dict, train: pd.DataFrame, stop: pd.DataFrame, cfg: dict) -> int:
    codes = {c: i for i, c in enumerate(CLASSES)}
    full = {**params, "objective": "multiclass", "num_class": len(CLASSES)}
    dtrain = lgb.Dataset(train[features].astype(np.float32), label=train.synoptic_regime.map(codes),
                         weight=balanced_weights(train.synoptic_regime))
    dstop = lgb.Dataset(stop[features].astype(np.float32), label=stop.synoptic_regime.map(codes),
                        weight=balanced_weights(stop.synoptic_regime), reference=dtrain)
    booster = lgb.train(full, dtrain, num_boost_round=cfg["max_boost_round"], valid_sets=[dstop],
                        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    return booster.best_iteration


MAP_CHANNELS = ("u850", "v850", "vort850_1e5", "mslp_hpa", "pwat_mm")


def load_maps(rows: pd.DataFrame, fields_dir) -> np.ndarray:
    """(n_rows, channels, lat, lon) forecast maps matching each row's init_time and lead_day."""
    out = []
    for year, group in rows.groupby(rows["init_time"].dt.year, sort=False):
        ds = xr.open_dataset(fields_dir / f"fields_{year}.nc").load()
        stacked = np.stack([ds[c].values for c in MAP_CHANNELS], axis=2)  # init, lead, channel, lat, lon
        i = ds.indexes["init_time"].get_indexer(group["init_time"])
        j = ds.indexes["lead_day"].get_indexer(group["lead_day"])
        if (i < 0).any() or (j < 0).any():
            raise KeyError(f"{year}: rows without matching forecast fields")
        out.append(pd.Series(list(stacked[i, j]), index=group.index))
    return np.stack(pd.concat(out).sort_index().to_list()).astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    cfg = load_yaml("models.yaml")["regime_classifier"]
    features = cfg["features"]
    table = pd.read_parquet(settings.paths.data_dir / "processed" / "regime_features.parquet")
    table = table.dropna(subset=["synoptic_regime"])
    (tr0, tr1), (va0, va1) = settings.split.train, settings.split.validation
    year = table["valid_date"].dt.year
    train, valid = table[year.between(tr0, tr1)].reset_index(drop=True), table[year.between(va0, va1)].reset_index(drop=True)
    log.info("train %d rows %s | validation %d rows %s", len(train), train.synoptic_regime.value_counts().to_dict(),
             len(valid), valid.synoptic_regime.value_counts().to_dict())

    rule = RuleClassifier().fit(train, train.synoptic_regime)
    rf = ForestClassifier(features).fit(train, train.synoptic_regime)
    rounds = choose_rounds(features, cfg["lgbm_params"], train[train.valid_date.dt.year < tr1],
                           train[train.valid_date.dt.year == tr1], cfg)
    lgbm = LGBMRegimeClassifier(features, cfg["lgbm_params"], rounds).fit(train, train.synoptic_regime)
    log.info("rule vorticity threshold %.2f; LightGBM rounds %d", rule.vort_threshold, rounds)

    fields_dir = settings.paths.data_dir / "interim" / "gefs_fields"
    train_maps, valid_maps = load_maps(train, fields_dir), load_maps(valid, fields_dir)
    inner = (train.valid_date.dt.year < tr1).to_numpy()
    probe = CNNRegimeClassifier(cfg["cnn_scalar_features"]).fit(
        train_maps[inner], train[inner], train.synoptic_regime[inner].to_numpy(),
        stop=(train_maps[~inner], train[~inner], train.synoptic_regime[~inner].to_numpy()))
    cnn = CNNRegimeClassifier(cfg["cnn_scalar_features"])
    cnn.epochs = probe.epochs
    cnn.fit(train_maps, train, train.synoptic_regime.to_numpy())
    log.info("CNN epochs chosen on %d: %d", tr1, cnn.epochs)

    models = {"R0_rule": rule, "RF": rf, "LGBM": lgbm}
    preds = {name: predict_with_confidence(m, valid) for name, m in models.items()}
    cnn_proba = cnn.predict_proba(valid_maps, valid)
    preds["CNN"] = pd.DataFrame({"predicted_regime": cnn_proba.idxmax(axis=1), "confidence": cnn_proba.max(axis=1)})
    truth = valid["synoptic_regime"].to_numpy()
    jul_aug = valid["valid_date"].dt.month.isin([7, 8]).to_numpy()
    results = {}
    for name, p in preds.items():
        y = p["predicted_regime"].to_numpy()
        results[name] = {
            "all": classification_scores(truth, y),
            "july_august_only": classification_scores(truth[jul_aug], y[jul_aug]),
            "by_lead": {int(ld): classification_scores(truth[valid.lead_day == ld], y[valid.lead_day == ld])
                        for ld in sorted(valid.lead_day.unique())},
        }
    majority = train.synoptic_regime.mode()[0]
    results["reference_always_" + majority] = {"all": classification_scores(truth, np.full(len(truth), majority))}

    reports = PROJECT_ROOT / "reports"
    header = {"scope": f"VALIDATION years {va0}-{va1}; test years not used", "data_kind": "real",
              "trained_on": [tr0, tr1], "sources": SOURCES}
    (reports / "phase5_validation.json").write_text(json.dumps({**header, "results": results}, indent=2, default=float),
                                                    encoding="utf-8")
    (reports / "phase5_validation.md").write_text(to_markdown(results, va0, va1), encoding="utf-8")
    log.info("report written to %s", reports / "phase5_validation.md")

    out = settings.paths.model_dir / "regime_classifier"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf.model, out / "rf.joblib")
    lgbm.booster.save_model(str(out / "lgbm.txt"))
    torch.save({"state_dict": cnn.net.state_dict(), "epochs": cnn.epochs, "channels": MAP_CHANNELS,
                "scalar_features": cnn.scalar_features, "map_mean": cnn.map_mean, "map_sd": cnn.map_sd,
                "scalar_mean": cnn.scalar_mean, "scalar_sd": cnn.scalar_sd}, out / "cnn.pt")
    (out / "rule.json").write_text(json.dumps({"vort_threshold": rule.vort_threshold,
                                               "anomaly_threshold": rule.anomaly_threshold}), encoding="utf-8")
    write_metadata(out, name="regime_classifier", model_type="R0 rule / RandomForest / LightGBM multiclass",
                   features=features, training_period=(tr0, tr1), validation_period=(va0, va1), data_sources=SOURCES,
                   validation_metrics={n: {k: r["all"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}
                                       for n, r in results.items()},
                   extra={"classes": CLASSES, "lgbm_rounds": rounds, "cnn_epochs": cnn.epochs,
                          "cnn_map_channels": list(MAP_CHANNELS), "rf_feature_importance": rf.feature_importance()})


def to_markdown(results: dict, va0: int, va1: int) -> str:
    def f(v: float) -> str:
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"

    names = list(results)
    lines = [f"## Phase 5 regime classifier — validation {va0}–{va1} (REAL DATA)", "",
             "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        lines.append(f"| {key} | " + " | ".join(f(results[n]["all"][key]) for n in names) + " |")
    for c in CLASSES:
        for m in ("precision", "recall", "f1"):
            lines.append(f"| {c} {m} | " + " | ".join(f(results[n]["all"]["per_class"][c][m]) for n in names) + " |")
    support = results[names[0]]["all"]["per_class"]
    lines += ["", "Support (validation rows): " + ", ".join(f"{c}: {support[c]['support']}" for c in CLASSES), "",
              "### July–August only (published active/break definition)", "",
              "| Metric | " + " | ".join(n for n in names if "july_august_only" in results[n]) + " |",
              "|---|" + "---|" * sum("july_august_only" in results[n] for n in names)]
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        lines.append(f"| {key} | " + " | ".join(f(results[n]["july_august_only"][key])
                                                 for n in names if "july_august_only" in results[n]) + " |")
    lines += ["", "### Confusion matrices (rows = true, columns = predicted; order " + ", ".join(CLASSES) + ")", ""]
    for n in names:
        lines.append(f"**{n}**: `{results[n]['all']['confusion_matrix']['rows_true_cols_pred']}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
