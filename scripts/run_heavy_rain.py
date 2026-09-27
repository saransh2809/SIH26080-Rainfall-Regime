"""Phase 7: heavy-rainfall probability, compared on VALIDATION years.

For each IMD threshold (heavy ≥ 64.5 mm, very heavy ≥ 115.6 mm):
  CLIM     climatological event frequency (training years; the Brier-skill reference)
  RAW01    raw GEFS exceedance as a 0/1 "probability" (only the control member was downloaded,
           so an ensemble-fraction baseline is not available)
  QM01     quantile-mapped GEFS exceedance as 0/1
  P_blind  LightGBM probability, forecast + static + climatology features, no regime
  P_regime P_blind features + predicted synoptic-regime probabilities + local regime

Rounds are chosen by early stopping on the last training year and the models are refit on all
training years. Test years are never read.

Usage:
    python scripts/run_heavy_rain.py
"""

from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.config import PROJECT_ROOT, load_settings, load_yaml
from rainpp.data.sources.imd import IMDGridded
from rainpp.models.heavy_rain import HeavyRainModel, fit_event_climatology
from rainpp.models.quantile_mapping import QuantileMapping
from rainpp.models.registry import write_metadata
from rainpp.regimes.augment import LOCAL_COLS, PROB_COLS, load_augmented, synoptic_probabilities
from rainpp.regimes.local import static_terrain
from rainpp.verification.bootstrap import block_bootstrap_difference
from rainpp.verification.probabilistic import brier_score, probability_scores

log = logging.getLogger("run_heavy_rain")


def add_climatology(table: pd.DataFrame, clim: xr.DataArray, column: str) -> None:
    table[column] = clim.sel(
        dayofyear=xr.DataArray(table["valid_date"].dt.dayofyear.to_numpy(), dims="row"),
        lat=xr.DataArray(table["lat"].to_numpy(), dims="row"),
        lon=xr.DataArray(table["lon"].to_numpy(), dims="row")).to_numpy().astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    mcfg, rcfg, tcfg = load_yaml("models.yaml"), load_yaml("regimes.yaml"), load_yaml("thresholds.yaml")
    data_dir, model_dir = settings.paths.data_dir, settings.paths.model_dir
    (tr0, tr1), (va0, va1) = settings.split.train, settings.split.validation
    train_years, valid_years = range(tr0, tr1 + 1), range(va0, va1 + 1)

    regime_table = pd.read_parquet(data_dir / "processed" / "regime_features.parquet")
    labels = pd.read_parquet(data_dir / "processed" / "regime_labels.parquet")["synoptic_regime"]
    rounds = json.loads((model_dir / "regime_classifier" / "model_metadata.json").read_text())["lgbm_rounds"]
    probs = synoptic_probabilities(regime_table, train_years, valid_years, mcfg["regime_correction"]["classifier"],
                                   mcfg["regime_classifier"], rounds)
    terrain = static_terrain(xr.open_dataset(data_dir / "interim" / "static" / "static.nc").load(),
                             settings.domain.resolution_deg)
    train = load_augmented(data_dir, train_years, terrain, rcfg["local"], probs, labels)
    valid = load_augmented(data_dir, valid_years, terrain, rcfg["local"], probs, labels)

    obs_train = IMDGridded(data_dir / "raw" / "imd").load(date(tr0, 1, 1), date(tr1, 12, 31))["precip_mm"]
    qm = QuantileMapping.load(model_dir / "quantile_mapping" / "quantile_mapping.npz")
    valid["pred_qm"] = qm.predict(valid)

    base = mcfg["features"] + mcfg["regime_correction"]["static_features"]
    params = mcfg["global_lgbm"]["params"]
    inner, stop = train["valid_date"].dt.year < tr1, train["valid_date"].dt.year == tr1
    results, significance, saved = {}, {}, {}
    for name in tcfg["probability_targets"]:
        t = float(tcfg["thresholds_mm"][name])
        clim_col = f"clim_p_ge_{t}"
        clim = fit_event_climatology(obs_train, t)
        add_climatology(train, clim, clim_col)
        add_climatology(valid, clim, clim_col)
        feature_sets = {"P_blind": base + [clim_col], "P_regime": base + [clim_col] + PROB_COLS + LOCAL_COLS}
        preds = {"CLIM": valid[clim_col].to_numpy(),
                 "RAW01": (valid["nwp_precip_mm"].to_numpy() >= t).astype(float),
                 "QM01": (valid["pred_qm"].to_numpy() >= t).astype(float)}
        for model_name, features in feature_sets.items():
            model = HeavyRainModel(t, features, params).fit(train[inner], train[stop], train)
            preds[model_name] = model.predict_proba(valid)
            valid[f"p_{model_name}_{name}"] = preds[model_name]
            saved[(name, model_name)] = model
            log.info("%s %s: %d rounds", name, model_name, model.rounds)

        outcome = (valid["obs_precip_mm"].to_numpy() >= t).astype(float)
        results[name] = {"threshold_mm": t, "all": {m: probability_scores(p, outcome, preds["CLIM"])
                                                    for m, p in preds.items()}}
        results[name]["by_lead"] = {
            int(ld): {m: {k: v for k, v in probability_scores(p[rows], outcome[rows], preds["CLIM"][rows]).items()
                          if k != "reliability"} for m, p in preds.items()}
            for ld in sorted(valid["lead_day"].unique()) for rows in [(valid["lead_day"] == ld).to_numpy()]}
        brier = lambda f, o, t=t: brier_score(f, (o >= t).astype(float))
        significance[name] = {
            "P_regime vs P_blind": block_bootstrap_difference(valid, brier, f"p_P_blind_{name}", f"p_P_regime_{name}",
                                                               **mcfg["regime_correction"]["bootstrap"]),
        }

    reports = PROJECT_ROOT / "reports"
    header = {"scope": f"VALIDATION years {va0}-{va1}; test years not used", "data_kind": "real",
              "note": "RAW01/QM01 are deterministic 0/1 exceedances; only the GEFS control member is available"}
    (reports / "phase7_validation.json").write_text(
        json.dumps({**header, "results": results, "significance": significance}, indent=2, default=float), encoding="utf-8")
    (reports / "phase7_validation.md").write_text(to_markdown(results, significance, va0, va1), encoding="utf-8")
    log.info("report written to %s", reports / "phase7_validation.md")

    for (name, model_name), model in saved.items():
        out = model_dir / "heavy_rain" / f"{model_name}_{name}"
        model.save(out)
        s = results[name]["all"][model_name]
        write_metadata(out, name=model.name, model_type="LightGBM binary (log-loss, unweighted)",
                       features=model.features, training_period=(tr0, tr1), validation_period=(va0, va1),
                       data_sources={"target": "IMD gridded rainfall >= threshold"},
                       validation_metrics={k: s[k] for k in ("brier", "bss_vs_climatology", "roc_auc")},
                       extra={"threshold_mm": model.threshold_mm, "rounds": model.rounds})


def to_markdown(results: dict, significance: dict, va0: int, va1: int) -> str:
    def f(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

    lines = [f"## Phase 7 heavy-rain probability — validation {va0}–{va1} (REAL DATA)", ""]
    for name, r in results.items():
        models = list(r["all"])
        base = r["all"][models[0]]
        title = (f"### {name} (≥ {r['threshold_mm']} mm) — {base['events']:,} events in {base['n']:,} cell-days "
                 f"(base rate {base['base_rate']:.4f})")
        lines += [title, "", "| Metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for k in ("brier", "bss_vs_climatology", "roc_auc", "mean_probability"):
            lines.append(f"| {k} | " + " | ".join(f(r["all"][m][k]) for m in models) + " |")
        for comp, res in significance[name].items():
            verdict = "significant" if res["significant"] else "not significant"
            lines += ["", (f"Brier improvement {comp}: {res['improvement']:.6f}, 95% CI "
                           f"[{res['ci95_low']:.6f}, {res['ci95_high']:.6f}] — {verdict}")]
        lines += ["", "Reliability of P_regime (forecast bin → mean forecast / observed frequency / count):", ""]
        for row in r["all"]["P_regime"]["reliability"]:
            if row["n"]:
                lines.append(f"- {row['bin'][0]:.1f}–{row['bin'][1]:.1f}: {row['mean_forecast']:.3f} / "
                             f"{row['observed_frequency']:.3f} / {row['n']:,}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
