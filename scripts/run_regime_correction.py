"""Phase 6: regime-aware bias correction, compared on VALIDATION years.

  B2      global LightGBM (Phase 4 model, reloaded)
  B2s     B2 + static terrain/coast features, no regime       (ablation: is it the regime or the terrain?)
  C1      B2s + predicted synoptic-regime probabilities + local regime one-hot
  C2      one LightGBM per predicted synoptic regime (B2s features + local regime)
  C1_oracle  C1 with OBSERVED synoptic regimes — diagnostic upper bound, not usable operationally

Regime probabilities for TRAINING rows are out-of-fold (classifier refit without the row's year), so
the corrector learns from the same kind of imperfect regime information it receives in operation.
Validation rows use the classifier trained on all training years. Test years are never read.

Usage:
    python scripts/run_regime_correction.py
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import xarray as xr

from rainpp.config import PROJECT_ROOT, load_settings, load_yaml
from rainpp.models.global_bc import LGBMCorrector
from rainpp.models.quantile_mapping import QuantileMapping
from rainpp.models.regime_bc import RegimeSplitCorrector
from rainpp.models.registry import write_metadata
from rainpp.regimes.augment import (
    LOCAL_COLS,
    ORACLE_COLS,
    PROB_COLS,
    load_augmented,
    synoptic_probabilities,
)
from rainpp.regimes.local import LOCAL_CODES, static_terrain
from rainpp.verification.bootstrap import block_bootstrap_difference
from rainpp.verification.metrics import calculate_ets, calculate_rmse
from rainpp.verification.report import compare, score, to_markdown

log = logging.getLogger("run_regime_correction")


def breakdown(valid: pd.DataFrame, models: dict[str, str], group_col: str) -> dict:
    """Continuous and heavy-rain scores within each regime (observed synoptic or local)."""
    out = {}
    for regime, group in valid.groupby(group_col):
        out[str(regime)] = {"rows": len(group), "observed_heavy_events": int((group.obs_precip_mm >= 64.5).sum())}
        for name, col in models.items():
            s = score(group, col, None, [64.5], [], [])
            out[str(regime)][name] = {"rmse": s["rmse"], "bias": s["bias"], "ets_64.5": s["categorical"]["64.5"]["ets"]}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    mcfg, rcfg = load_yaml("models.yaml"), load_yaml("regimes.yaml")
    ccfg, bcfg = mcfg["regime_classifier"], mcfg["regime_correction"]
    data_dir, model_dir = settings.paths.data_dir, settings.paths.model_dir
    (tr0, tr1), (va0, va1) = settings.split.train, settings.split.validation
    train_years, valid_years = range(tr0, tr1 + 1), range(va0, va1 + 1)

    regime_table = pd.read_parquet(data_dir / "processed" / "regime_features.parquet")
    labels = pd.read_parquet(data_dir / "processed" / "regime_labels.parquet")["synoptic_regime"]
    rounds = json.loads((model_dir / "regime_classifier" / "model_metadata.json").read_text())["lgbm_rounds"]
    probs = synoptic_probabilities(regime_table, train_years, valid_years, bcfg["classifier"], ccfg, rounds)
    terrain = static_terrain(xr.open_dataset(data_dir / "interim" / "static" / "static.nc").load(),
                             settings.domain.resolution_deg)

    needed = (mcfg["features"] + bcfg["static_features"] + PROB_COLS + LOCAL_COLS + ORACLE_COLS
              + ["pred_regime", "obs_regime", "valid_date", "lead_day", "lat", "lon", "obs_precip_mm"])
    needed = list(dict.fromkeys(needed))
    train = load_augmented(data_dir, train_years, terrain, rcfg["local"], probs, labels,
                           cell_every=mcfg["training"]["cell_every"], columns=needed)
    valid = load_augmented(data_dir, valid_years, terrain, rcfg["local"], probs, labels, columns=needed)
    base = mcfg["features"]
    static = bcfg["static_features"]
    feature_sets = {"B2s": base + static, "C1": base + static + PROB_COLS + LOCAL_COLS,
                    "C1_oracle": base + static + ORACLE_COLS + LOCAL_COLS}

    g = mcfg["global_lgbm"]
    b2_rounds = json.loads((model_dir / "global_lgbm" / "model_metadata.json").read_text())["boosting_rounds"]
    valid["pred_raw_qm"] = QuantileMapping.load(model_dir / "quantile_mapping" / "quantile_mapping.npz").predict(valid)
    valid["pred_B2"] = LGBMCorrector.load(model_dir / "global_lgbm").predict(valid)
    fitted = {}
    for name, features in feature_sets.items():
        fitted[name] = LGBMCorrector(features, g["params"]).fit_fixed_rounds(train, b2_rounds)
        valid[f"pred_{name}"] = fitted[name].predict(valid)
        log.info("%s fitted (%d features)", name, len(features))
    c2 = RegimeSplitCorrector(base + static + LOCAL_COLS, g["params"], b2_rounds, "pred_regime",
                              bcfg["c2_min_rows"]).fit(train)
    valid["pred_C2"] = c2.predict(valid)
    log.info("C2 fitted; fallback regimes: %s", c2.fallback_regimes)

    models = {"A_raw_nwp": "nwp_precip_mm", "B1_quantile_mapping": "pred_raw_qm", "B2_global_lgbm": "pred_B2",
              "B2s_plus_static": "pred_B2s", "C1_regime_features": "pred_C1", "C2_regime_split": "pred_C2",
              "C1_oracle_DIAGNOSTIC": "pred_C1_oracle"}
    thresholds = load_yaml("thresholds.yaml")["verification_thresholds_mm"]
    ev = mcfg["evaluation"]
    results = compare(valid, models, settings.domain, thresholds, ev["fss_threshold_mm"], ev["fss_window_cells"])

    valid["local_regime"] = valid[LOCAL_COLS].to_numpy().argmax(axis=1)
    valid["local_regime"] = valid["local_regime"].map({v: k for k, v in LOCAL_CODES.items()})
    key_models = {k: models[k] for k in ("A_raw_nwp", "B2s_plus_static", "C1_regime_features", "C2_regime_split",
                                         "C1_oracle_DIAGNOSTIC")}
    by_regime = {"observed_synoptic": breakdown(valid, key_models, "obs_regime"),
                 "local": breakdown(valid, key_models, "local_regime")}

    bs = bcfg["bootstrap"]
    ets = lambda f, o: calculate_ets(f, o, 64.5)
    significance = {}
    for a, b in (("pred_B2s", "pred_C1"), ("pred_B2s", "pred_C2"), ("pred_B2", "pred_B2s")):
        significance[f"{b} vs {a}"] = {
            "rmse": block_bootstrap_difference(valid, calculate_rmse, a, b, block_days=bs["block_days"], n_boot=bs["n_boot"]),
            "ets_64.5": block_bootstrap_difference(valid, ets, a, b, block_days=bs["block_days"], n_boot=bs["n_boot"],
                                                   higher_is_better=True),
        }
        log.info("bootstrap %s vs %s done", b, a)

    reports = PROJECT_ROOT / "reports"
    header = {"scope": f"VALIDATION years {va0}-{va1}; test years not used", "data_kind": "real",
              "regime_probabilities": f"{bcfg['classifier']} classifier; out-of-fold for training years",
              "note": "C1_oracle uses observed regimes and is NOT an operational forecast"}
    payload = {**header, "results": results, "by_regime": by_regime, "significance_vs_baseline": significance,
               "c2_routing_validation": c2.routing(valid), "c2_fallback_regimes": c2.fallback_regimes}
    (reports / "phase6_validation.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    md = to_markdown(results, f"Phase 6 regime-aware correction — validation {va0}–{va1} (REAL DATA)")
    md += "\n## Significance (paired 5-day block bootstrap, 95% CI of improvement; positive = second model better)\n\n"
    md += "| Comparison | Metric | Improvement | 95% CI | Significant |\n|---|---|---|---|---|\n"
    for comp, metrics in significance.items():
        for m, r in metrics.items():
            md += (f"| {comp} | {m} | {r['improvement']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | "
                   f"{'yes' if r['significant'] else 'no'} |\n")
    (reports / "phase6_validation.md").write_text(md, encoding="utf-8")
    log.info("report written to %s", reports / "phase6_validation.md")

    out = model_dir / "regime_bc"
    c2.save(out / "C2")
    write_metadata(out / "C2", name="C2", model_type="one LightGBM Tweedie regressor per predicted synoptic regime",
                   features=c2.features, training_period=(tr0, tr1), validation_period=(va0, va1),
                   data_sources={"regime_probabilities": f"{bcfg['classifier']} out-of-fold"},
                   validation_metrics={str(lead): {"rmse": r["C2_regime_split"]["rmse"], "bias": r["C2_regime_split"]["bias"]}
                                       for lead, r in results.items()},
                   extra={"fallback_regimes": c2.fallback_regimes})
    report_key = {"B2s": "B2s_plus_static", "C1": "C1_regime_features"}
    for name, key in report_key.items():
        fitted[name].save(out / name)
        write_metadata(out / name, name=name, model_type="LightGBM Tweedie regressor", features=feature_sets[name],
                       training_period=(tr0, tr1), validation_period=(va0, va1),
                       data_sources={"regime_probabilities": f"{bcfg['classifier']} out-of-fold"},
                       validation_metrics={str(lead): {"rmse": r[key]["rmse"], "bias": r[key]["bias"]}
                                           for lead, r in results.items()})


if __name__ == "__main__":
    main()
