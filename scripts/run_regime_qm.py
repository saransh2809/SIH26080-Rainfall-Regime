"""Phase 6b: regime-conditioned quantile mapping (C3) vs global quantile mapping (B1), VALIDATION years.

Quantile mapping keeps heavy-rain frequency, unlike the mean-regression correctors of Phase 6, so this
tests the regime hypothesis in the family that can actually help heavy rain.
  B1         global quantile mapping (Phase 4 model, reloaded)
  C3         one quantile mapping per PREDICTED synoptic regime (out-of-fold for training years)
  C3_oracle  same with OBSERVED regimes — diagnostic upper bound only

Usage:
    python scripts/run_regime_qm.py
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import xarray as xr

from rainpp.config import PROJECT_ROOT, load_settings, load_yaml
from rainpp.models.quantile_mapping import QuantileMapping
from rainpp.models.regime_bc import RegimeQuantileMapping
from rainpp.regimes.augment import load_augmented, synoptic_probabilities
from rainpp.regimes.local import static_terrain
from rainpp.verification.bootstrap import block_bootstrap_difference
from rainpp.verification.metrics import calculate_ets, calculate_rmse
from rainpp.verification.report import compare, score, to_markdown

log = logging.getLogger("run_regime_qm")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    mcfg, rcfg = load_yaml("models.yaml"), load_yaml("regimes.yaml")
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
    keep = ["valid_date", "lead_day", "lat", "lon", "nwp_precip_mm", "obs_precip_mm", "pred_regime", "obs_regime"]
    train = load_augmented(data_dir, train_years, terrain, rcfg["local"], probs, labels)[keep]
    valid = load_augmented(data_dir, valid_years, terrain, rcfg["local"], probs, labels)[keep]

    valid["pred_B1"] = QuantileMapping.load(model_dir / "quantile_mapping" / "quantile_mapping.npz").predict(valid)
    c3 = RegimeQuantileMapping("pred_regime").fit(train)
    valid["pred_C3"] = c3.predict(valid)
    oracle = RegimeQuantileMapping("obs_regime").fit(train)
    valid["pred_C3_oracle"] = oracle.predict(valid)
    log.info("C3 fallback regimes: %s | oracle fallback: %s", c3.fallback_regimes, oracle.fallback_regimes)

    models = {"A_raw_nwp": "nwp_precip_mm", "B1_quantile_mapping": "pred_B1", "C3_regime_qm": "pred_C3",
              "C3_oracle_DIAGNOSTIC": "pred_C3_oracle"}
    thresholds = load_yaml("thresholds.yaml")["verification_thresholds_mm"]
    ev, bs = mcfg["evaluation"], mcfg["regime_correction"]["bootstrap"]
    results = compare(valid, models, settings.domain, thresholds, ev["fss_threshold_mm"], ev["fss_window_cells"])

    significance = {}
    for t in (64.5, 115.6):
        ets = lambda f, o, t=t: calculate_ets(f, o, t)
        significance[f"ets_{t}"] = block_bootstrap_difference(valid, ets, "pred_B1", "pred_C3", higher_is_better=True,
                                                              **bs)
    significance["rmse"] = block_bootstrap_difference(valid, calculate_rmse, "pred_B1", "pred_C3", **bs)

    by_regime = {}
    for regime, group in valid.groupby("obs_regime"):
        by_regime[str(regime)] = {name: {"rmse": s["rmse"], "ets_64.5": s["categorical"]["64.5"]["ets"],
                                         "frequency_bias_64.5": s["categorical"]["64.5"]["frequency_bias"]}
                                  for name, col in models.items() for s in [score(group, col, None, [64.5], [], [])]}

    reports = PROJECT_ROOT / "reports"
    payload = {"scope": f"VALIDATION years {va0}-{va1}; test years not used", "data_kind": "real",
               "note": "C3_oracle uses observed regimes and is NOT an operational forecast",
               "results": results, "significance_C3_vs_B1": significance, "by_observed_regime": by_regime,
               "c3_fallback_regimes": c3.fallback_regimes}
    (reports / "phase6b_validation.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    md = to_markdown(results, f"Phase 6b regime-conditioned quantile mapping — validation {va0}–{va1} (REAL DATA)")
    md += "\n## C3 vs B1 (paired 5-day block bootstrap, 95% CI; positive = C3 better)\n\n"
    md += "| Metric | Improvement | 95% CI | Significant |\n|---|---|---|---|\n"
    for m, r in significance.items():
        md += (f"| {m} | {r['improvement']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | "
               f"{'yes' if r['significant'] else 'no'} |\n")
    (reports / "phase6b_validation.md").write_text(md, encoding="utf-8")
    log.info("report written to %s", reports / "phase6b_validation.md")

    out = model_dir / "regime_qm"
    for regime, qm in c3.by_regime.items():
        qm.save(out / regime)
    c3.global_qm.save(out / "GLOBAL")


if __name__ == "__main__":
    main()
