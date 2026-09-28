"""One-time evaluation of the FROZEN models on data never used for training, tuning or model choice.

  test         reforecast test years (settings.split.test) through the archive pipeline
  operational  NOAA GEFSv12 operational runs (settings.split.operational_test) through the live pipeline
               (forecast-only classifier), i.e. exactly what live mode would have issued

Every number comes from the same code that builds dashboard products (ProductBuilder.predict). Scores are
reported whether they favour the system or not. A lock file with the hashes of the evaluated model files
prevents silent re-runs; --rerun-reason allows one and records why.

Usage:
    python scripts/run_final_evaluation.py
    python scripts/run_final_evaluation.py --datasets operational --rerun-reason "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, date, datetime

import pandas as pd
import xarray as xr

from rainpp.config import PROJECT_ROOT, load_settings, load_yaml
from rainpp.data.sources.gefs import open_interim
from rainpp.data.sources.imd import IMDGridded
from rainpp.models.global_bc import LGBMCorrector
from rainpp.models.regime_bc import RegimeSplitCorrector
from rainpp.products import ProductBuilder
from rainpp.regimes.local import LOCAL_CODES
from rainpp.verification.bootstrap import block_bootstrap_difference
from rainpp.verification.metrics import calculate_ets, calculate_rmse
from rainpp.verification.probabilistic import brier_score, probability_scores
from rainpp.verification.report import compare, score, to_markdown

log = logging.getLogger("run_final_evaluation")
REPORTS = PROJECT_ROOT / "reports"
LOCK = REPORTS / "final_evaluation.lock.json"
MODELS = {"A_raw_nwp": "nwp_precip_mm", "B1_quantile_mapping": "qm_mm", "B2_global_lgbm": "pred_B2",
          "C1_regime_features": "corrected_mm", "C2_regime_split": "pred_C2"}
KEEP = ["init_time", "valid_date", "lead_day", "lat", "lon", "obs_precip_mm", "obs_regime", "local_regime",
        *MODELS.values(), "p_heavy", "p_very_heavy"]


def model_hashes(model_dir) -> dict[str, str]:
    return {str(p.relative_to(model_dir)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model_dir.rglob("*")) if p.is_file() and p.suffix in {".txt", ".json", ".npz", ".joblib"}}


def with_extra_models(table: pd.DataFrame, b2: LGBMCorrector, c2: RegimeSplitCorrector) -> pd.DataFrame:
    table["pred_B2"] = b2.predict(table)
    table["pred_C2"] = c2.predict(table)
    return table


def test_rows(builder: ProductBuilder, years: range, labels: pd.Series, b2, c2, t_heavy, t_very) -> pd.DataFrame:
    parts = []
    for year in years:
        table, regime_rows, fields = builder.archive_inputs(year)
        table, _ = builder.predict(table, regime_rows, fields, live=False, labels=labels)
        table = with_extra_models(table, b2, c2)
        parts.append(table[KEEP + [f"clim_p_ge_{t_heavy}", f"clim_p_ge_{t_very}"]])
        log.info("test %d: %d rows", year, len(table))
    return pd.concat(parts, ignore_index=True)


def operational_rows(builder: ProductBuilder, years: range, labels: pd.Series, b2, c2, t_heavy, t_very) -> pd.DataFrame:
    settings = builder.s
    op_dir = settings.paths.data_dir / "interim" / "gefs_operational"
    imd = IMDGridded(settings.paths.data_dir / "raw" / "imd")
    parts = []
    for year in years:
        fc = open_interim(op_dir / f"gefs_apcp_{year}.nc")
        fields = xr.open_dataset(op_dir / f"fields_{year}.nc").load()
        observed = imd.load(date(year, 1, 1), date(year, 12, 31))
        table, regime_rows = builder.live_inputs(fc, fields, observed)
        table = table[table["obs_precip_mm"].notna()].reset_index(drop=True)
        table, _ = builder.predict(table, regime_rows, fields, live=True, labels=labels)
        table = with_extra_models(table, b2, c2)
        parts.append(table[KEEP + [f"clim_p_ge_{t_heavy}", f"clim_p_ge_{t_very}"]])
        log.info("operational %d: %d rows", year, len(table))
    return pd.concat(parts, ignore_index=True)


def by_regime(rows: pd.DataFrame, column: str) -> dict:
    out = {}
    for regime, group in rows.groupby(column):
        out[str(regime)] = {"rows": len(group), "observed_heavy_events": int((group.obs_precip_mm >= 64.5).sum())}
        for name, col in MODELS.items():
            s = score(group, col, None, [64.5], [], [])
            out[str(regime)][name] = {"rmse": s["rmse"], "bias": s["bias"], "ets_64.5": s["categorical"]["64.5"]["ets"]}
    return out


def evaluate(rows: pd.DataFrame, settings, thresholds_mm: dict, bootstrap: dict) -> dict:
    ev = load_yaml("models.yaml")["evaluation"]
    vthr = load_yaml("thresholds.yaml")["verification_thresholds_mm"]
    deterministic = compare(rows, MODELS, settings.domain, vthr, ev["fss_threshold_mm"], ev["fss_window_cells"])

    probabilistic = {}
    for name in ("heavy", "very_heavy"):
        t = float(thresholds_mm[name])
        outcome = (rows["obs_precip_mm"].to_numpy() >= t).astype(float)
        preds = {"CLIM": rows[f"clim_p_ge_{t}"].to_numpy(), "P_regime": rows[f"p_{name}"].to_numpy(),
                 "RAW01": (rows["nwp_precip_mm"].to_numpy() >= t).astype(float),
                 "QM01": (rows["qm_mm"].to_numpy() >= t).astype(float)}
        probabilistic[name] = {"threshold_mm": t, "all": {m: probability_scores(p, outcome, preds["CLIM"])
                                                          for m, p in preds.items()}}
        probabilistic[name]["by_lead"] = {
            int(ld): {m: {k: v for k, v in probability_scores(p[sel], outcome[sel], preds["CLIM"][sel]).items()
                          if k != "reliability"} for m, p in preds.items()}
            for ld in sorted(rows["lead_day"].unique()) for sel in [(rows["lead_day"] == ld).to_numpy()]}

    ets = lambda f, o: calculate_ets(f, o, 64.5)
    significance = {}
    for a, b in (("nwp_precip_mm", "corrected_mm"), ("pred_B2", "corrected_mm"), ("pred_B2", "pred_C2"),
                 ("nwp_precip_mm", "qm_mm")):
        significance[f"{b} vs {a}"] = {
            "rmse": block_bootstrap_difference(rows, calculate_rmse, a, b, **bootstrap),
            "ets_64.5": block_bootstrap_difference(rows, ets, a, b, higher_is_better=True, **bootstrap)}
    for name in ("heavy", "very_heavy"):
        t = float(thresholds_mm[name])
        rows[f"_clim_{name}"] = rows[f"clim_p_ge_{t}"]
        brier = lambda f, o, t=t: brier_score(f, (o >= t).astype(float))
        significance[f"P_regime vs CLIM ({name})"] = {
            "brier": block_bootstrap_difference(rows, brier, f"_clim_{name}", f"p_{name}", **bootstrap)}
    names = {v: k for k, v in LOCAL_CODES.items()}
    rows["local_name"] = rows["local_regime"].map(names)
    return {"rows": len(rows), "valid_dates": [str(rows.valid_date.min().date()), str(rows.valid_date.max().date())],
            "deterministic": deterministic, "probabilistic": probabilistic, "significance": significance,
            "by_observed_regime": by_regime(rows, "obs_regime"), "by_local_regime": by_regime(rows, "local_name")}


def markdown(report: dict) -> str:
    md = ["# Final evaluation of the frozen models (REAL DATA, one-time)", "",
          (f"Run {report['run_utc']}. Models trained {report['trained_on'][0]}–{report['trained_on'][1]}, chosen on "
           f"validation {report['validated_on'][0]}–{report['validated_on'][1]}. Nothing below was used for "
           "training or model choice."), ""]
    if report.get("rerun_reason"):
        md += [f"**Re-run.** Reason given: {report['rerun_reason']}", ""]
    for key, ds in report["datasets"].items():
        md += [f"## {key}: {ds['description']}", "",
               f"Rows: {ds['rows']:,}; valid dates {ds['valid_dates'][0]} → {ds['valid_dates'][1]}.", ""]
        md.append(to_markdown(ds["deterministic"], f"{key} — deterministic scores"))
        md += ["", "### Heavy-rain probabilities (Brier skill vs climatology; AUC)", "",
               "| Target | Model | Brier | BSS vs CLIM | ROC AUC | Events |", "|---|---|---|---|---|---|"]
        for name, res in ds["probabilistic"].items():
            for m, s in res["all"].items():
                md.append(f"| {name} ≥{res['threshold_mm']} mm | {m} | {s['brier']:.5f} | "
                          f"{s['bss_vs_climatology']:.3f} | {s['roc_auc']:.3f} | {s['events']:,} |")
        md += ["", "### Significance (paired 5-day block bootstrap; positive = second model better)", "",
               "| Comparison | Metric | Improvement | 95% CI | Significant |", "|---|---|---|---|---|"]
        for comp, metrics in ds["significance"].items():
            for m, r in metrics.items():
                md.append(f"| {comp} | {m} | {r['improvement']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | "
                          f"{'yes' if r['significant'] else 'no'} |")
        md.append("")
    return "\n".join(md)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=["test", "operational"], default=["test", "operational"])
    parser.add_argument("--rerun-reason", help="required when a final evaluation has already been run")
    args = parser.parse_args()
    settings = load_settings()
    model_dir = settings.paths.model_dir
    hashes = model_hashes(model_dir)
    if LOCK.exists():
        previous = json.loads(LOCK.read_text(encoding="utf-8"))
        if not args.rerun_reason:
            raise SystemExit(f"final evaluation already run at {previous['run_utc']} (see reports/final_evaluation.md); "
                             "pass --rerun-reason to run again, which is recorded in the report")
        log.warning("re-running final evaluation: %s (models changed: %s)", args.rerun_reason,
                    previous["model_hashes"] != hashes)

    thresholds_mm = load_yaml("thresholds.yaml")["thresholds_mm"]
    t_heavy, t_very = float(thresholds_mm["heavy"]), float(thresholds_mm["very_heavy"])
    bootstrap = load_yaml("models.yaml")["regime_correction"]["bootstrap"]
    labels = pd.read_parquet(settings.paths.data_dir / "processed" / "regime_labels.parquet")["synoptic_regime"]
    builder = ProductBuilder(settings)
    b2 = LGBMCorrector.load(model_dir / "global_lgbm")
    c2 = RegimeSplitCorrector.load(model_dir / "regime_bc" / "C2")

    datasets = {}
    if "test" in args.datasets:
        t0, t1 = settings.split.test
        rows = test_rows(builder, range(t0, t1 + 1), labels, b2, c2, t_heavy, t_very)
        datasets["test"] = {"description": f"GEFSv12 reforecast, test years {t0}–{t1}, archive pipeline",
                            **evaluate(rows, settings, thresholds_mm, bootstrap)}
    if "operational" in args.datasets:
        if settings.split.operational_test is None:
            raise SystemExit("settings.split.operational_test is not configured")
        o0, o1 = settings.split.operational_test
        rows = operational_rows(builder, range(o0, o1 + 1), labels, b2, c2, t_heavy, t_very)
        datasets["operational"] = {
            "description": f"GEFSv12 operational forecasts, JJAS {o0}–{o1}, live pipeline (forecast-only classifier)",
            **evaluate(rows, settings, thresholds_mm, bootstrap)}

    report = {"run_utc": datetime.now(UTC).isoformat(timespec="seconds"), "data_kind": "real",
              "trained_on": list(settings.split.train), "validated_on": list(settings.split.validation),
              "rerun_reason": args.rerun_reason, "model_hashes": hashes, "datasets": datasets}
    (REPORTS / "final_evaluation.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    (REPORTS / "final_evaluation.md").write_text(markdown(report), encoding="utf-8")
    LOCK.write_text(json.dumps({"run_utc": report["run_utc"], "model_hashes": hashes,
                                "rerun_reason": args.rerun_reason}, indent=2), encoding="utf-8")
    log.info("final evaluation written to %s", REPORTS / "final_evaluation.md")


if __name__ == "__main__":
    main()
