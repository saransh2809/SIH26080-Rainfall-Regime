"""Phase 8: U-Net rainfall corrector vs the tree correctors, on VALIDATION years.

The deep-learning experiment the problem statement allows only against a baseline: A raw, B2 global LightGBM and
C1 regime-aware LightGBM are scored on exactly the same validation rows as the U-Net, with a paired block bootstrap
for U-Net vs C1. Regime probabilities are out-of-fold for training years, as for C1. Test years are never read.
The U-Net is saved for reference; products keep using C1 unless this report shows a clear, significant gain.

Usage:
    python scripts/run_cnn_correction.py
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
import xarray as xr

from rainpp.config import PROJECT_ROOT, load_settings, load_yaml
from rainpp.data.align import valid_dates
from rainpp.data.features import fit_climatology
from rainpp.data.sources.gefs import open_interim
from rainpp.data.sources.imd import IMDGridded
from rainpp.models.cnn_bc import CNNCorrector
from rainpp.models.global_bc import LGBMCorrector
from rainpp.models.registry import write_metadata
from rainpp.regimes.augment import LOCAL_COLS, PROB_COLS, load_augmented, synoptic_probabilities
from rainpp.regimes.local import static_terrain
from rainpp.verification.bootstrap import block_bootstrap_difference
from rainpp.verification.metrics import calculate_ets, calculate_rmse
from rainpp.verification.report import compare, to_markdown

log = logging.getLogger("run_cnn_correction")


def year_arrays(data_dir, year: int, imd: IMDGridded, probs: pd.DataFrame) -> dict:
    """One sample per (init, lead): forecast map, valid-day observation map (NaN off land) and scalars."""
    fc = open_interim(data_dir / "interim" / "gefs" / f"gefs_apcp_{year}.nc")
    vd = valid_dates(fc)
    obs = imd.load(date(year, 1, 1), date(year, 12, 31) + pd.Timedelta(days=5))["precip_mm"]
    rain = fc["precip_mm"].isel(member=0)
    inits, leads = fc.init_time.values, fc.lead_day.values
    n = len(inits) * len(leads)
    forecast = rain.values.reshape(n, rain.sizes["lat"], rain.sizes["lon"]).astype(np.float32)
    days = pd.DatetimeIndex(vd.values.ravel())
    target = obs.reindex(time=days).values.astype(np.float32)
    keys = pd.DataFrame({"init_time": np.repeat(pd.DatetimeIndex(inits).floor("D"), len(leads)),
                         "lead_day": np.tile(leads, len(inits))})
    p = keys.merge(probs, on=["init_time", "lead_day"], how="left", validate="one_to_one")
    if p[PROB_COLS].isna().any().any():
        raise ValueError(f"{year}: samples without regime probabilities")
    doy = days.dayofyear.to_numpy()
    scalars = np.column_stack([keys["lead_day"], np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                               p[PROB_COLS].to_numpy()]).astype(np.float32)
    return {"forecast": forecast, "target": target, "doy": doy, "scalars": scalars, "keys": keys,
            "lat": rain.lat.values, "lon": rain.lon.values}


def stack(parts: list[dict]) -> dict:
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("forecast", "target", "doy", "scalars")}
    out["keys"] = pd.concat([p["keys"] for p in parts], ignore_index=True)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(max(1, torch.get_num_threads()))
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
    probs["init_time"] = pd.to_datetime(probs["init_time"]).dt.floor("D")

    imd = IMDGridded(data_dir / "raw" / "imd")
    climatology = fit_climatology(imd.load(date(tr0, 1, 1), date(tr1, 12, 31))["precip_mm"])
    static_nc = xr.open_dataset(data_dir / "interim" / "static" / "static.nc").load()
    terrain = static_terrain(static_nc, settings.domain.resolution_deg)
    train = stack([year_arrays(data_dir, y, imd, probs) for y in train_years])
    valid_parts = [year_arrays(data_dir, y, imd, probs) for y in valid_years]
    valid = stack(valid_parts)
    lat, lon = valid_parts[0]["lat"], valid_parts[0]["lon"]
    on_grid = {"lat": lat, "lon": lon, "method": "nearest"}
    static = np.nan_to_num(np.stack([terrain["elevation_m"].sel(**on_grid).values,
                                     terrain["coast_km"].sel(**on_grid).values,
                                     static_nc["land_fraction"].sel(**on_grid).values])).astype(np.float32)
    clim = np.nan_to_num(climatology.sel(lat=lat, lon=lon, method="nearest").values).astype(np.float32)
    for d in (train, valid):
        d["static"], d["climatology"] = static, clim
    log.info("train samples %d, validation samples %d, grid %s", len(train["forecast"]), len(valid["forecast"]),
             train["forecast"].shape[1:])

    start = time.time()
    stop = (train["keys"]["init_time"].dt.year == tr1).to_numpy()
    cnn = CNNCorrector().fit(train, stop)
    log.info("U-Net: %d epochs chosen on %d, trained in %.0f min", cnn.epochs, tr1, (time.time() - start) / 60)
    pred = cnn.predict(valid)

    needed = list(dict.fromkeys(mcfg["features"] + mcfg["regime_correction"]["static_features"] + PROB_COLS
                                + LOCAL_COLS + ["init_time", "valid_date", "lead_day", "lat", "lon", "obs_precip_mm"]))
    rows = load_augmented(data_dir, valid_years, terrain, rcfg["local"], probs, labels, columns=needed)
    rows["pred_B2"] = LGBMCorrector.load(model_dir / "global_lgbm").predict(rows)
    rows["pred_C1"] = LGBMCorrector.load(model_dir / "regime_bc" / "C1").predict(rows)
    sample = pd.Series(np.arange(len(valid["keys"])),
                       index=pd.MultiIndex.from_frame(valid["keys"].assign(init_time=valid["keys"]["init_time"])))
    s_idx = sample.reindex(pd.MultiIndex.from_arrays([pd.to_datetime(rows["init_time"]).dt.floor("D"),
                                                      rows["lead_day"]])).to_numpy()
    i_idx = np.searchsorted(lat, rows["lat"].to_numpy())
    j_idx = np.searchsorted(lon, rows["lon"].to_numpy())
    rows["pred_CNN"] = pred[s_idx, i_idx, j_idx]

    models = {"A_raw_nwp": "nwp_precip_mm", "B2_global_lgbm": "pred_B2", "C1_regime_features": "pred_C1",
              "D_unet": "pred_CNN"}
    thresholds = load_yaml("thresholds.yaml")["verification_thresholds_mm"]
    ev = mcfg["evaluation"]
    results = compare(rows, models, settings.domain, thresholds, ev["fss_threshold_mm"], ev["fss_window_cells"])
    bs = mcfg["regime_correction"]["bootstrap"]
    ets = lambda f, o: calculate_ets(f, o, 64.5)
    significance = {"pred_CNN vs pred_C1": {
        "rmse": block_bootstrap_difference(rows, calculate_rmse, "pred_C1", "pred_CNN", **bs),
        "ets_64.5": block_bootstrap_difference(rows, ets, "pred_C1", "pred_CNN", higher_is_better=True, **bs)}}

    reports = PROJECT_ROOT / "reports"
    payload = {"scope": f"VALIDATION years {va0}-{va1}; test years not used", "data_kind": "real",
               "epochs": cnn.epochs, "results": results, "significance": significance}
    (reports / "phase8_validation.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    md = to_markdown(results, f"Phase 8 U-Net corrector vs tree correctors — validation {va0}–{va1} (REAL DATA)")
    md += "\n## Significance (paired 5-day block bootstrap; positive = U-Net better than C1)\n\n"
    md += "| Metric | Improvement | 95% CI | Significant |\n|---|---|---|---|\n"
    for m, r in significance["pred_CNN vs pred_C1"].items():
        md += (f"| {m} | {r['improvement']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | "
               f"{'yes' if r['significant'] else 'no'} |\n")
    (reports / "phase8_validation.md").write_text(md, encoding="utf-8")
    log.info("report written to %s", reports / "phase8_validation.md")

    out = model_dir / "cnn_bc"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": cnn.net.state_dict(), "width": cnn.width, "epochs": cnn.epochs,
                "static_mean": cnn.static_mean, "static_sd": cnn.static_sd,
                "scalar_mean": cnn.scalar_mean, "scalar_sd": cnn.scalar_sd}, out / "unet.pt")
    write_metadata(out, name="D_unet", model_type="U-Net (PyTorch), Tweedie deviance", features=[
        "log1p forecast map", "log1p observed climatology map", "elevation", "coast distance", "land",
        "lead_day", "doy_sin", "doy_cos", *PROB_COLS], training_period=(tr0, tr1), validation_period=(va0, va1),
        data_sources={"regime_probabilities": "LGBM out-of-fold"},
        validation_metrics={str(ld): {"rmse": r["D_unet"]["rmse"], "bias": r["D_unet"]["bias"]}
                            for ld, r in results.items()}, extra={"epochs": cnn.epochs})


if __name__ == "__main__":
    main()
