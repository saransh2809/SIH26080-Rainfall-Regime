"""How much does 0.5° input (operational GEFS winds and MSL pressure) change the regime classifier?

Operational GEFS publishes 850 hPa wind and MSL pressure at 0.5° only. On reforecast validation days
this script computes the classifier's synoptic features twice — from the native 0.25° fields, and from
the same fields subsampled to 0.5° and interpolated back exactly as live mode does — then compares
feature values and the classifier's regime probabilities. Validation years only; test years untouched.

Usage:
    python scripts/check_operational_fields.py --every 6
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from rainpp.config import load_settings, load_yaml
from rainpp.data.sources.gefs import ECCODES_LOCK, open_interim
from rainpp.data.sources.gefs_fields import (
    FIELDS,
    GEFSFields,
    block_mean,
    crop_box,
    find_message,
    mid_window_hour,
    relative_vorticity,
)
from rainpp.data.sources.gefs_operational import upsample_half_degree
from rainpp.data.sources.imd import IMDGridded
from rainpp.regimes.classifier import CLASSES
from rainpp.regimes.features import build_classifier_table

log = logging.getLogger("check_operational_fields")
DEGRADED = {"u850", "v850", "mslp_hpa"}
FIELD_FEATURES = ["llj_u850", "central_u850", "trough_vort_max", "trough_vort_mean", "trough_mslp_min",
                  "mslp_south_minus_trough", "core_pwat_mm"]


def decode(data: bytes) -> np.ndarray:
    import eccodes

    with ECCODES_LOCK:
        gid = eccodes.codes_new_from_message(data)
        try:
            return eccodes.codes_get_values(gid).reshape(721, 1440)
        finally:
            eccodes.codes_release(gid)


def both_versions(src: GEFSFields, init: date, lead_days: list[int]) -> tuple[dict, dict]:
    _, lats, _ = crop_box(np.zeros((721, 1440)), src.box)
    native = {spec.name: [] for spec in FIELDS}
    coarse = {spec.name: [] for spec in FIELDS}
    for spec in FIELDS:
        path = src.file_path(init, spec.file)
        idx = src.fs.cat(path + ".idx").decode()
        for ld in lead_days:
            start, end = find_message(idx, spec.variable, spec.level, mid_window_hour(ld))
            full = decode(src.fs.cat_file(path, start=start, end=end)) * spec.scale
            native[spec.name].append(crop_box(full, src.box)[0])
            degraded = upsample_half_degree(full[::2, ::2]) if spec.name in DEGRADED else full
            coarse[spec.name].append(crop_box(degraded, src.box)[0])
    out = []
    for raw in (native, coarse):
        fields = {name: np.stack([block_mean(f) for f in fs]) for name, fs in raw.items()}
        fields["vort850_1e5"] = np.stack([block_mean(relative_vorticity(u, v, lats, 0.25)) * 1e5
                                          for u, v in zip(raw["u850"], raw["v850"], strict=True)])
        out.append(fields)
    return out[0], out[1]


def to_dataset(results: list[dict], inits: list[date], lead_days: list[int], box) -> xr.Dataset:
    _, lats, lons = crop_box(np.zeros((721, 1440)), box)
    lat1, lon1 = block_mean(lats[:, None].repeat(4, 1))[:, 0], block_mean(lons[None, :].repeat(4, 0))[0]
    return xr.Dataset({name: (("init_time", "lead_day", "lat", "lon"), np.stack([r[name] for r in results]))
                       for name in results[0]},
                      coords={"init_time": pd.to_datetime(inits), "lead_day": lead_days, "lat": lat1, "lon": lon1})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=int, default=6, help="use every n-th JJAS day of the validation years")
    args = parser.parse_args()
    settings = load_settings()
    cfg = load_yaml("regimes.yaml")
    data_dir = settings.paths.data_dir
    v0, v1 = settings.split.validation
    lead_days = list(settings.forecast.lead_days)

    rows = pd.read_parquet(data_dir / "processed" / "regime_features.parquet")
    rows = rows[rows["init_time"].dt.year.between(v0, v1) & rows["init_time"].dt.month.isin(settings.season.months)]
    inits = sorted(rows["init_time"].dt.date.unique())[:: args.every]
    log.info("%d validation initialisations", len(inits))

    src = GEFSFields()
    with ThreadPoolExecutor(8) as pool:
        pairs = list(pool.map(lambda d: both_versions(src, d, lead_days), inits))
    native = to_dataset([p[0] for p in pairs], inits, lead_days, src.box)
    coarse = to_dataset([p[1] for p in pairs], inits, lead_days, src.box)

    rain = xr.concat([open_interim(data_dir / "interim" / "gefs" / f"gefs_apcp_{y}.nc") for y in range(v0, v1 + 1)],
                     dim="init_time").sel(init_time=pd.to_datetime(inits))
    obs = IMDGridded(data_dir / "raw" / "imd").load(date(v0, 6, 1), date(v0, 6, 1))
    land_mask = obs["precip_mm"].isel(time=0).notnull()
    core_box = cfg["synoptic"]["active_break"]["core_zone_box"]
    empty = pd.Series(dtype=float)
    feats = {}
    for name, fields in (("native", native), ("coarse", coarse)):
        t = build_classifier_table(fields, rain, land_mask, cfg["classifier_features"], core_box, empty)
        feats[name] = t.set_index(["init_time", "lead_day"])[FIELD_FEATURES]

    base = rows.set_index(["init_time", "lead_day"]).loc[feats["native"].index]
    consistency = float((feats["native"] - base[FIELD_FEATURES]).abs().max().max())
    diff = feats["coarse"] - feats["native"]
    sd = base[FIELD_FEATURES].std()
    feature_report = {c: {"mean_abs_diff": float(diff[c].abs().mean()), "mean_diff": float(diff[c].mean()),
                          "feature_sd": float(sd[c]), "mean_abs_diff_in_sd": float(diff[c].abs().mean() / sd[c])}
                      for c in FIELD_FEATURES}

    booster = lgb.Booster(model_file=str(settings.paths.model_dir / "regime_classifier" / "lgbm.txt"))
    features = load_yaml("models.yaml")["regime_classifier"]["features"]
    probs = {}
    for name in ("native", "coarse"):
        x = base.copy()
        x[FIELD_FEATURES] = feats[name]
        probs[name] = pd.DataFrame(booster.predict(x.reset_index()[features].astype(np.float32)), columns=CLASSES,
                                   index=x.index)
    agree = float((probs["native"].idxmax(axis=1) == probs["coarse"].idxmax(axis=1)).mean())
    max_dp = float((probs["native"] - probs["coarse"]).abs().max(axis=1).mean())

    report = {"validation_years": [v0, v1], "initialisations": len(inits), "rows": len(base),
              "native_recomputation_max_abs_error": consistency,
              "regime_argmax_agreement": agree, "mean_max_abs_probability_change": max_dp,
              "features": feature_report}
    out = settings.paths.model_dir.parent / "reports" / "operational_fields_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
