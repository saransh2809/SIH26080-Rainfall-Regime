"""Sanity checks on the built training tables — TRAINING YEARS ONLY.

Validation and test years are never read here, so looking at these numbers cannot leak into
model or parameter choices. Output: data/processed/sanity_train.json.

Usage:
    python scripts/check_dataset.py
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from rainpp.config import load_settings, load_yaml

log = logging.getLogger("check_dataset")


def describe_lead(df: pd.DataFrame, thresholds: dict[str, float]) -> dict:
    obs, nwp = df["obs_precip_mm"].to_numpy(), df["nwp_precip_mm"].to_numpy()
    out = {
        "rows": len(df),
        "obs_mean_mm": float(obs.mean()),
        "nwp_mean_mm": float(nwp.mean()),
        "mean_error_nwp_minus_obs_mm": float((nwp - obs).mean()),
        "pearson_r": float(np.corrcoef(nwp, obs)[0, 1]),
        "obs_rain_day_frac_ge_2.5mm": float((obs >= 2.5).mean()),
        "nwp_rain_day_frac_ge_2.5mm": float((nwp >= 2.5).mean()),
    }
    for name, t in thresholds.items():
        out[f"obs_frac_ge_{name}"] = float((obs >= t).mean())
        out[f"nwp_frac_ge_{name}"] = float((nwp >= t).mean())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    thresholds = load_yaml("thresholds.yaml")["thresholds_mm"]
    processed = settings.paths.data_dir / "processed"
    first, last = settings.split.train
    years = list(range(first, last + 1))

    df = pd.concat([pd.read_parquet(processed / f"table_{y}.parquet") for y in years], ignore_index=True)
    problems = []
    if df.isna().any().any():
        problems.append(f"NaN in columns: {df.columns[df.isna().any()].tolist()}")
    if (df[["obs_precip_mm", "nwp_precip_mm"]] < 0).any().any():
        problems.append("negative rainfall present")
    rows_per_date = df.groupby(["valid_date", "lead_day"]).size()
    if rows_per_date.nunique() > 1:
        problems.append(f"observed-cell count varies by date: {rows_per_date.min()}-{rows_per_date.max()}")

    report = {
        "scope": f"TRAINING YEARS ONLY {first}-{last}; raw NWP characterisation, not a model evaluation",
        "data_kind": "real",
        "total_rows": len(df),
        "cells_per_day": int(rows_per_date.median()),
        "problems": problems,
        "by_lead_day": {int(ld): describe_lead(g, thresholds) for ld, g in df.groupby("lead_day")},
    }
    (processed / "sanity_train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
