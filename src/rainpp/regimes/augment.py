"""Add regime information to the correction tables (shared by Phases 6 and 7).

Columns added per row: static terrain (elevation_m, coast_km), local regime one-hot (local_*),
predicted synoptic-regime probabilities (p_*) and hard prediction (pred_regime), and the OBSERVED
synoptic regime (obs_*, obs_regime) — the latter only for diagnostics and scoring, never as an
operational input.

Training-year probabilities are out-of-fold: the classifier is refit without the row's year, so
downstream models learn from regime information as imperfect as what they receive in operation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from rainpp.regimes.classifier import CLASSES, ForestClassifier, LGBMRegimeClassifier
from rainpp.regimes.local import LOCAL_CODES, local_regime_codes

log = logging.getLogger(__name__)

PROB_COLS = [f"p_{c}" for c in CLASSES]
LOCAL_COLS = [f"local_{c}" for c in LOCAL_CODES]
ORACLE_COLS = [f"obs_{c}" for c in CLASSES]


def make_classifier(name: str, features: list[str], cfg: dict, rounds: int):
    if name == "RF":
        return ForestClassifier(features)
    if name == "LGBM":
        return LGBMRegimeClassifier(features, cfg["lgbm_params"], rounds)
    raise ValueError(f"unsupported classifier for correction: {name}")


def synoptic_probabilities(regime_table: pd.DataFrame, train_years: range, other_years: range,
                           name: str, cfg: dict, rounds: int) -> pd.DataFrame:
    """Out-of-fold probabilities for training years; full-training-fit probabilities for other_years."""
    features = cfg["features"]
    year = regime_table["valid_date"].dt.year
    labelled = regime_table["synoptic_regime"].notna()

    def fit_predict(fit_rows: pd.Series, predict_rows: pd.Series) -> pd.DataFrame:
        model = make_classifier(name, features, cfg, rounds).fit(
            regime_table[fit_rows], regime_table.loc[fit_rows, "synoptic_regime"])
        rows = regime_table[predict_rows]
        return model.predict_proba(rows).set_axis(rows.index)

    parts = [fit_predict(year.isin(train_years) & (year != held_out) & labelled, year == held_out)
             for held_out in train_years]
    parts.append(fit_predict(year.isin(train_years) & labelled, year.isin(other_years)))
    proba = pd.concat(parts).sort_index()
    out = regime_table.loc[proba.index, ["init_time", "lead_day"]].copy()
    out[PROB_COLS] = proba[CLASSES].to_numpy()
    out["pred_regime"] = proba[CLASSES].idxmax(axis=1).to_numpy()
    return out


def add_regime_columns(table: pd.DataFrame, fields: xr.Dataset, terrain: xr.Dataset, local_cfg: dict,
                       probs: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    pts = {"lat": xr.DataArray(table["lat"].to_numpy(), dims="row"),
           "lon": xr.DataArray(table["lon"].to_numpy(), dims="row")}
    table["elevation_m"] = terrain["elevation_m"].sel(**pts).to_numpy().astype(np.float32)
    table["coast_km"] = terrain["coast_km"].sel(**pts).to_numpy().astype(np.float32)
    codes = local_regime_codes(fields, terrain, local_cfg)
    local = codes.sel(init_time=xr.DataArray(table["init_time"].to_numpy(), dims="row"),
                      lead_day=xr.DataArray(table["lead_day"].to_numpy(), dims="row"), **pts).to_numpy()
    for name, code in LOCAL_CODES.items():
        table[f"local_{name}"] = (local == code).astype(np.float32)
    table = table.merge(probs, on=["init_time", "lead_day"], how="left", validate="many_to_one")
    if table[PROB_COLS].isna().any().any():
        raise ValueError("rows without regime probabilities")
    table[PROB_COLS] = table[PROB_COLS].astype(np.float32)
    observed = labels.reindex(table["valid_date"]).to_numpy()
    for c in CLASSES:
        table[f"obs_{c}"] = (observed == c).astype(np.float32)
    table["obs_regime"] = pd.Series(observed).fillna("UNLABELLED").to_numpy()
    return table


def checkerboard_mask(lat: np.ndarray, lon: np.ndarray, resolution_deg: float, every: int) -> np.ndarray:
    """Keep 1 in `every` cells in a regular diagonal pattern (every=1 keeps all)."""
    i = np.rint(np.asarray(lat) / resolution_deg).astype(int)
    j = np.rint(np.asarray(lon) / resolution_deg).astype(int)
    return (i + j) % every == 0


def load_augmented(data_dir: Path, years: range, terrain: xr.Dataset, local_cfg: dict,
                   probs: pd.DataFrame, labels: pd.Series, cell_every: int = 1,
                   columns: list[str] | None = None, resolution_deg: float = 0.25) -> pd.DataFrame:
    """Augmented tables for `years`.

    cell_every > 1 keeps a checkerboard subset of cells (memory control for model FITTING only; evaluation
    always uses every cell). `columns` drops everything else after augmentation.
    """
    parts = []
    for year in years:
        table = pd.read_parquet(data_dir / "processed" / f"table_{year}.parquet")
        if cell_every > 1:
            table = table[checkerboard_mask(table["lat"], table["lon"], resolution_deg, cell_every)]
        fields = xr.open_dataset(data_dir / "interim" / "gefs_fields" / f"fields_{year}.nc").load()
        table = add_regime_columns(table.reset_index(drop=True), fields, terrain, local_cfg, probs, labels)
        parts.append(table[columns] if columns else table)
        log.info("%d: regime columns added (%d rows)", year, len(table))
    return pd.concat(parts, ignore_index=True)
