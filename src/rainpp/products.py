"""Forecast products: everything the dashboard shows for one 00 UTC initialisation.

A product is built only from saved, versioned artifacts (models, calibrators, static fields) applied
to that initialisation's forecast data. Observed rainfall, when present, is stored as a separate,
labelled variable for verification — it is never an input to any product value.

Files (data/products/): {init}.nc (grids), {init}.districts.parquet, {init}.json (regime + explanation).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from rainpp.config import Settings, load_yaml
from rainpp.data.features import build_forecast_table, fit_climatology
from rainpp.data.sources.imd import IMDGridded
from rainpp.models.global_bc import LGBMCorrector
from rainpp.models.heavy_rain import IsotonicCalibrator, fit_event_climatology
from rainpp.models.quantile_mapping import QuantileMapping
from rainpp.regimes.augment import LOCAL_COLS, PROB_COLS, add_regime_columns
from rainpp.regimes.classifier import CLASSES
from rainpp.regimes.features import (
    add_forecast_anomaly,
    build_classifier_table,
    fit_forecast_core_stats,
)
from rainpp.regimes.local import LOCAL_CODES, static_terrain
from rainpp.spatial.districts import DistrictWeights

log = logging.getLogger(__name__)

GRID_VARS = ("raw_mm", "corrected_mm", "qm_mm", "p_heavy", "p_very_heavy", "local_regime", "observed_mm")


@dataclass
class Product:
    grids: xr.Dataset
    districts: pd.DataFrame
    summary: dict

    def save(self, directory: Path, init: date) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"{init:%Y-%m-%d}"
        self.grids.to_netcdf(stem.with_suffix(".nc"), encoding={v: {"zlib": True} for v in self.grids.data_vars})
        self.districts.to_parquet(stem.with_suffix(".districts.parquet"), index=False)
        stem.with_suffix(".json").write_text(json.dumps(self.summary, indent=2, default=float), encoding="utf-8")


class ProductBuilder:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.data_dir, self.model_dir = settings.paths.data_dir, settings.paths.model_dir
        self.thresholds = load_yaml("thresholds.yaml")["thresholds_mm"]

    # ---- artifacts (loaded once) ----
    @cached_property
    def terrain(self) -> xr.Dataset:
        static = xr.open_dataset(self.data_dir / "interim" / "static" / "static.nc").load()
        return static_terrain(static, self.s.domain.resolution_deg)

    @cached_property
    def classifier(self) -> lgb.Booster:
        return lgb.Booster(model_file=str(self.model_dir / "regime_classifier" / "lgbm.txt"))

    @cached_property
    def live_classifier(self) -> lgb.Booster:
        return lgb.Booster(model_file=str(self.model_dir / "regime_classifier" / "lgbm_live.txt"))

    @cached_property
    def classifier_features(self) -> list[str]:
        return load_yaml("models.yaml")["regime_classifier"]["features"]

    @cached_property
    def live_classifier_features(self) -> list[str]:
        cfg = load_yaml("models.yaml")["regime_classifier"]
        return [f for f in cfg["features"] if f not in cfg["live_excluded_features"]]

    @cached_property
    def corrector(self) -> LGBMCorrector:
        return LGBMCorrector.load(self.model_dir / "regime_bc" / "C1")

    @cached_property
    def qm(self) -> QuantileMapping:
        return QuantileMapping.load(self.model_dir / "quantile_mapping" / "quantile_mapping.npz")

    def heavy(self, target: str) -> tuple[lgb.Booster, list[str], callable]:
        directory = self.model_dir / "heavy_rain" / f"P_regime_{target}"
        spec = json.loads(next(directory.glob("heavy_rain_ge_*.json")).read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(next(directory.glob("heavy_rain_ge_*.txt"))))
        if load_yaml("models.yaml")["heavy_rain"]["calibrate"][target]:
            return booster, spec["features"], IsotonicCalibrator.load_transform(directory / "isotonic_calibration.json")
        return booster, spec["features"], lambda p: p

    @cached_property
    def train_obs(self) -> xr.DataArray:
        tr0, tr1 = self.s.split.train
        return IMDGridded(self.data_dir / "raw" / "imd").load(date(tr0, 1, 1), date(tr1, 12, 31))["precip_mm"]

    @cached_property
    def event_climatology(self) -> dict[float, xr.DataArray]:
        return {float(self.thresholds[t]): fit_event_climatology(self.train_obs, float(self.thresholds[t]))
                for t in ("heavy", "very_heavy")}

    @cached_property
    def rain_climatology(self) -> xr.DataArray:
        return fit_climatology(self.train_obs)

    @cached_property
    def land_mask(self) -> xr.DataArray:
        return self.train_obs.isel(time=0).notnull().drop_vars("time")

    @cached_property
    def forecast_core_stats(self) -> pd.DataFrame:
        rows = pd.read_parquet(self.data_dir / "processed" / "regime_features.parquet",
                               columns=["valid_date", "lead_day", "month", "fc_core_rain_mm"])
        tr0, tr1 = self.s.split.train
        return fit_forecast_core_stats(rows[rows["valid_date"].dt.year.between(tr0, tr1)])

    @cached_property
    def districts(self) -> DistrictWeights:
        return DistrictWeights.load(self.data_dir / "interim" / "static" / "district_weights")

    @cached_property
    def classifier_scores(self) -> dict[str, float]:
        """Validation macro-F1 per classifier from the Phase 5 report ({} if it has not been run)."""
        report = self.s.paths.model_dir.parent / "reports" / "phase5_validation.json"
        if not report.is_file():
            return {}
        results = json.loads(report.read_text(encoding="utf-8"))["results"]
        return {name: r["all"]["macro_f1"] for name, r in results.items() if "macro_f1" in r.get("all", {})}

    def _classifier_meta(self, name: str, description: str) -> dict:
        return {"classifier": description, "classifier_macro_f1": self.classifier_scores.get(name),
                "rule_macro_f1": self.classifier_scores.get("R0_rule")}

    @cached_property
    def validation_rmse(self) -> dict:
        report = self.s.paths.model_dir.parent / "reports" / "phase6_validation.json"
        if not report.is_file():
            return {}
        res = json.loads(report.read_text(encoding="utf-8"))["results"]
        return {lead: {"raw": r["A_raw_nwp"]["rmse"], "corrected": r["C1_regime_features"]["rmse"]}
                for lead, r in res.items()}

    # ---- inputs ----
    def archive_inputs(self, year: int, init: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame, xr.Dataset]:
        """Cell table, classifier rows and synoptic fields from the archived reforecast (one init or a whole year)."""
        filters = None if init is None else [("init_time", "==", pd.Timestamp(init))]
        table = pd.read_parquet(self.data_dir / "processed" / f"table_{year}.parquet", filters=filters)
        if table.empty:
            raise ValueError(f"no forecast rows for {init or year}")
        inits = pd.DatetimeIndex(table["init_time"].unique())
        regime_rows = pd.read_parquet(self.data_dir / "processed" / "regime_features.parquet")
        regime_rows = regime_rows[regime_rows["init_time"].isin(inits)].sort_values(["init_time", "lead_day"])
        fields = xr.open_dataset(self.data_dir / "interim" / "gefs_fields" / f"fields_{year}.nc").load()
        return table, regime_rows, fields.sel(init_time=inits)

    def live_inputs(self, fc: xr.Dataset, fields: xr.Dataset,
                    observed: xr.Dataset | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cell table and classifier rows built directly from operational forecasts (any number of inits)."""
        table = build_forecast_table(fc, self.rain_climatology, self.land_mask)
        if observed is not None:
            days = pd.DatetimeIndex(table["valid_date"].unique())
            obs = observed["precip_mm"].reindex(time=days)
            lookup = obs.sel(time=xr.DataArray(table["valid_date"].to_numpy(), dims="row"),
                             lat=xr.DataArray(table["lat"].to_numpy(), dims="row"),
                             lon=xr.DataArray(table["lon"].to_numpy(), dims="row"))
            table["obs_precip_mm"] = lookup.to_numpy().astype(np.float32)
        cfg = load_yaml("regimes.yaml")
        months = list(self.s.season.months)
        regime_rows = build_classifier_table(fields, fc, self.land_mask, cfg["classifier_features"],
                                             cfg["synoptic"]["active_break"]["core_zone_box"],
                                             pd.Series(dtype=float), (min(months), max(months)))
        regime_rows = add_forecast_anomaly(regime_rows, self.forecast_core_stats).sort_values(["init_time", "lead_day"])
        return table, regime_rows

    # ---- predictions (shared by products and the one-time evaluation) ----
    def predict(self, table: pd.DataFrame, regime_rows: pd.DataFrame, fields: xr.Dataset, live: bool,
                labels: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add regime inputs, corrected rainfall and heavy-rain probabilities to every cell row.

        `labels` (observed regimes) are attached only as obs_* diagnostic columns for scoring, never used as inputs.
        """
        classifier = self.live_classifier if live else self.classifier
        features = self.live_classifier_features if live else self.classifier_features
        proba = pd.DataFrame(classifier.predict(regime_rows[features].astype(np.float32)), columns=CLASSES)
        probs = regime_rows[["init_time", "lead_day"]].reset_index(drop=True)
        probs[PROB_COLS] = proba.to_numpy()
        probs["pred_regime"] = proba.idxmax(axis=1)
        labels = pd.Series(dtype=object) if labels is None else labels
        table = add_regime_columns(table, fields, self.terrain, load_yaml("regimes.yaml")["local"], probs, labels)

        table["corrected_mm"] = self.corrector.predict(table)
        table["qm_mm"] = self.qm.predict(table)
        for target in ("heavy", "very_heavy"):
            t = float(self.thresholds[target])
            clim = self.event_climatology[t]
            table[f"clim_p_ge_{t}"] = clim.sel(
                dayofyear=xr.DataArray(table["valid_date"].dt.dayofyear.to_numpy(), dims="row"),
                lat=xr.DataArray(table["lat"].to_numpy(), dims="row"),
                lon=xr.DataArray(table["lon"].to_numpy(), dims="row")).to_numpy()
            booster, heavy_features, calibrate = self.heavy(target)
            table[f"p_{target}"] = calibrate(booster.predict(table[heavy_features].astype(np.float32)))
        table["local_regime"] = table[LOCAL_COLS].to_numpy().argmax(axis=1)
        return table, probs

    # ---- products ----
    def build(self, init: date) -> Product:
        """Product from the archived reforecast tables (training, validation or test years)."""
        table, regime_rows, fields = self.archive_inputs(init.year, init)
        table, probs = self.predict(table, regime_rows, fields, live=False)
        return self._product(init, probs, table, {
            "source": "archive", "forecast": "NOAA GEFSv12 reforecast, control member",
            **self._classifier_meta("LGBM", "LightGBM on forecast fields and rainfall observed before the forecast "
                                            "was issued"),
            "in_season": True, "verification": "observed rainfall attached"})

    def build_live(self, init: date, fc: xr.Dataset, fields: xr.Dataset,
                   observed: xr.Dataset | None = None) -> Product:
        """Product from a freshly downloaded operational forecast.

        Uses the classifier trained without observed persistence (IMD publishes each year's grid only after the
        year ends). `observed`, when IMD has published the valid days, is attached for verification only.
        """
        table, regime_rows = self.live_inputs(fc, fields, observed)
        table, probs = self.predict(table, regime_rows, fields, live=True)
        months = set(self.s.season.months)
        valid_months = {int(m) for m in table["valid_date"].dt.month.unique()}
        return self._product(init, probs, table, {
            "source": "live", "forecast": "NOAA GEFSv12 operational, control member",
            **self._classifier_meta("LGBM_live", "LightGBM on forecast fields only"),
            "in_season": valid_months <= months,
            "verification": "observed rainfall attached" if observed is not None
            else "pending: IMD has not published observations for these days"})

    def _product(self, init: date, probs: pd.DataFrame, table: pd.DataFrame, meta: dict) -> Product:
        summary = {**self._summary(init, probs, table), **meta}  # needs the model's own feature names
        display = table.rename(columns={"nwp_precip_mm": "raw_mm", "obs_precip_mm": "observed_mm"})
        return Product(self._grids(display), self._districts(display), summary)

    def _grids(self, table: pd.DataFrame) -> xr.Dataset:
        indexed = table.set_index(["lead_day", "lat", "lon"])[list(GRID_VARS)]
        ds = indexed.to_xarray()
        d = self.s.domain
        lat = np.round(np.arange(d.lat_min, d.lat_max + 1e-9, d.resolution_deg), 4)
        lon = np.round(np.arange(d.lon_min, d.lon_max + 1e-9, d.resolution_deg), 4)
        ds = ds.reindex(lat=lat, lon=lon, method="nearest", tolerance=1e-6)
        ds.attrs = {"data_kind": "real", "observed_mm": "IMD gridded observation, for verification only",
                    "corrected_mm": "C1 regime-aware LightGBM", "qm_mm": "B1 quantile mapping (preserves heavy rain)",
                    "p_heavy": f"P(rain >= {self.thresholds['heavy']} mm), LightGBM with regime inputs",
                    "p_very_heavy": f"P(rain >= {self.thresholds['very_heavy']} mm), isotonic-calibrated",
                    "local_regime": json.dumps(LOCAL_CODES)}
        return ds.astype(np.float32)

    def _districts(self, table: pd.DataFrame) -> pd.DataFrame:
        w = self.districts
        cell_index = pd.MultiIndex.from_frame(w.cells[["lat", "lon"]])
        rows = []
        for lead, group in table.groupby("lead_day"):
            g = group.set_index(["lat", "lon"]).reindex(cell_index)
            out = w.districts[["district_id", "district_name", "state_name", "coverage_fraction"]].copy()
            out["lead_day"] = int(lead)
            out["valid_date"] = group["valid_date"].iloc[0]
            for col in ("raw_mm", "corrected_mm", "qm_mm", "observed_mm"):
                values = g[col].to_numpy(dtype=float)
                out[col] = w.aggregate_mean(np.nan_to_num(values)) if not np.isnan(values).all() else np.nan
            for col in ("p_heavy", "p_very_heavy"):
                out[f"{col}_max"] = w.aggregate_max(g[col].fillna(0.0).to_numpy(dtype=float))
            rows.append(out)
        return pd.concat(rows, ignore_index=True)

    def _summary(self, init: date, probs: pd.DataFrame, table: pd.DataFrame) -> dict:
        contrib = self._contributions(table)
        leads = {}
        for _, row in probs.iterrows():
            lead = int(row["lead_day"])
            p = {c: float(row[f"p_{c}"]) for c in CLASSES}
            leads[str(lead)] = {
                "valid_date": str((pd.Timestamp(init) + pd.Timedelta(days=lead)).date()),
                "predicted_regime": row["pred_regime"], "regime_confidence": max(p.values()), "regime_probabilities": p,
                "validation_rmse_mm": self.validation_rmse.get(str(lead)),
                "top_correction_features": contrib.get(lead, []),
            }
        return {"init_date": str(init), "data_kind": "real", "mode": self.s.mode,
                "system": "post-processing of GEFSv12 rainfall forecasts (control member)",
                "correction_model": "C1: LightGBM Tweedie with predicted regime probabilities and local regime as inputs",
                "heavy_rain_model": "LightGBM binary with regime inputs; very-heavy probabilities isotonic-calibrated on out-of-fold training data",
                "leads": leads}

    def _contributions(self, table: pd.DataFrame) -> dict[int, list[dict]]:
        """Mean |SHAP| contribution of each input to the corrected rainfall, per lead (from the model)."""
        features = self.corrector.features
        booster = self.corrector.booster
        out = {}
        for lead, group in table.groupby("lead_day"):
            contrib = booster.predict(group[features].astype(np.float32), pred_contrib=True)[:, :-1]
            share = np.abs(contrib).mean(axis=0)
            share = share / share.sum()
            order = np.argsort(share)[::-1][:5]
            out[int(lead)] = [{"feature": features[i], "share": float(share[i])} for i in order]
        return out
