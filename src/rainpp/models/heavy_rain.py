"""Heavy-rainfall probability: P(observed rain ≥ threshold | forecast-time features).

A LightGBM binary classifier with plain log-loss and NO class re-weighting, because re-weighting
inflates probabilities and breaks reliability. Rounds are chosen by early stopping on held-out
training data, then the model is refit (same protocol as the correctors). The climatological event
frequency (training years only) is both an input feature and the reference forecast for skill.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from rainpp.data.features import fit_climatology


def fit_event_climatology(obs_train: xr.DataArray, threshold_mm: float) -> xr.DataArray:
    """Smoothed day-of-year frequency of rain ≥ threshold per cell. Pass TRAINING years only."""
    events = (obs_train >= threshold_mm).astype(float).where(obs_train.notnull())
    return fit_climatology(events).rename(f"clim_p_ge_{threshold_mm}")


class HeavyRainModel:
    def __init__(self, threshold_mm: float, features: list[str], params: dict,
                 max_rounds: int = 2000, early_stopping_rounds: int = 50) -> None:
        self.threshold_mm = threshold_mm
        self.features = list(features)
        self.params = {**params, "objective": "binary"}
        self.params.pop("tweedie_variance_power", None)
        self.max_rounds, self.early_stopping_rounds = max_rounds, early_stopping_rounds
        self.booster: lgb.Booster | None = None
        self.rounds: int | None = None

    @property
    def name(self) -> str:
        return f"heavy_rain_ge_{self.threshold_mm}"

    def _data(self, table: pd.DataFrame, reference=None) -> lgb.Dataset:
        label = (table["obs_precip_mm"].to_numpy() >= self.threshold_mm).astype(np.float32)
        return lgb.Dataset(table[self.features].astype(np.float32), label=label, reference=reference)

    def fit(self, train: pd.DataFrame, stop: pd.DataFrame, refit_on: pd.DataFrame) -> HeavyRainModel:
        """Choose rounds with early stopping on `stop` (trained on `train`), then refit on `refit_on`."""
        dtrain = self._data(train)
        probe = lgb.train(self.params, dtrain, num_boost_round=self.max_rounds, valid_sets=[self._data(stop, dtrain)],
                          callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
        self.rounds = max(1, probe.best_iteration)
        self.booster = lgb.train(self.params, self._data(refit_on), num_boost_round=self.rounds)
        return self

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return self.booster.predict(table[self.features].astype(np.float32))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / f"{self.name}.txt"))
        (directory / f"{self.name}.json").write_text(
            json.dumps({"threshold_mm": self.threshold_mm, "features": self.features, "rounds": self.rounds}),
            encoding="utf-8")


class IsotonicCalibrator:
    """Monotone map from a model's probability to observed event frequency.

    Must be fitted on OUT-OF-FOLD predictions (the model never saw those rows), otherwise it learns
    the model's in-sample over-confidence instead of its real error.
    """

    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression

        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, p_oof: np.ndarray, outcome: np.ndarray) -> IsotonicCalibrator:
        self.iso.fit(p_oof, outcome)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.iso.predict(p), 0.0, 1.0)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"x": self.iso.X_thresholds_.tolist(), "y": self.iso.y_thresholds_.tolist()}),
                        encoding="utf-8")

    @staticmethod
    def load_transform(path: Path):
        table = json.loads(path.read_text(encoding="utf-8"))
        x, y = np.array(table["x"]), np.array(table["y"])
        return lambda p: np.clip(np.interp(p, x, y), 0.0, 1.0)


def out_of_fold_probabilities(model: HeavyRainModel, train: pd.DataFrame, years: pd.Series) -> np.ndarray:
    """Leave-one-year-out predictions with the model's chosen round count (no re-tuning per fold)."""
    if model.rounds is None:
        raise RuntimeError("fit the model first so the round count is known")
    oof = np.full(len(train), np.nan)
    for year in np.unique(years):
        held = (years == year).to_numpy()
        booster = lgb.train(model.params, model._data(train[~held]), num_boost_round=model.rounds)
        oof[held] = booster.predict(train.loc[held, model.features].astype(np.float32))
    return oof
