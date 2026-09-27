"""Baseline B2: one LightGBM corrector for all situations (regime-blind).

It learns observed rainfall from forecast-time features with a Tweedie objective, which fits
rainfall's many exact zeros and long right tail better than squared error. Predictions are the
Tweedie mean, which is non-negative by construction; a final clip guards against numerical noise.
Early stopping uses the validation years, never the test years.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

TARGET = "obs_precip_mm"


class LGBMCorrector:
    name = "global_lgbm"

    def __init__(self, features: list[str], params: dict, num_boost_round: int = 2000,
                 early_stopping_rounds: int = 50) -> None:
        self.features = list(features)
        self.params = dict(params)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: lgb.Booster | None = None

    def _matrix(self, table: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in table.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        return table[self.features].astype(np.float32)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> LGBMCorrector:
        dtrain = lgb.Dataset(self._matrix(train), label=train[TARGET].to_numpy(), free_raw_data=True)
        dvalid = lgb.Dataset(self._matrix(valid), label=valid[TARGET].to_numpy(), reference=dtrain)
        self.booster = lgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False), lgb.log_evaluation(100)],
        )
        return self

    def fit_fixed_rounds(self, train: pd.DataFrame, rounds: int) -> LGBMCorrector:
        """Refit with a round count chosen earlier by early stopping on held-out data."""
        dtrain = lgb.Dataset(self._matrix(train), label=train[TARGET].to_numpy(), free_raw_data=True)
        self.booster = lgb.train(self.params, dtrain, num_boost_round=rounds)
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        pred = self.booster.predict(self._matrix(table), num_iteration=self.booster.best_iteration)
        return np.clip(pred, 0.0, None)

    def feature_importance(self) -> dict[str, float]:
        """Share of total split gain per feature (sums to 1)."""
        gain = self.booster.feature_importance(importance_type="gain")
        return {f: float(g / gain.sum()) for f, g in zip(self.features, gain, strict=True)}

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.txt"
        self.booster.save_model(str(path), num_iteration=self.booster.best_iteration)
        (directory / f"{self.name}_features.json").write_text(json.dumps(self.features), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> LGBMCorrector:
        features = json.loads((directory / f"{cls.name}_features.json").read_text(encoding="utf-8"))
        model = cls(features, params={})
        model.booster = lgb.Booster(model_file=str(directory / f"{cls.name}.txt"))
        return model
