"""Synoptic regime classifiers. All share: fit(X, y) → self; predict_proba(X) → DataFrame over CLASSES.

  R0 rule baseline — the ±1 core-anomaly threshold applied to the FORECAST anomaly day by day
     (the 3-day persistence of Rajeevan et al. cannot be checked within a 3-day forecast whose first
     rain day is unfinished at issue time), plus a depression rule: forecast 850 hPa vorticity maximum
     in the trough box above a threshold fitted on training data only.
  RF  RandomForest with balanced class weights.
  LGBM LightGBM multiclass with balanced class weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from rainpp.regimes.labels import SYNOPTIC_CLASSES

CLASSES = list(SYNOPTIC_CLASSES)


def _one_hot(labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({c: (labels == c).astype(float) for c in CLASSES})


class RuleClassifier:
    name = "R0_rule"

    def __init__(self, anomaly_threshold: float = 1.0) -> None:
        self.anomaly_threshold = anomaly_threshold
        self.vort_threshold: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RuleClassifier:
        """Choose the vorticity threshold maximising depression F1 on the training rows."""
        is_dep = (y.to_numpy() == "MONSOON_DEPRESSION")
        candidates = np.quantile(X["trough_vort_max"], np.linspace(0.5, 0.99, 50))
        scores = [f1_score(is_dep, X["trough_vort_max"].to_numpy() >= t, zero_division=0) for t in candidates]
        self.vort_threshold = float(candidates[int(np.argmax(scores))])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.vort_threshold is None:
            raise RuntimeError("model is not fitted")
        anom = X["fc_core_anom"].to_numpy()
        return np.select(
            [X["trough_vort_max"].to_numpy() >= self.vort_threshold, anom > self.anomaly_threshold,
             anom < -self.anomaly_threshold],
            ["MONSOON_DEPRESSION", "ACTIVE", "BREAK"], "NORMAL")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return _one_hot(self.predict(X))


def balanced_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map(len(y) / (len(counts) * counts)).to_numpy()


class ForestClassifier:
    name = "RF"

    def __init__(self, features: list[str], seed: int = 42, **params) -> None:
        self.features = list(features)
        self.model = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=seed, **params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ForestClassifier:
        self.model.fit(X[self.features], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = pd.DataFrame(self.model.predict_proba(X[self.features]), columns=self.model.classes_)
        return proba.reindex(columns=CLASSES, fill_value=0.0)

    def feature_importance(self) -> dict[str, float]:
        return dict(zip(self.features, map(float, self.model.feature_importances_), strict=True))


class LGBMRegimeClassifier:
    name = "LGBM"

    def __init__(self, features: list[str], params: dict, num_boost_round: int) -> None:
        self.features = list(features)
        self.params = {**params, "objective": "multiclass", "num_class": len(CLASSES)}
        self.num_boost_round = num_boost_round
        self.booster = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LGBMRegimeClassifier:
        import lightgbm as lgb

        codes = y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        data = lgb.Dataset(X[self.features].astype(np.float32), label=codes, weight=balanced_weights(y))
        self.booster = lgb.train(self.params, data, num_boost_round=self.num_boost_round)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.booster.predict(X[self.features].astype(np.float32)), columns=CLASSES)


def predict_with_confidence(model, X: pd.DataFrame) -> pd.DataFrame:
    """predicted_regime and its probability (for R0 the 'probability' is 1, it is not calibrated)."""
    proba = model.predict_proba(X)
    return pd.DataFrame({"predicted_regime": proba.idxmax(axis=1), "confidence": proba.max(axis=1)})


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, balanced accuracy, macro F1, per-class P/R/F1/support and the confusion matrix."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=np.nan)
    return {
        "n": len(y_true),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": CLASSES,
                             "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()},
    }
