"""Classifier tests on a SYNTHETIC separable problem with known structure."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rainpp.regimes.classifier import (
    CLASSES,
    ForestClassifier,
    LGBMRegimeClassifier,
    RuleClassifier,
    classification_scores,
    predict_with_confidence,
)

FEATURES = ["fc_core_anom", "trough_vort_max"]


def _data(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    anom, vort = rng.normal(0, 1, n), rng.normal(5, 2, n)
    vort[rng.random(n) < 0.1] += 12.0
    y = np.select([vort > 12, anom > 1, anom < -1], ["MONSOON_DEPRESSION", "ACTIVE", "BREAK"], "NORMAL")
    return pd.DataFrame({"fc_core_anom": anom, "trough_vort_max": vort}), pd.Series(y)


def test_rule_learns_vorticity_threshold_from_training() -> None:
    X, y = _data(3000, 0)
    rule = RuleClassifier().fit(X, y)
    assert 9 < rule.vort_threshold < 15
    Xt, yt = _data(1000, 1)
    assert classification_scores(yt.to_numpy(), rule.predict(Xt))["macro_f1"] > 0.9


@pytest.mark.parametrize("make", [
    lambda: ForestClassifier(FEATURES),
    lambda: LGBMRegimeClassifier(FEATURES, {"learning_rate": 0.1, "num_leaves": 7, "verbose": -1, "seed": 0}, 100),
])
def test_ml_classifiers_output_valid_probabilities(make) -> None:
    X, y = _data(2000, 2)
    model = make().fit(X, y)
    proba = model.predict_proba(_data(300, 3)[0])
    assert list(proba.columns) == CLASSES
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert ((proba >= 0) & (proba <= 1)).all().all()
    out = predict_with_confidence(model, _data(300, 3)[0])
    assert set(out["predicted_regime"]) <= set(CLASSES)
    Xt, yt = _data(1000, 4)
    predicted = predict_with_confidence(model, Xt)["predicted_regime"].to_numpy()
    assert classification_scores(yt.to_numpy(), predicted)["macro_f1"] > 0.8


def test_scores_known_case_and_absent_class_is_nan() -> None:
    y_true = np.array(["NORMAL", "NORMAL", "ACTIVE", "ACTIVE"])
    y_pred = np.array(["NORMAL", "ACTIVE", "ACTIVE", "ACTIVE"])
    s = classification_scores(y_true, y_pred)
    assert s["accuracy"] == 0.75
    assert s["per_class"]["ACTIVE"]["precision"] == pytest.approx(2 / 3)
    assert s["per_class"]["NORMAL"]["recall"] == 0.5
    assert math.isnan(s["per_class"]["BREAK"]["recall"])  # no BREAK in truth: undefined, not 0
    assert s["confusion_matrix"]["rows_true_cols_pred"][0][:2] == [1, 1]
