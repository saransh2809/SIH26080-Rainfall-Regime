"""LightGBM corrector tests on a small SYNTHETIC table with a known bias."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rainpp.models.global_bc import LGBMCorrector

PARAMS = {"objective": "tweedie", "tweedie_variance_power": 1.5, "learning_rate": 0.1,
          "num_leaves": 15, "min_data_in_leaf": 20, "verbose": -1, "seed": 0}


def _synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nwp = rng.gamma(0.5, 20, n)
    obs = np.where(rng.random(n) < 0.3, 0.0, 0.5 * nwp * rng.lognormal(0, 0.2, n))  # model over-forecasts x2
    return pd.DataFrame({"nwp_precip_mm": nwp, "lat": rng.uniform(8, 30, n), "obs_precip_mm": obs})


@pytest.fixture(scope="module")
def fitted() -> LGBMCorrector:
    model = LGBMCorrector(["nwp_precip_mm", "lat"], PARAMS, num_boost_round=200, early_stopping_rounds=20)
    return model.fit(_synthetic(4000, 0), _synthetic(1000, 1))


def test_learns_to_remove_overforecast_bias(fitted) -> None:
    test = _synthetic(2000, 2)
    raw_bias = (test.nwp_precip_mm - test.obs_precip_mm).mean()
    corrected_bias = (fitted.predict(test) - test.obs_precip_mm).mean()
    assert abs(corrected_bias) < 0.3 * abs(raw_bias)


def test_predictions_non_negative_and_right_shape(fitted) -> None:
    pred = fitted.predict(_synthetic(500, 3))
    assert pred.shape == (500,)
    assert (pred >= 0).all()


def test_importance_sums_to_one_and_favours_signal(fitted) -> None:
    imp = fitted.feature_importance()
    assert sum(imp.values()) == pytest.approx(1.0)
    assert imp["nwp_precip_mm"] > imp["lat"]


def test_missing_feature_rejected(fitted) -> None:
    with pytest.raises(KeyError, match="lat"):
        fitted.predict(pd.DataFrame({"nwp_precip_mm": [1.0]}))


def test_fixed_round_refit_uses_requested_rounds() -> None:
    model = LGBMCorrector(["nwp_precip_mm", "lat"], PARAMS).fit_fixed_rounds(_synthetic(1000, 5), 7)
    assert model.booster.num_trees() == 7


def test_save_load_round_trip(fitted, tmp_path: Path) -> None:
    fitted.save(tmp_path)
    loaded = LGBMCorrector.load(tmp_path)
    test = _synthetic(100, 4)
    np.testing.assert_allclose(loaded.predict(test), fitted.predict(test), rtol=1e-6)
