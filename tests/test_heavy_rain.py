"""Heavy-rain probability tests on SYNTHETIC data with a known event probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rainpp.models.heavy_rain import HeavyRainModel, fit_event_climatology
from rainpp.verification.probabilistic import brier_skill_score, reliability_table

PARAMS = {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 50, "verbose": -1, "seed": 0,
          "tweedie_variance_power": 1.5}


def _data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nwp = rng.gamma(0.7, 25, n)
    p_true = 1 / (1 + np.exp(-(nwp - 80) / 15))           # heavier forecast rain -> likelier heavy event
    heavy = rng.uniform(0, 1, n) < p_true
    obs = np.where(heavy, 64.5 + rng.gamma(1, 20, n), rng.uniform(0, 60, n))
    return pd.DataFrame({"nwp_precip_mm": nwp, "obs_precip_mm": obs, "p_true": p_true})


def test_learns_calibrated_probabilities() -> None:
    model = HeavyRainModel(64.5, ["nwp_precip_mm"], PARAMS, max_rounds=300)
    model.fit(_data(20000, 0), _data(5000, 1), _data(25000, 2))
    assert "tweedie_variance_power" not in model.params
    test = _data(20000, 3)
    p = model.predict_proba(test)
    assert ((p >= 0) & (p <= 1)).all()
    o = (test.obs_precip_mm >= 64.5).astype(float).to_numpy()
    assert brier_skill_score(p, np.full_like(p, o.mean()), o) > 0.2
    for row in reliability_table(p, o, bins=5):
        if row["n"] > 500:
            assert row["observed_frequency"] == pytest.approx(row["mean_forecast"], abs=0.08)


def test_event_climatology_is_a_frequency() -> None:
    time = pd.date_range("2010-06-01", "2012-09-30")
    rain = np.where(time.month == 7, 100.0, 0.0)[:, None, None] * np.ones((1, 1, 1))
    clim = fit_event_climatology(xr.DataArray(rain, dims=("time", "lat", "lon"), coords={"time": time}), 64.5)
    assert clim.sel(dayofyear=196).item() > 0.9   # mid-July; the ±15-day window reaches 30 June
    assert clim.sel(dayofyear=250).item() == pytest.approx(0.0)


def test_isotonic_calibration_fixes_overconfidence(tmp_path) -> None:
    from rainpp.models.heavy_rain import IsotonicCalibrator

    rng = np.random.default_rng(5)
    p_true = rng.uniform(0, 0.4, 100_000)
    outcome = (rng.uniform(0, 1, p_true.size) < p_true).astype(float)
    overconfident = np.clip(p_true * 2.0, 0, 1)                   # says 0.6 when the truth is 0.3
    cal = IsotonicCalibrator().fit(overconfident[:50_000], outcome[:50_000])
    fixed = cal.transform(overconfident[50_000:])
    for row in reliability_table(fixed, outcome[50_000:], bins=5):
        if row["n"] > 2_000:
            assert row["observed_frequency"] == pytest.approx(row["mean_forecast"], abs=0.03)
    cal.save(tmp_path / "cal.json")
    np.testing.assert_allclose(IsotonicCalibrator.load_transform(tmp_path / "cal.json")(overconfident[:10]),
                               cal.transform(overconfident[:10]), atol=1e-9)
