"""C2 regime-split corrector on a SYNTHETIC problem where the bias differs by regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rainpp.models.regime_bc import RegimeSplitCorrector

PARAMS = {"objective": "tweedie", "tweedie_variance_power": 1.5, "learning_rate": 0.1,
          "num_leaves": 15, "min_data_in_leaf": 20, "verbose": -1, "seed": 0}


def _data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = rng.choice(["ACTIVE", "BREAK", "RARE"], n, p=[0.5, 0.48, 0.02])
    nwp = rng.gamma(0.8, 15, n)
    factor = np.select([regime == "ACTIVE", regime == "BREAK"], [2.0, 0.3], 1.0)  # opposite biases
    return pd.DataFrame({"nwp_precip_mm": nwp, "regime": regime,
                         "obs_precip_mm": factor * nwp * rng.lognormal(0, 0.1, n)})


def test_split_model_learns_regime_specific_bias_and_falls_back_for_rare() -> None:
    model = RegimeSplitCorrector(["nwp_precip_mm"], PARAMS, rounds=150, regime_column="regime", min_rows=500)
    model.fit(_data(6000, 0))
    assert model.fallback_regimes == ["RARE"]
    test = _data(3000, 1)
    pred = model.predict(test)
    for regime in ("ACTIVE", "BREAK"):
        rows = test.regime == regime
        ratio = pred[rows].sum() / test.obs_precip_mm[rows].sum()
        assert 0.9 < ratio < 1.1, (regime, ratio)
    routing = model.routing(test)
    assert routing["pooled_fallback"] == int((test.regime == "RARE").sum())
    assert (pred >= 0).all()


def _qm_table(n_days: int, seed: int) -> pd.DataFrame:
    """Two cells; the forecast is 2x too wet in ACTIVE and 3x too dry in BREAK. SYNTHETIC."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2010-06-01", periods=n_days):
        regime = "ACTIVE" if rng.random() < 0.5 else "BREAK"
        for lon in (76.0, 76.25):
            nwp = rng.gamma(0.8, 15)
            obs = nwp / 2 if regime == "ACTIVE" else nwp * 3
            rows.append({"valid_date": d, "lead_day": 1, "lat": 10.0, "lon": lon, "regime": regime,
                         "nwp_precip_mm": nwp, "obs_precip_mm": obs})
    return pd.DataFrame(rows)


def test_regime_qm_learns_opposite_biases_and_falls_back() -> None:
    from rainpp.models.regime_bc import RegimeQuantileMapping

    train = _qm_table(400, 0)
    model = RegimeQuantileMapping("regime", min_days=40).fit(train)
    assert set(model.by_regime) == {"ACTIVE", "BREAK"} and model.fallback_regimes == []
    test = _qm_table(300, 1)
    pred = model.predict(test)
    for regime in ("ACTIVE", "BREAK"):
        rows = (test.regime == regime).to_numpy()
        ratio = pred[rows].mean() / test.obs_precip_mm[rows].mean()
        assert 0.8 < ratio < 1.25, (regime, ratio)
    sparse = RegimeQuantileMapping("regime", min_days=10_000).fit(train)
    assert sorted(sparse.fallback_regimes) == ["ACTIVE", "BREAK"]


def test_checkerboard_keeps_regular_half_of_cells() -> None:
    from rainpp.regimes.augment import checkerboard_mask

    lat, lon = np.meshgrid(np.arange(10.0, 12.0, 0.25), np.arange(70.0, 72.0, 0.25), indexing="ij")
    mask = checkerboard_mask(lat.ravel(), lon.ravel(), 0.25, 2).reshape(lat.shape)
    assert mask.mean() == 0.5
    assert not (mask[:, :-1] & mask[:, 1:]).any()   # no two kept cells side by side
    assert checkerboard_mask(lat.ravel(), lon.ravel(), 0.25, 1).all()
