"""Block-bootstrap tests on SYNTHETIC forecasts with a known quality difference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rainpp.verification.bootstrap import block_bootstrap_difference
from rainpp.verification.metrics import calculate_rmse


def _table(noise_b: float, n_days: int = 120, cells: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2016-06-01", periods=n_days), cells)
    obs = rng.gamma(0.6, 10, dates.size)
    return pd.DataFrame({"valid_date": dates, "obs_precip_mm": obs,
                         "a": obs + rng.normal(0, 5.0, dates.size), "b": obs + rng.normal(0, noise_b, dates.size)})


def test_clearly_better_forecast_is_significant() -> None:
    res = block_bootstrap_difference(_table(noise_b=2.0), calculate_rmse, "a", "b", n_boot=200)
    assert res["improvement"] > 0 and res["significant"]
    assert res["ci95_low"] > 0


def test_equal_forecasts_are_not_significant() -> None:
    res = block_bootstrap_difference(_table(noise_b=5.0), calculate_rmse, "a", "b", n_boot=200)
    assert res["ci95_low"] < 0 < res["ci95_high"]
    assert not res["significant"]


def test_blocks_keep_days_together() -> None:
    res = block_bootstrap_difference(_table(noise_b=2.0, n_days=20), calculate_rmse, "a", "b", n_boot=20, block_days=5)
    assert res["n_blocks"] == 4
