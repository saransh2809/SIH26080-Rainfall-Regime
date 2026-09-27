"""Quantile mapping tests use small SYNTHETIC tables with known distributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rainpp.models.quantile_mapping import QuantileMapping


def _table(fc_by_cell: dict[tuple[float, float], np.ndarray], ob_by_cell: dict, lead: int = 1) -> pd.DataFrame:
    rows = []
    for cell, fc in fc_by_cell.items():
        dates = pd.date_range("2010-06-01", periods=fc.size)
        for d, f, o in zip(dates, fc, ob_by_cell[cell], strict=True):
            rows.append({"valid_date": d, "lead_day": lead, "lat": cell[0], "lon": cell[1],
                         "nwp_precip_mm": f, "obs_precip_mm": o})
    return pd.DataFrame(rows)


def test_learns_multiplicative_bias_per_cell() -> None:
    base = np.arange(1.0, 101.0)
    train = _table({(10.0, 76.0): base, (10.0, 76.25): base},
                   {(10.0, 76.0): 2 * base, (10.0, 76.25): 0.5 * base})
    qm = QuantileMapping().fit(train)
    query = train[train["valid_date"] == train["valid_date"].iloc[49]]  # forecast value 50
    np.testing.assert_allclose(qm.predict(query), [100.0, 25.0], rtol=0.02)


def test_dry_forecast_maps_into_dry_observations() -> None:
    fc = np.r_[np.zeros(40), np.arange(1.0, 61.0)]   # 40% dry forecasts
    ob = np.r_[np.zeros(60), np.arange(1.0, 41.0)]   # 60% dry observations
    qm = QuantileMapping().fit(_table({(10.0, 76.0): fc}, {(10.0, 76.0): ob}))
    q = _table({(10.0, 76.0): np.array([0.0])}, {(10.0, 76.0): np.array([0.0])})
    assert qm.predict(q)[0] == 0.0


def test_extrapolates_additively_above_training_maximum() -> None:
    fc, ob = np.arange(0.0, 100.0), np.arange(0.0, 100.0) + 10.0
    qm = QuantileMapping().fit(_table({(10.0, 76.0): fc}, {(10.0, 76.0): ob}))
    q = _table({(10.0, 76.0): np.array([300.0])}, {(10.0, 76.0): np.array([0.0])})
    assert qm.predict(q)[0] == pytest.approx(109.0 + 201.0)


def test_output_is_non_negative_and_monotonic() -> None:
    rng = np.random.default_rng(0)
    fc, ob = rng.gamma(0.4, 15, 500), rng.gamma(0.6, 12, 500)
    qm = QuantileMapping().fit(_table({(10.0, 76.0): fc}, {(10.0, 76.0): ob}))
    grid = np.linspace(0, 200, 400)
    out = qm.predict(_table({(10.0, 76.0): grid}, {(10.0, 76.0): grid}))
    assert (out >= 0).all()
    assert (np.diff(out) >= -1e-9).all()


def test_unseen_cell_rejected() -> None:
    qm = QuantileMapping().fit(_table({(10.0, 76.0): np.arange(10.0)}, {(10.0, 76.0): np.arange(10.0)}))
    with pytest.raises(ValueError, match="not seen"):
        qm.predict(_table({(20.0, 80.0): np.array([1.0])}, {(20.0, 80.0): np.array([1.0])}))


def test_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    train = _table({(10.0, 76.0): rng.gamma(0.5, 10, 50)}, {(10.0, 76.0): rng.gamma(0.5, 10, 50)})
    qm = QuantileMapping().fit(train)
    loaded = QuantileMapping.load(qm.save(tmp_path))
    np.testing.assert_array_equal(loaded.predict(train), qm.predict(train))
