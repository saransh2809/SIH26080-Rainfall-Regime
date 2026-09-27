"""Regime label tests on small SYNTHETIC series and grids with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainpp.regimes.labels import (
    coast_distance_km,
    depression_days,
    local_regime,
    normalised_anomaly,
    rain_day_of,
    spell_mask,
    synoptic_labels,
    terrain_gradient,
)

CFG = {"active_break": {"anomaly_threshold": 1.0, "min_consecutive_days": 3, "original_months": [7, 8],
                        "climatology_years": [2000, 2011]}}
LOCAL_CFG = {"coastal": {"max_distance_to_coast_km": 100},
             "orographic": {"min_forced_ascent_ms": 0.05}}


def _daily(start: str, end: str, value: float = 10.0) -> pd.Series:
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(0)
    return pd.Series(value + rng.normal(0, 1.0, idx.size), index=idx)


def test_spell_needs_three_consecutive_days() -> None:
    idx = pd.date_range("2012-07-01", periods=8)
    cond = pd.Series([1, 1, 0, 1, 1, 1, 0, 1], index=idx, dtype=bool)
    assert spell_mask(cond, 3).tolist() == [False, False, False, True, True, True, False, False]


def test_spell_rejects_gappy_series() -> None:
    cond = pd.Series([True, True], index=pd.to_datetime(["2012-07-01", "2012-07-03"]))
    with pytest.raises(ValueError, match="continuous"):
        spell_mask(cond, 3)


def test_anomaly_is_standardised_against_base_years() -> None:
    series = _daily("2000-01-01", "2012-12-31")
    anom = normalised_anomaly(series, (2000, 2011))
    base = anom[anom.index.year <= 2011]
    assert abs(base.mean()) < 0.05
    assert base.std() == pytest.approx(1.0, abs=0.1)


def test_anomaly_requires_enough_base_years() -> None:
    with pytest.raises(ValueError, match="10 base years"):
        normalised_anomaly(_daily("2010-01-01", "2012-12-31"), (2010, 2012))


def test_synoptic_labels_priority_and_evidence() -> None:
    series = _daily("2000-01-01", "2012-12-31")
    series["2012-07-10":"2012-07-14"] += 10.0   # 5-day wet spell  -> ACTIVE
    series["2012-08-01":"2012-08-04"] -= 10.0   # 4-day dry spell  -> BREAK
    series["2012-06-20":"2012-06-23"] += 10.0   # June wet spell   -> ACTIVE, derived
    depressions = {pd.Timestamp("2012-07-12")}
    labels = synoptic_labels(series, depressions, CFG, [6, 7, 8, 9])
    assert labels.loc["2012-07-11", "synoptic_regime"] == "ACTIVE"
    assert labels.loc["2012-07-12", "synoptic_regime"] == "MONSOON_DEPRESSION"  # depression wins
    assert labels.loc["2012-08-02", "synoptic_regime"] == "BREAK"
    assert labels.loc["2012-07-25", "synoptic_regime"] == "NORMAL"
    assert labels.loc["2012-06-21", "evidence"] == "derived"
    assert labels.loc["2012-07-11", "evidence"] == "published"


def test_rain_day_of_uses_window_end_date() -> None:
    times = pd.Series(pd.to_datetime(["2012-07-10 02:00", "2012-07-10 04:00"]))
    assert rain_day_of(times).dt.date.astype(str).tolist() == ["2012-07-10", "2012-07-11"]


def test_depression_days_filter_box_and_grade() -> None:
    track = pd.DataFrame({
        "time": pd.to_datetime(["2012-07-10 06:00", "2012-07-10 06:00", "2012-07-11 06:00"]),
        "lat": [20.0, 5.0, 20.0], "lon": [85.0, 85.0, 85.0], "grade": ["D", "D", None]})
    box = {"lat_min": 12, "lat_max": 30, "lon_min": 68, "lon_max": 92}
    assert depression_days(track, box) == {pd.Timestamp("2012-07-11")}


def test_coast_distance_grows_inland() -> None:
    land = np.ones((5, 9))
    land[:, 0] = 0.0  # sea on the west edge
    dist = coast_distance_km(land, np.full(5, 20.0), 0.25)
    assert dist[2, 0] == 0.0
    assert dist[2, 1] < dist[2, 4] < dist[2, 8]


def test_local_regime_upslope_wind_makes_orographic() -> None:
    elevation = np.tile(np.arange(9) * 300.0, (5, 1))  # rises eastward ~11 m/km at 0.25°
    dh_dx, dh_dy = terrain_gradient(elevation, np.full(5, 12.0), 0.25)
    coast = np.full((5, 9), 500.0)
    westerly = local_regime(coast, dh_dx, dh_dy, np.full((5, 9), 10.0), np.zeros((5, 9)), LOCAL_CFG)
    easterly = local_regime(coast, dh_dx, dh_dy, np.full((5, 9), -10.0), np.zeros((5, 9)), LOCAL_CFG)
    assert (westerly[:, 2:-2] == "OROGRAPHIC").all()   # wind blowing up the slope
    assert (easterly == "INLAND").all()                  # downslope wind: not orographic


def test_local_regime_coastal_when_flat_and_near_sea() -> None:
    flat = np.zeros((3, 3))
    coast = np.array([[50.0] * 3, [150.0] * 3, [300.0] * 3])
    out = local_regime(coast, flat, flat, np.ones((3, 3)), np.ones((3, 3)), LOCAL_CFG)
    assert out[:, 0].tolist() == ["COASTAL", "INLAND", "INLAND"]
