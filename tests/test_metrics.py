from __future__ import annotations

import math

import numpy as np
import pytest

from rainpp.verification.metrics import (
    Contingency,
    calculate_bias,
    calculate_correlation,
    calculate_csi,
    calculate_ets,
    calculate_far,
    calculate_fss,
    calculate_mae,
    calculate_pod,
    calculate_rmse,
    contingency,
)

# ---- continuous -------------------------------------------------------------

def test_continuous_known_values() -> None:
    f, o = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 2.0, 2.0])
    assert calculate_rmse(f, o) == pytest.approx(math.sqrt((1 + 0 + 1 + 4) / 4))
    assert calculate_mae(f, o) == pytest.approx(1.0)
    assert calculate_bias(f, o) == pytest.approx(0.5)


def test_nan_pairs_are_dropped_not_zero_filled() -> None:
    f, o = np.array([1.0, np.nan, 3.0]), np.array([1.0, 50.0, np.nan])
    assert calculate_rmse(f, o) == 0.0


def test_all_missing_gives_nan() -> None:
    nan = np.array([np.nan, np.nan])
    assert math.isnan(calculate_rmse(nan, nan))
    assert math.isnan(calculate_bias(nan, nan))


def test_correlation_undefined_for_constant_field() -> None:
    assert math.isnan(calculate_correlation(np.zeros(5), np.arange(5.0)))


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        calculate_rmse(np.zeros(3), np.zeros(4))


# ---- categorical ------------------------------------------------------------

def test_contingency_scores_textbook_example() -> None:
    table = Contingency(64.5, hits=20, false_alarms=10, misses=5, correct_negatives=65)
    assert table.pod == pytest.approx(0.8)
    assert table.far == pytest.approx(1 / 3)
    assert table.csi == pytest.approx(20 / 35)
    assert table.ets == pytest.approx(12.5 / 27.5)  # a_r = 30 * 25 / 100 = 7.5
    assert table.frequency_bias == pytest.approx(1.2)


def test_contingency_counts_use_greater_or_equal() -> None:
    f = np.array([64.5, 64.4, 70.0, 0.0])
    o = np.array([64.5, 70.0, 0.0, 0.0])
    t = contingency(f, o, 64.5)
    assert (t.hits, t.false_alarms, t.misses, t.correct_negatives) == (1, 1, 1, 1)


def test_undefined_scores_are_nan_not_zero() -> None:
    dry = np.zeros(10)
    assert math.isnan(calculate_pod(dry, dry, 64.5))  # no observed events
    assert math.isnan(calculate_far(dry, dry, 64.5))  # no forecast events
    assert math.isnan(calculate_csi(dry, dry, 64.5))
    assert math.isnan(calculate_ets(dry, dry, 64.5))


def test_perfect_forecast_scores() -> None:
    o = np.array([0.0, 100.0, 5.0, 80.0, 0.0])
    assert calculate_pod(o, o, 64.5) == 1.0
    assert calculate_far(o, o, 64.5) == 0.0
    assert calculate_csi(o, o, 64.5) == 1.0
    assert calculate_ets(o, o, 64.5) > 0.0


def test_random_forecast_has_near_zero_ets() -> None:
    rng = np.random.default_rng(1)
    o, f = rng.gamma(0.5, 20, 200_000), rng.gamma(0.5, 20, 200_000)
    assert abs(calculate_ets(f, o, 15.6)) < 0.01


# ---- FSS --------------------------------------------------------------------

def _field(points: list[tuple[int, int]], shape=(11, 11)) -> np.ndarray:
    field = np.zeros(shape)
    for i, j in points:
        field[i, j] = 100.0
    return field


def test_fss_identical_fields_is_one() -> None:
    f = _field([(5, 5)])
    assert calculate_fss(f, f, 64.5, 1) == 1.0


def test_fss_displacement_penalised_at_grid_scale_but_credited_at_larger_scale() -> None:
    f, o = _field([(5, 5)]), _field([(5, 6)])
    assert calculate_fss(f, o, 64.5, 1) == 0.0
    assert calculate_fss(f, o, 64.5, 3) > 0.5
    assert calculate_fss(f, o, 64.5, 5) > calculate_fss(f, o, 64.5, 3)


def test_fss_undefined_when_no_events_anywhere() -> None:
    dry = np.zeros((5, 5))
    assert math.isnan(calculate_fss(dry, dry, 64.5, 3))


def test_fss_ignores_cells_outside_domain() -> None:
    f, o = _field([(5, 5)]), _field([(5, 5)])
    f[0, 0] = 999.0  # event only in the forecast...
    o[0, 0] = np.nan  # ...at a cell outside the verification domain
    assert calculate_fss(f, o, 64.5, 1) == 1.0


def test_fss_accumulates_over_days() -> None:
    good, bad = _field([(5, 5)]), _field([(1, 1)])
    stack_f = np.stack([good, good])
    stack_o = np.stack([good, bad])
    assert 0.0 < calculate_fss(stack_f, stack_o, 64.5, 1) < 1.0


def test_fss_rejects_even_window() -> None:
    with pytest.raises(ValueError):
        calculate_fss(np.zeros((5, 5)), np.zeros((5, 5)), 1.0, 2)
