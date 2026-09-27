from __future__ import annotations

import math

import numpy as np
import pytest

from rainpp.verification.probabilistic import (
    brier_score,
    brier_skill_score,
    probability_scores,
    reliability_table,
    roc_auc,
)


def test_brier_known_values() -> None:
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert brier_score(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == pytest.approx(0.25)


def test_bss_positive_only_when_better_than_reference() -> None:
    o = np.array([1.0, 0.0, 0.0, 0.0])
    clim = np.full(4, 0.25)
    assert brier_skill_score(np.array([0.9, 0.1, 0.1, 0.1]), clim, o) > 0
    assert brier_skill_score(clim, clim, o) == pytest.approx(0.0)
    assert brier_skill_score(np.array([0.0, 1.0, 1.0, 1.0]), clim, o) < 0


def test_auc_undefined_without_events() -> None:
    assert math.isnan(roc_auc(np.array([0.1, 0.2]), np.array([0.0, 0.0])))
    assert roc_auc(np.array([0.9, 0.1]), np.array([1.0, 0.0])) == 1.0


def test_reliable_forecast_lies_on_diagonal() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    o = (rng.uniform(0, 1, p.size) < p).astype(float)  # events happen with the stated probability
    for row in reliability_table(p, o):
        assert row["observed_frequency"] == pytest.approx(row["mean_forecast"], abs=0.02)


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        brier_score(np.array([1.2]), np.array([1.0]))
    with pytest.raises(ValueError, match="0 or 1"):
        brier_score(np.array([0.5]), np.array([2.0]))


def test_summary_counts_events() -> None:
    s = probability_scores(np.array([0.2, 0.8, np.nan]), np.array([0.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5]))
    assert s["n"] == 2 and s["events"] == 1
