"""CNN tests on a SYNTHETIC spatial-pattern problem (fast, CPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rainpp.regimes.classifier import CLASSES
from rainpp.regimes.cnn import CNNRegimeClassifier


def _data(n: int, seed: int):
    """Class is encoded by WHERE a blob sits in channel 0, which only a spatial model can read."""
    rng = np.random.default_rng(seed)
    maps = rng.normal(0, 1, (n, 2, 12, 16)).astype(np.float32)
    y = rng.choice(CLASSES, n)
    corners = {"NORMAL": (0, 0), "ACTIVE": (0, 8), "BREAK": (6, 0), "MONSOON_DEPRESSION": (6, 8)}
    for i, c in enumerate(y):
        r, k = corners[c]
        maps[i, 0, r:r + 6, k:k + 8] += 3.0
    X = pd.DataFrame({"lead_day": rng.integers(1, 4, n).astype(float)})
    return maps, X, y


def test_cnn_learns_spatial_pattern_and_outputs_probabilities() -> None:
    maps, X, y = _data(600, 0)
    stop = _data(150, 1)
    model = CNNRegimeClassifier(["lead_day"], max_epochs=40, patience=5).fit(maps, X, y, stop=stop)
    assert 1 <= model.epochs <= 40
    tm, tX, ty = _data(200, 2)
    proba = model.predict_proba(tm, tX)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    accuracy = np.mean(proba.idxmax(axis=1).to_numpy() == ty)
    assert accuracy > 0.9


def test_cnn_is_reproducible_with_fixed_seed() -> None:
    maps, X, y = _data(200, 3)
    a = CNNRegimeClassifier(["lead_day"], max_epochs=3).fit(maps, X, y).predict_proba(maps[:5], X[:5])
    b = CNNRegimeClassifier(["lead_day"], max_epochs=3).fit(maps, X, y).predict_proba(maps[:5], X[:5])
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-6)
