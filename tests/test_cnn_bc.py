"""U-Net corrector on a SYNTHETIC problem: the forecast is displaced and 2x too wet in the west half."""

from __future__ import annotations

import numpy as np
import torch

from rainpp.models.cnn_bc import CNNCorrector, pad_to_multiple, tweedie_deviance_loss


def _data(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    h, w = 10, 14
    truth = rng.gamma(0.7, 8.0, (n, h, w))
    forecast = np.roll(truth, 1, axis=2) * np.where(np.arange(w) < w // 2, 2.0, 1.0)
    target = truth.copy()
    target[:, 0, :] = np.nan  # sea cells: no observation
    return {"forecast": forecast, "target": target, "doy": np.full(n, 200),
            "climatology": np.full((366, h, w), 5.0), "static": rng.normal(size=(3, h, w)),
            "scalars": rng.normal(size=(n, 4))}


def test_pad_to_multiple_pads_bottom_right_only() -> None:
    a = np.ones((2, 10, 14))
    p = pad_to_multiple(a)
    assert p.shape == (2, 12, 16) and p[:, :10, :14].all() and not p[:, 10:, :].any()


def test_tweedie_loss_ignores_masked_cells_and_is_minimised_at_the_truth() -> None:
    y = torch.tensor([[3.0, 100.0]])
    mask = torch.tensor([[1.0, 0.0]])
    at_truth = tweedie_deviance_loss(torch.log(torch.tensor([[3.0, 1.0]])), y, mask)
    off = tweedie_deviance_loss(torch.log(torch.tensor([[6.0, 1.0]])), y, mask)
    assert at_truth < off


def test_corrector_learns_the_bias_and_stays_non_negative() -> None:
    train, test = _data(120, 0), _data(40, 1)
    stop = np.arange(120) >= 100
    model = CNNCorrector(width=8, max_epochs=25, patience=4, batch_size=8).fit(train, stop)
    pred = model.predict(test)
    assert pred.shape == test["forecast"].shape and (pred >= 0).all()
    land = np.isfinite(test["target"])
    raw_rmse = np.sqrt(np.mean((test["forecast"][land] - test["target"][land]) ** 2))
    cnn_rmse = np.sqrt(np.mean((pred[land] - test["target"][land]) ** 2))
    assert cnn_rmse < raw_rmse
