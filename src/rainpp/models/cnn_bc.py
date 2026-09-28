"""Deep-learning experiment: a small U-Net rainfall corrector (compared against LightGBM C1, not a replacement).

The LightGBM correctors see each cell with a 7x7 neighbourhood summary. The U-Net sees the whole forecast map, so
it can learn displacement and spatial structure. Inputs per (init, lead): log1p forecast rain, log1p observed
climatology for the valid day (training years), terrain elevation, distance to coast and land fraction as maps;
lead day, season and the predicted regime probabilities as constant planes. Output: expected rain per cell,
trained with the same Tweedie deviance (p = 1.5) as the LightGBM correctors, on observed land cells only.
The epoch count is chosen by early stopping on the last training year and the model is refit, as for the trees.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

TWEEDIE_P = 1.5


def tweedie_deviance_loss(log_mu: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Tweedie negative log-likelihood (up to a constant) over masked cells; mu = exp(log_mu)."""
    p = TWEEDIE_P
    loss = -y * torch.exp(log_mu * (1 - p)) / (1 - p) + torch.exp(log_mu * (2 - p)) / (2 - p)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())


class UNet(nn.Module):
    """Two-level U-Net; input H and W must be divisible by 4. The skip from the raw forecast keeps the
    network close to the forecast it corrects: it predicts log(mu) as log1p(forecast) plus a learned term."""

    def __init__(self, n_maps: int, n_scalars: int, width: int = 16) -> None:
        super().__init__()
        c = n_maps + n_scalars
        self.enc1, self.enc2, self.mid = _block(c, width), _block(width, 2 * width), _block(2 * width, 4 * width)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2), _block(4 * width, 2 * width)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2), _block(2 * width, width)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor, log_forecast: torch.Tensor) -> torch.Tensor:
        planes = scalars[:, :, None, None].expand(-1, -1, maps.shape[2], maps.shape[3])
        x = torch.cat([maps, planes], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.log(torch.expm1(log_forecast).clamp(min=0) + 0.1) + self.out(d1)[:, 0]


def pad_to_multiple(a: np.ndarray, multiple: int = 4) -> np.ndarray:
    """Zero-pad the last two axes up to a multiple (bottom/right), so the U-Net can pool twice."""
    h, w = a.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)])


class CNNCorrector:
    """Holds normalisation fitted on training samples only, trains with early stopping, predicts rain maps."""

    name = "cnn_unet"

    def __init__(self, width: int = 16, lr: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 16,
                 max_epochs: int = 60, patience: int = 6, seed: int = 42) -> None:
        self.width, self.lr, self.weight_decay = width, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience, self.seed = batch_size, max_epochs, patience, seed
        self.net: UNet | None = None
        self.epochs: int | None = None

    def _fit_norm(self, static: np.ndarray, scalars: np.ndarray) -> None:
        self.static_mean = static.mean(axis=(1, 2), keepdims=True)
        self.static_sd = static.std(axis=(1, 2), keepdims=True) + 1e-6
        self.scalar_mean, self.scalar_sd = scalars.mean(axis=0), scalars.std(axis=0) + 1e-6

    def _batch(self, data: dict, idx: np.ndarray) -> tuple[torch.Tensor, ...]:
        log_fc = np.log1p(data["forecast"][idx])
        clim = np.log1p(data["climatology"][data["doy"][idx] - 1])
        static = np.broadcast_to((data["static"] - self.static_mean) / self.static_sd,
                                 (len(idx), *data["static"].shape))
        maps = np.concatenate([log_fc[:, None], clim[:, None], static], axis=1)
        scalars = (data["scalars"][idx] - self.scalar_mean) / self.scalar_sd
        tensors = [torch.tensor(pad_to_multiple(a).astype(np.float32)) for a in (maps, log_fc)]
        out = (tensors[0], torch.tensor(scalars.astype(np.float32)), tensors[1])
        if "target" in data:
            y = data["target"][idx]
            mask = np.isfinite(y)
            out += (torch.tensor(pad_to_multiple(np.nan_to_num(y)).astype(np.float32)),
                    torch.tensor(pad_to_multiple(mask).astype(np.float32)))
        return out

    def _train(self, data: dict, epochs: int, stop: dict | None = None) -> int:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_maps = 2 + data["static"].shape[0]
        self.net = UNet(n_maps, data["scalars"].shape[1], self.width)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = len(data["forecast"])
        best, best_epoch, since = np.inf, 0, 0
        for epoch in range(1, epochs + 1):
            self.net.train()
            order = rng.permutation(n)
            for k in range(0, n, self.batch_size):
                idx = order[k:k + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs more than one sample
                    continue
                maps, scalars, log_fc, y, mask = self._batch(data, idx)
                opt.zero_grad()
                tweedie_deviance_loss(self.net(maps, scalars, log_fc), y, mask).backward()
                opt.step()
            if stop is not None:
                loss = self._loss(stop)
                if loss < best - 1e-5:
                    best, best_epoch, since = loss, epoch, 0
                else:
                    since += 1
                    if since >= self.patience:
                        break
        return best_epoch if stop is not None else epochs

    @torch.no_grad()
    def _loss(self, data: dict) -> float:
        self.net.eval()
        total, weight = 0.0, 0.0
        for k in range(0, len(data["forecast"]), self.batch_size):
            maps, scalars, log_fc, y, mask = self._batch(data, np.arange(k, min(k + self.batch_size, len(data["forecast"]))))
            total += float(tweedie_deviance_loss(self.net(maps, scalars, log_fc), y, mask)) * float(mask.sum())
            weight += float(mask.sum())
        return total / weight

    def fit(self, train: dict, stop_split: np.ndarray) -> CNNCorrector:
        """stop_split: boolean per training sample marking the early-stopping hold-out (e.g. the last year)."""
        self._fit_norm(train["static"], train["scalars"][~stop_split])
        inner = {k: (v[~stop_split] if k in _PER_SAMPLE else v) for k, v in train.items()}
        held = {k: (v[stop_split] if k in _PER_SAMPLE else v) for k, v in train.items()}
        self.epochs = max(1, self._train(inner, self.max_epochs, stop=held))
        self._fit_norm(train["static"], train["scalars"])
        self._train(train, self.epochs)
        return self

    @torch.no_grad()
    def predict(self, data: dict) -> np.ndarray:
        """Expected rain (mm) with the input grid shape; non-negative by construction."""
        self.net.eval()
        h, w = data["forecast"].shape[1:]
        out = []
        for k in range(0, len(data["forecast"]), self.batch_size):
            maps, scalars, log_fc = self._batch(data, np.arange(k, min(k + self.batch_size, len(data["forecast"]))))[:3]
            out.append(torch.exp(self.net(maps, scalars, log_fc)).numpy()[:, :h, :w])
        return np.concatenate(out)


_PER_SAMPLE = {"forecast", "doy", "scalars", "target"}
