"""Deep-learning experiment: a small hybrid CNN regime classifier.

Maps (channels x 40 x 70 at 1°) pass through three conv layers and global average pooling; the
pooled vector is concatenated with scalar features and classified. Maps are standardised as
anomalies from the per-pixel TRAINING mean, divided by the per-channel training SD. The network is
deliberately small (~15k parameters) because only ~2,200 labelled training samples exist.
Training uses class-weighted cross-entropy; the epoch count is chosen by early stopping on held-out
training data and the model is then refit, mirroring the tree-model protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from rainpp.regimes.classifier import CLASSES


class HybridCNN(nn.Module):
    def __init__(self, n_channels: int, n_scalars: int, n_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.maps = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 + n_scalars, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes))

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.maps(maps), scalars], dim=1))


class CNNRegimeClassifier:
    name = "CNN"

    def __init__(self, scalar_features: list[str], seed: int = 42, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 64, max_epochs: int = 200, patience: int = 20) -> None:
        self.scalar_features = list(scalar_features)
        self.seed, self.lr, self.weight_decay = seed, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.net: HybridCNN | None = None
        self.epochs: int | None = None

    # ---- normalisation (fitted on training data only) ----
    def _fit_norm(self, maps: np.ndarray, scalars: np.ndarray) -> None:
        self.map_mean = maps.mean(axis=0, keepdims=True)
        self.map_sd = (maps - self.map_mean).std(axis=(0, 2, 3), keepdims=True) + 1e-6
        self.scalar_mean = np.nanmean(scalars, axis=0)
        self.scalar_sd = np.nanstd(scalars, axis=0) + 1e-6

    def _tensors(self, maps: np.ndarray, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        m = (maps - self.map_mean) / self.map_sd
        s = (X[self.scalar_features].to_numpy(np.float64) - self.scalar_mean) / self.scalar_sd
        return torch.tensor(m, dtype=torch.float32), torch.tensor(np.nan_to_num(s), dtype=torch.float32)

    def _train(self, maps, X, y, epochs: int, stop=None) -> int:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = HybridCNN(maps.shape[1], len(self.scalar_features))
        m, s = self._tensors(maps, X)
        target = torch.tensor(pd.Series(y).map({c: i for i, c in enumerate(CLASSES)}).to_numpy())
        counts = np.bincount(target.numpy(), minlength=len(CLASSES)).astype(float)
        weights = torch.tensor(len(target) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if stop is not None:
            sm, ss = self._tensors(stop[0], stop[1])
            st = torch.tensor(pd.Series(stop[2]).map({c: i for i, c in enumerate(CLASSES)}).to_numpy())
        best, best_epoch, since = np.inf, 0, 0
        for epoch in range(1, epochs + 1):
            self.net.train()
            order = rng.permutation(len(target))
            for k in range(0, len(order), self.batch_size):
                idx = order[k:k + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss_fn(self.net(m[idx], s[idx]), target[idx]).backward()
                opt.step()
            if stop is not None:
                self.net.eval()
                with torch.no_grad():
                    val = float(loss_fn(self.net(sm, ss), st))
                if val < best - 1e-4:
                    best, best_epoch, since = val, epoch, 0
                else:
                    since += 1
                    if since >= self.patience:
                        break
        return best_epoch if stop is not None else epochs

    def fit(self, maps: np.ndarray, X: pd.DataFrame, y, stop: tuple | None = None) -> CNNRegimeClassifier:
        """With stop=(maps, X, y) held-out data, first choose the epoch count, then refit on all of maps/X/y."""
        self._fit_norm(maps, X[self.scalar_features].to_numpy(np.float64))
        if stop is not None:
            self.epochs = max(1, self._train(maps, X, y, self.max_epochs, stop))
        else:
            self.epochs = self.epochs or self.max_epochs
        self._train(maps, X, y, self.epochs)
        return self

    def predict_proba(self, maps: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
        self.net.eval()
        m, s = self._tensors(maps, X)
        with torch.no_grad():
            proba = torch.softmax(self.net(m, s), dim=1).numpy()
        return pd.DataFrame(proba, columns=CLASSES)
