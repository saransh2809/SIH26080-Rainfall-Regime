"""Forecast verification metrics.

Continuous (rainfall amount): RMSE, MAE, mean error (bias), Pearson correlation.
Categorical (event = rainfall >= threshold), from the 2x2 contingency table
    hits a, false alarms b, misses c, correct negatives d, n = a + b + c + d:
    POD  = a / (a + c)                       probability of detection
    FAR  = b / (a + b)                       false alarm RATIO
    CSI  = a / (a + b + c)                   critical success index (threat score)
    ETS  = (a - a_r) / (a + b + c - a_r),    a_r = (a + b)(a + c) / n   (Gilbert skill score)
    FBI  = (a + b) / (a + c)                 frequency bias
Spatial: Fractions Skill Score (Roberts & Lean 2008, Mon. Wea. Rev.)
    FSS = 1 - sum (Pf - Po)^2 / (sum Pf^2 + sum Po^2), with Pf, Po the fraction of event cells in an
    n x n neighbourhood, accumulated over all days before taking the ratio.

Undefined scores (zero denominator, e.g. POD when no event was observed) are returned as NaN,
never 0, and every result carries the counts it was computed from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _valid_pairs(forecast: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, o = np.asarray(forecast, dtype=float).ravel(), np.asarray(observed, dtype=float).ravel()
    if f.shape != o.shape:
        raise ValueError(f"shape mismatch: forecast {f.shape} vs observed {o.shape}")
    keep = ~(np.isnan(f) | np.isnan(o))
    return f[keep], o[keep]


def calculate_rmse(forecast: np.ndarray, observed: np.ndarray) -> float:
    f, o = _valid_pairs(forecast, observed)
    return float(np.sqrt(np.mean((f - o) ** 2))) if f.size else float("nan")


def calculate_mae(forecast: np.ndarray, observed: np.ndarray) -> float:
    f, o = _valid_pairs(forecast, observed)
    return float(np.mean(np.abs(f - o))) if f.size else float("nan")


def calculate_bias(forecast: np.ndarray, observed: np.ndarray) -> float:
    """Mean error, forecast minus observed (mm). Positive = over-forecasting."""
    f, o = _valid_pairs(forecast, observed)
    return float(np.mean(f - o)) if f.size else float("nan")


def calculate_correlation(forecast: np.ndarray, observed: np.ndarray) -> float:
    f, o = _valid_pairs(forecast, observed)
    if f.size < 2 or np.std(f) == 0 or np.std(o) == 0:
        return float("nan")
    return float(np.corrcoef(f, o)[0, 1])


@dataclass(frozen=True)
class Contingency:
    threshold_mm: float
    hits: int
    false_alarms: int
    misses: int
    correct_negatives: int

    @property
    def n(self) -> int:
        return self.hits + self.false_alarms + self.misses + self.correct_negatives

    @property
    def pod(self) -> float:
        return _ratio(self.hits, self.hits + self.misses)

    @property
    def far(self) -> float:
        return _ratio(self.false_alarms, self.hits + self.false_alarms)

    @property
    def csi(self) -> float:
        return _ratio(self.hits, self.hits + self.false_alarms + self.misses)

    @property
    def ets(self) -> float:
        if self.n == 0:
            return float("nan")
        a_random = (self.hits + self.false_alarms) * (self.hits + self.misses) / self.n
        return _ratio(self.hits - a_random, self.hits + self.false_alarms + self.misses - a_random)

    @property
    def frequency_bias(self) -> float:
        return _ratio(self.hits + self.false_alarms, self.hits + self.misses)

    def as_dict(self) -> dict:
        return {**asdict(self), "n": self.n, "pod": self.pod, "far": self.far, "csi": self.csi,
                "ets": self.ets, "frequency_bias": self.frequency_bias}


def contingency(forecast: np.ndarray, observed: np.ndarray, threshold_mm: float) -> Contingency:
    """2x2 table for the event rainfall >= threshold, over cells where both values exist."""
    f, o = _valid_pairs(forecast, observed)
    fe, oe = f >= threshold_mm, o >= threshold_mm
    return Contingency(threshold_mm, int(np.sum(fe & oe)), int(np.sum(fe & ~oe)),
                       int(np.sum(~fe & oe)), int(np.sum(~fe & ~oe)))


def calculate_pod(forecast, observed, threshold_mm: float) -> float:
    return contingency(forecast, observed, threshold_mm).pod


def calculate_far(forecast, observed, threshold_mm: float) -> float:
    return contingency(forecast, observed, threshold_mm).far


def calculate_csi(forecast, observed, threshold_mm: float) -> float:
    return contingency(forecast, observed, threshold_mm).csi


def calculate_ets(forecast, observed, threshold_mm: float) -> float:
    return contingency(forecast, observed, threshold_mm).ets


def _neighbourhood_fractions(events: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    """Fraction of VALID cells in each size x size window that are events (NaN where none valid)."""
    counts = uniform_filter(np.where(valid, events, 0.0), size=size, mode="constant")
    valid_counts = uniform_filter(valid.astype(float), size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid_counts > 1e-12, counts / valid_counts, np.nan)


def calculate_fss(forecast: np.ndarray, observed: np.ndarray, threshold_mm: float, size: int) -> float:
    """Fractions Skill Score over one or more 2-D fields (dims: [..., lat, lon]).

    NaN cells (e.g. outside India) are excluded from both fields. Returns NaN when neither field
    has any event, because the score is undefined there.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("neighbourhood size must be a positive odd number of cells")
    f, o = np.asarray(forecast, dtype=float), np.asarray(observed, dtype=float)
    if f.shape != o.shape or f.ndim < 2:
        raise ValueError("forecast and observed must be matching arrays with at least 2 dims")
    f2, o2 = f.reshape(-1, *f.shape[-2:]), o.reshape(-1, *o.shape[-2:])
    num = den = 0.0
    for fi, oi in zip(f2, o2, strict=True):
        valid = ~(np.isnan(fi) | np.isnan(oi))
        pf = _neighbourhood_fractions(np.nan_to_num(fi) >= threshold_mm, valid, size)
        po = _neighbourhood_fractions(np.nan_to_num(oi) >= threshold_mm, valid, size)
        keep = valid & ~np.isnan(pf) & ~np.isnan(po)
        num += float(np.sum((pf[keep] - po[keep]) ** 2))
        den += float(np.sum(pf[keep] ** 2) + np.sum(po[keep] ** 2))
    return 1.0 - num / den if den > 0 else float("nan")
