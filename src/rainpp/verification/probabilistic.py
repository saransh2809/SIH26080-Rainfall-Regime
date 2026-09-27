"""Verification of probability forecasts for a binary event (rain ≥ threshold).

  Brier score   BS  = mean (p − o)²,  o ∈ {0, 1}            lower is better, 0 is perfect
  Brier skill   BSS = 1 − BS / BS_ref                          > 0 means better than the reference
  ROC AUC       discrimination: can the forecast rank events above non-events (0.5 = no skill)
  Reliability   observed event frequency within bins of forecast probability (diagonal = reliable)

Undefined values (e.g. AUC when no event occurred) are NaN, never 0.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _clean(p: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, o = np.asarray(p, dtype=float).ravel(), np.asarray(o, dtype=float).ravel()
    if p.shape != o.shape:
        raise ValueError("shape mismatch")
    keep = ~(np.isnan(p) | np.isnan(o))
    p, o = p[keep], o[keep]
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isin(o, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0 or 1")
    return p, o


def brier_score(p: np.ndarray, o: np.ndarray) -> float:
    p, o = _clean(p, o)
    return float(np.mean((p - o) ** 2)) if p.size else float("nan")


def brier_skill_score(p: np.ndarray, p_ref: np.ndarray, o: np.ndarray) -> float:
    bs, ref = brier_score(p, o), brier_score(p_ref, o)
    return 1.0 - bs / ref if ref > 0 else float("nan")


def roc_auc(p: np.ndarray, o: np.ndarray) -> float:
    p, o = _clean(p, o)
    if o.size == 0 or o.min() == o.max():
        return float("nan")
    return float(roc_auc_score(o, p))


def reliability_table(p: np.ndarray, o: np.ndarray, bins: int = 10) -> list[dict]:
    """Per probability bin: forecast count, mean forecast probability, observed frequency."""
    p, o = _clean(p, o)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        n = int(sel.sum())
        rows.append({"bin": [float(edges[b]), float(edges[b + 1])], "n": n,
                     "mean_forecast": float(p[sel].mean()) if n else float("nan"),
                     "observed_frequency": float(o[sel].mean()) if n else float("nan")})
    return rows


def probability_scores(p: np.ndarray, o: np.ndarray, p_climatology: np.ndarray) -> dict:
    p_, o_ = _clean(p, o)
    return {"n": int(o_.size), "events": int(o_.sum()), "base_rate": float(o_.mean()) if o_.size else float("nan"),
            "brier": brier_score(p, o), "bss_vs_climatology": brier_skill_score(p, p_climatology, o),
            "roc_auc": roc_auc(p, o), "mean_probability": float(p_.mean()) if p_.size else float("nan"),
            "reliability": reliability_table(p, o)}
