"""Paired block-bootstrap confidence intervals for the difference between two forecasts.

Rainfall errors are correlated across neighbouring cells on a day and across consecutive days
(monsoon spells last several days), so rows are resampled in blocks of consecutive valid dates, with
every cell and lead of a date kept together. Both forecasts are scored on the SAME resample, so the
interval is for the paired difference. An interval that excludes 0 is the minimum evidence required
before claiming one forecast is better.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def block_bootstrap_difference(table: pd.DataFrame, metric: Callable[[np.ndarray, np.ndarray], float],
                               col_a: str, col_b: str, obs_col: str = "obs_precip_mm", block_days: int = 5,
                               n_boot: int = 500, seed: int = 0, higher_is_better: bool = False) -> dict:
    """metric(b) − metric(a), oriented so that a positive 'improvement' means b is better than a."""
    dates = np.sort(table["valid_date"].unique())
    date_pos = pd.Series(np.arange(dates.size), index=dates)
    block_of_row = (date_pos.reindex(table["valid_date"]).to_numpy() // block_days)
    n_blocks = int(block_of_row.max()) + 1
    rows_by_block = [np.flatnonzero(block_of_row == b) for b in range(n_blocks)]
    a, b, o = (table[c].to_numpy() for c in (col_a, col_b, obs_col))
    sign = 1.0 if higher_is_better else -1.0

    def improvement(idx: np.ndarray) -> float:
        return sign * (metric(b[idx], o[idx]) - metric(a[idx], o[idx]))

    point = improvement(np.arange(len(table)))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for k in range(n_boot):
        chosen = rng.integers(0, n_blocks, n_blocks)
        samples[k] = improvement(np.concatenate([rows_by_block[c] for c in chosen]))
    samples = samples[~np.isnan(samples)]
    lo, hi = (np.percentile(samples, [2.5, 97.5]) if samples.size else (np.nan, np.nan))
    return {"improvement": float(point), "ci95_low": float(lo), "ci95_high": float(hi),
            "significant": bool(samples.size and (lo > 0 or hi < 0)), "n_boot_valid": int(samples.size),
            "block_days": block_days, "n_blocks": n_blocks}
