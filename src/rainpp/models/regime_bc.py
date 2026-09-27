"""Regime-aware correctors (Phase 6).

C1 is the global LightGBM corrector given regime inputs as features (see scripts/run_regime_correction.py);
it needs no class of its own. C2 below trains one corrector per synoptic regime and routes each row by
its (predicted) regime. Regimes with too few training rows fall back to a pooled model, and the
fallback is recorded rather than hidden.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rainpp.models.global_bc import LGBMCorrector
from rainpp.models.quantile_mapping import QuantileMapping

log = logging.getLogger(__name__)


class RegimeQuantileMapping:
    """C3: quantile mapping fitted separately per (predicted) synoptic regime, per cell and lead.

    Distribution-preserving, so it keeps heavy-rain frequency like B1 while letting the mapping differ by
    regime. Regimes with fewer than `min_days` training days per lead use the global mapping (recorded).
    """

    name = "regime_quantile_mapping"

    def __init__(self, regime_column: str, min_days: int = 40) -> None:
        self.regime_column, self.min_days = regime_column, min_days
        self.global_qm: QuantileMapping | None = None
        self.by_regime: dict[str, QuantileMapping] = {}
        self.fallback_regimes: list[str] = []

    def fit(self, train: pd.DataFrame) -> RegimeQuantileMapping:
        self.global_qm = QuantileMapping().fit(train)
        for regime, group in train.groupby(self.regime_column):
            days = group.groupby("lead_day")["valid_date"].nunique().min()
            if days < self.min_days:
                self.fallback_regimes.append(str(regime))
                log.info("regime %s has %d days per lead (< %d): uses global mapping", regime, days, self.min_days)
                continue
            self.by_regime[str(regime)] = QuantileMapping().fit(group)
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        out = self.global_qm.predict(table)
        regimes = table[self.regime_column].astype(str).to_numpy()
        for regime, qm in self.by_regime.items():
            rows = regimes == regime
            if rows.any():
                out[rows] = qm.predict(table[rows])
        return out


class RegimeSplitCorrector:
    name = "regime_split_lgbm"

    def __init__(self, features: list[str], params: dict, rounds: int, regime_column: str,
                 min_rows: int = 200_000) -> None:
        self.features, self.params, self.rounds = list(features), dict(params), rounds
        self.regime_column, self.min_rows = regime_column, min_rows
        self.models: dict[str, LGBMCorrector] = {}
        self.fallback: LGBMCorrector | None = None
        self.fallback_regimes: list[str] = []

    def fit(self, train: pd.DataFrame) -> RegimeSplitCorrector:
        self.fallback = LGBMCorrector(self.features, self.params).fit_fixed_rounds(train, self.rounds)
        for regime, group in train.groupby(self.regime_column):
            if len(group) < self.min_rows:
                self.fallback_regimes.append(str(regime))
                log.info("regime %s has %d rows (< %d): uses pooled model", regime, len(group), self.min_rows)
                continue
            self.models[str(regime)] = LGBMCorrector(self.features, self.params).fit_fixed_rounds(group, self.rounds)
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        out = self.fallback.predict(table)
        regimes = table[self.regime_column].astype(str).to_numpy()
        for regime, model in self.models.items():
            rows = regimes == regime
            if rows.any():
                out[rows] = model.predict(table[rows])
        return out

    def routing(self, table: pd.DataFrame) -> dict[str, int]:
        """How many rows each specialised model (or the pooled fallback) handled."""
        regimes = table[self.regime_column].astype(str)
        return {r: int((regimes == r).sum()) if r in self.models else 0 for r in regimes.unique()} | {
            "pooled_fallback": int((~regimes.isin(list(self.models))).sum())}
