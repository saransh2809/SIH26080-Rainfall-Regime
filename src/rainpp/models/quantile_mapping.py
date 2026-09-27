"""Baseline B1: empirical quantile mapping per grid cell and lead day.

For a forecast value x at cell c:
  1. p = mid-rank CDF of x among that cell's TRAINING forecasts: (#below + 0.5 * #equal) / n.
     The mid-rank handles the many tied zeros correctly: if 40% of forecasts are dry, a dry
     forecast maps to p = 0.2, i.e. into the dry part of the observed distribution when that
     part is at least 20%.
  2. corrected = observed training quantile at p (Hazen position p*n - 0.5, linear interpolation).
  3. Above the largest training forecast, the excess is added to the largest observation
     (additive extrapolation), so extremes are neither capped nor amplified.
Output is non-negative by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Offset that places each cell's sorted values in its own disjoint range, so one global
# searchsorted serves all cells. Must exceed any rainfall value (mm).
_CELL_OFFSET = 1.0e5


def _as_stored(values: np.ndarray) -> np.ndarray:
    """Round to float32 (the saved precision) so ties compare identically before and after save."""
    return np.asarray(values, dtype=np.float32).astype(np.float64)


class QuantileMapping:
    name = "quantile_mapping"

    def __init__(self) -> None:
        self.leads: list[int] = []
        self.cells: pd.MultiIndex | None = None
        self.sorted_fc: dict[int, np.ndarray] = {}   # lead -> (n_samples, n_cells)
        self.sorted_obs: dict[int, np.ndarray] = {}

    def fit(self, table: pd.DataFrame) -> QuantileMapping:
        """Fit on a training table with columns lead_day, lat, lon, valid_date, nwp/obs precip."""
        self.cells = pd.MultiIndex.from_frame(table[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]))
        self.leads = sorted(int(v) for v in table["lead_day"].unique())
        for lead, group in table.groupby("lead_day"):
            fc = group.pivot_table(index="valid_date", columns=["lat", "lon"], values="nwp_precip_mm")
            ob = group.pivot_table(index="valid_date", columns=["lat", "lon"], values="obs_precip_mm")
            fc, ob = fc.reindex(columns=self.cells), ob.reindex(columns=self.cells)
            if fc.isna().any().any() or ob.isna().any().any():
                raise ValueError(f"lead {lead}: every cell needs a value on every training date")
            self.sorted_fc[int(lead)] = np.sort(_as_stored(fc.to_numpy()), axis=0)
            self.sorted_obs[int(lead)] = np.sort(_as_stored(ob.to_numpy()), axis=0)
        return self

    def _cell_index(self, table: pd.DataFrame) -> np.ndarray:
        idx = self.cells.get_indexer(pd.MultiIndex.from_frame(table[["lat", "lon"]]))
        if (idx < 0).any():
            raise ValueError("prediction contains grid cells not seen in training")
        return idx

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        if self.cells is None:
            raise RuntimeError("model is not fitted")
        out = np.empty(len(table), dtype=np.float64)
        cell_all = self._cell_index(table)
        leads = table["lead_day"].to_numpy()
        values = _as_stored(table["nwp_precip_mm"].to_numpy())
        for lead in np.unique(leads):
            if int(lead) not in self.sorted_fc:
                raise ValueError(f"lead day {lead} was not fitted")
            rows = leads == lead
            out[rows] = self._map(int(lead), values[rows], cell_all[rows])
        return out

    def _map(self, lead: int, x: np.ndarray, cell: np.ndarray) -> np.ndarray:
        sfc, sob = self.sorted_fc[lead], self.sorted_obs[lead]
        n, n_cells = sfc.shape
        offsets = np.arange(n_cells) * _CELL_OFFSET
        flat = (sfc + offsets).T.ravel()                     # cell-major, globally sorted
        query = x + cell * _CELL_OFFSET
        below = np.searchsorted(flat, query, side="left") - cell * n
        upto = np.searchsorted(flat, query, side="right") - cell * n
        p = (below + 0.5 * (upto - below)) / n

        pos = np.clip(p * n - 0.5, 0.0, n - 1.0)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        w = pos - lo
        mapped = (1 - w) * sob[lo, cell] + w * sob[hi, cell]

        top_fc, top_ob = sfc[-1, cell], sob[-1, cell]
        above = x > top_fc
        mapped[above] = top_ob[above] + (x[above] - top_fc[above])
        return np.clip(mapped, 0.0, None)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "quantile_mapping.npz"
        arrays = {f"fc_{k}": v.astype(np.float32) for k, v in self.sorted_fc.items()}
        arrays |= {f"obs_{k}": v.astype(np.float32) for k, v in self.sorted_obs.items()}
        np.savez_compressed(path, lat=self.cells.get_level_values("lat").to_numpy(),
                            lon=self.cells.get_level_values("lon").to_numpy(), leads=np.array(self.leads), **arrays)
        return path

    @classmethod
    def load(cls, path: Path) -> QuantileMapping:
        model = cls()
        with np.load(path) as data:
            model.cells = pd.MultiIndex.from_arrays([data["lat"], data["lon"]], names=["lat", "lon"])
            model.leads = [int(v) for v in data["leads"]]
            for lead in model.leads:
                model.sorted_fc[lead] = data[f"fc_{lead}"].astype(np.float64)
                model.sorted_obs[lead] = data[f"obs_{lead}"].astype(np.float64)
        return model
