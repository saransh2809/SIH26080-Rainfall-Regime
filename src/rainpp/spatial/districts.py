"""Grid → district aggregation by exact area overlap.

Each 0.25° cell is a polygon; a district's value is the average of the cells it overlaps, weighted
by overlap area (computed in an equal-area projection). Only cells that have observations are used,
so coastal districts are not diluted by sea cells. Every district carries its coverage fraction;
districts with no covered area return NaN rather than a borrowed value.

Two aggregations are provided:
  mean — area-weighted mean (rainfall amount)
  max  — maximum over overlapping cells (e.g. probability of heavy rain somewhere in the district)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from shapely import box, make_valid

EQUAL_AREA_CRS = "EPSG:6933"


def load_districts(path: Path) -> gpd.GeoDataFrame:
    """Survey-of-India-based Census-2011 districts (DataMeet); invalid polygons repaired."""
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf = gdf.rename(columns={"censuscode": "district_id", "DISTRICT": "district_name", "ST_NM": "state_name"})
    gdf["district_id"] = pd.to_numeric(gdf["district_id"], errors="raise").astype(int)  # stored as text in the .dbf
    if gdf["district_id"].duplicated().any():
        raise ValueError("district ids are not unique")
    return gdf[["district_id", "district_name", "state_name", "geometry"]].reset_index(drop=True)


def cell_polygons(lats: np.ndarray, lons: np.ndarray, resolution_deg: float) -> gpd.GeoDataFrame:
    """One square polygon per (lat, lon) cell centre."""
    h = resolution_deg / 2
    return gpd.GeoDataFrame({"lat": lats, "lon": lons},
                            geometry=[box(x - h, y - h, x + h, y + h) for y, x in zip(lats, lons, strict=True)],
                            crs="EPSG:4326")


@dataclass
class DistrictWeights:
    districts: pd.DataFrame          # district_id, district_name, state_name, coverage_fraction
    cells: pd.DataFrame              # lat, lon (column order of the weight matrix)
    weights: sparse.csr_matrix       # (n_districts, n_cells), rows sum to 1 where covered

    def aggregate_mean(self, values: np.ndarray) -> np.ndarray:
        """values: (..., n_cells) → (..., n_districts); NaN for districts with no coverage."""
        v = np.asarray(values, dtype=float)
        if np.isnan(v).any():
            raise ValueError("cell values contain NaN; mask them before aggregating")
        out = (self.weights @ v.reshape(-1, v.shape[-1]).T).T.reshape(*v.shape[:-1], -1)
        out[..., self.districts["coverage_fraction"].to_numpy() == 0] = np.nan
        return out

    def aggregate_max(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float).reshape(-1, len(self.cells))
        out = np.full((v.shape[0], len(self.districts)), np.nan)
        w = self.weights.tocsr()
        for d in range(len(self.districts)):
            idx = w.indices[w.indptr[d]:w.indptr[d + 1]]
            if idx.size:
                out[:, d] = v[:, idx].max(axis=1)
        return out.reshape(*np.asarray(values).shape[:-1], -1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(path.with_suffix(".weights.npz"), self.weights)
        self.districts.to_parquet(path.with_suffix(".districts.parquet"), index=False)
        self.cells.to_parquet(path.with_suffix(".cells.parquet"), index=False)

    @classmethod
    def load(cls, path: Path) -> DistrictWeights:
        return cls(pd.read_parquet(path.with_suffix(".districts.parquet")),
                   pd.read_parquet(path.with_suffix(".cells.parquet")),
                   sparse.load_npz(path.with_suffix(".weights.npz")).tocsr())


def build_weights(districts: gpd.GeoDataFrame, lats: np.ndarray, lons: np.ndarray,
                  resolution_deg: float) -> DistrictWeights:
    """Area-overlap weights between districts and the given (valid) cells."""
    cells = cell_polygons(lats, lons, resolution_deg)
    d_ea, c_ea = districts.to_crs(EQUAL_AREA_CRS), cells.to_crs(EQUAL_AREA_CRS)
    pairs = gpd.sjoin(d_ea[["geometry"]], c_ea[["geometry"]], predicate="intersects", how="inner")
    d_idx, c_idx = pairs.index.to_numpy(), pairs["index_right"].to_numpy()
    overlap = d_ea.geometry.values[d_idx].intersection(c_ea.geometry.values[c_idx]).area
    keep = overlap > 0
    d_idx, c_idx, overlap = d_idx[keep], c_idx[keep], np.asarray(overlap)[keep]

    raw = sparse.csr_matrix((overlap, (d_idx, c_idx)), shape=(len(districts), len(cells)))
    covered_area = np.asarray(raw.sum(axis=1)).ravel()
    district_area = d_ea.geometry.area.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = sparse.diags(np.where(covered_area > 0, 1.0 / covered_area, 0.0))
    table = districts.drop(columns="geometry").copy()
    table["coverage_fraction"] = np.clip(covered_area / district_area, 0.0, 1.0)
    return DistrictWeights(table.reset_index(drop=True), pd.DataFrame({"lat": lats, "lon": lons}),
                           (norm @ raw).tocsr())
