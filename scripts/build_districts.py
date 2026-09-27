"""Build area-overlap weights between Census-2011 districts and the IMD 0.25° land cells.

Output: data/interim/static/district_weights.{weights.npz,districts.parquet,cells.parquet}

Usage:
    python scripts/build_districts.py
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from rainpp.config import load_settings
from rainpp.spatial.districts import build_weights, load_districts

log = logging.getLogger("build_districts")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    data_dir = settings.paths.data_dir
    districts = load_districts(data_dir / "raw" / "boundaries" / "India-Districts-2011Census.shp")
    sample = xr.open_dataset(data_dir / "raw" / "imd" / "imd_rf25_2018.nc")["RAINFALL"].isel(TIME=0)
    valid = sample.notnull().values
    lat2d, lon2d = np.meshgrid(sample.LATITUDE.values, sample.LONGITUDE.values, indexing="ij")
    weights = build_weights(districts, lat2d[valid], lon2d[valid], settings.domain.resolution_deg)
    weights.save(data_dir / "interim" / "static" / "district_weights")

    cov = weights.districts["coverage_fraction"]
    log.info("%d districts x %d cells | coverage: none %d, <50%% %d, >=95%% %d", len(cov), len(weights.cells),
             int((cov == 0).sum()), int(((cov > 0) & (cov < 0.5)).sum()), int((cov >= 0.95).sum()))
    low = weights.districts[cov < 0.5].sort_values("coverage_fraction")
    log.info("districts with < 50%% coverage:\n%s", low.to_string(index=False))


if __name__ == "__main__":
    main()
