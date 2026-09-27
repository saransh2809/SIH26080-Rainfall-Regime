from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rainpp.data.sources.imd import IMDGridded, download_year, year_file


def _write_imd_like_file(raw_dir: Path, year: int) -> None:
    """Tiny file in IMD's native layout (upper-case names). Synthetic, for tests only."""
    time = pd.date_range(f"{year}-08-01", periods=3)
    lat, lon = np.array([10.0, 10.25]), np.array([76.0, 76.25, 76.5])
    rain = np.arange(time.size * lat.size * lon.size, dtype="float32").reshape(time.size, lat.size, lon.size)
    rain[:, 0, 0] = np.nan
    xr.Dataset(
        {"RAINFALL": (("TIME", "LATITUDE", "LONGITUDE"), rain)},
        coords={"TIME": time, "LATITUDE": lat, "LONGITUDE": lon},
    ).to_netcdf(year_file(raw_dir, year))


class _SyntheticIMD(IMDGridded):
    data_kind = IMDGridded.data_kind.SYNTHETIC

    def _load(self, start, end):
        ds = super()._load(start, end)
        ds.attrs["data_kind"] = "synthetic"
        return ds


def test_native_layout_converted_to_canonical(tmp_path: Path) -> None:
    _write_imd_like_file(tmp_path, 2018)
    ds = _SyntheticIMD(tmp_path).load(date(2018, 8, 1), date(2018, 8, 2))
    assert ds["precip_mm"].dims == ("time", "lat", "lon")
    assert ds.sizes["time"] == 2
    assert np.isnan(ds["precip_mm"].values[:, 0, 0]).all()


def test_multi_year_range_is_concatenated_in_order(tmp_path: Path) -> None:
    _write_imd_like_file(tmp_path, 2010)
    _write_imd_like_file(tmp_path, 2011)
    ds = _SyntheticIMD(tmp_path).load(date(2010, 8, 2), date(2011, 8, 2))
    assert ds.sizes["time"] == 4  # 2-3 Aug 2010 + 1-2 Aug 2011
    assert pd.DatetimeIndex(ds.time.values).is_monotonic_increasing


def test_missing_year_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="imd_rf25_2019"):
        IMDGridded(tmp_path).load(date(2019, 6, 1), date(2019, 6, 2))


def test_download_rejects_impossible_year(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        download_year(1850, tmp_path)


def test_download_skips_existing_file(tmp_path: Path) -> None:
    _write_imd_like_file(tmp_path, 2018)
    assert download_year(2018, tmp_path) == year_file(tmp_path, 2018)
