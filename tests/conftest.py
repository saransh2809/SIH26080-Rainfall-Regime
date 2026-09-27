"""Shared fixtures. Datasets here are tiny SYNTHETIC test fixtures, never used outside tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rainpp.data.schema import DATA_KIND_ATTR, PRECIP_VAR, SOURCE_ATTR

LAT = np.array([10.0, 10.25, 10.5])
LON = np.array([76.0, 76.25])


def make_observation(kind: str = "synthetic", values: np.ndarray | None = None) -> xr.Dataset:
    time = pd.date_range("2018-08-01", periods=2)
    if values is None:
        values = np.full((len(time), LAT.size, LON.size), 5.0)
    return xr.Dataset(
        {PRECIP_VAR: (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": LAT, "lon": LON},
        attrs={DATA_KIND_ATTR: kind, SOURCE_ATTR: "test_fixture"},
    )


def make_forecast(kind: str = "synthetic") -> xr.Dataset:
    shape = (1, 2, 1, LAT.size, LON.size)
    return xr.Dataset(
        {PRECIP_VAR: (("init_time", "lead_day", "member", "lat", "lon"), np.full(shape, 3.0))},
        coords={
            "init_time": pd.to_datetime(["2018-08-01"]),
            "lead_day": [1, 2],
            "member": ["c00"],
            "lat": LAT,
            "lon": LON,
        },
        attrs={DATA_KIND_ATTR: kind, SOURCE_ATTR: "test_fixture"},
    )


@pytest.fixture
def observation() -> xr.Dataset:
    return make_observation()


@pytest.fixture
def forecast() -> xr.Dataset:
    return make_forecast()
