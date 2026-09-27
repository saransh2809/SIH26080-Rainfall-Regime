"""Canonical dataset layout and provenance rules shared by every data source.

Every dataset entering the pipeline must:
  * use the canonical dimension and variable names below, and
  * declare ``attrs["data_kind"]`` as ``real`` or ``synthetic`` and ``attrs["source"]``.

``assert_same_kind`` is the guard that keeps synthetic data out of real training/evaluation.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import xarray as xr

DATA_KIND_ATTR = "data_kind"
SOURCE_ATTR = "source"
PRECIP_VAR = "precip_mm"

FORECAST_DIMS = ("init_time", "lead_day", "member", "lat", "lon")
OBSERVATION_DIMS = ("time", "lat", "lon")


class DataKind(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class SchemaError(ValueError):
    """Dataset does not follow the canonical layout."""


class ProvenanceError(ValueError):
    """Dataset provenance is missing or datasets of different kinds were combined."""


def data_kind(ds: xr.Dataset) -> DataKind:
    """Return the declared data kind, raising if it is missing or unknown."""
    value = ds.attrs.get(DATA_KIND_ATTR)
    try:
        return DataKind(value)
    except ValueError:
        raise ProvenanceError(
            f"dataset must declare attrs['{DATA_KIND_ATTR}'] as 'real' or 'synthetic', got {value!r}"
        ) from None


def assert_same_kind(*datasets: xr.Dataset) -> DataKind:
    """Raise ProvenanceError unless all datasets share one data kind; return that kind."""
    if not datasets:
        raise ValueError("at least one dataset is required")
    kinds = {data_kind(ds) for ds in datasets}
    if len(kinds) > 1:
        raise ProvenanceError(f"refusing to combine datasets of different kinds: {sorted(k.value for k in kinds)}")
    return kinds.pop()


def _validate(ds: xr.Dataset, expected_dims: tuple[str, ...]) -> None:
    data_kind(ds)
    if not ds.attrs.get(SOURCE_ATTR):
        raise ProvenanceError(f"dataset must declare attrs['{SOURCE_ATTR}']")
    if PRECIP_VAR not in ds:
        raise SchemaError(f"missing variable '{PRECIP_VAR}'")
    dims = ds[PRECIP_VAR].dims
    if tuple(dims) != expected_dims:
        raise SchemaError(f"'{PRECIP_VAR}' dims must be {expected_dims}, got {tuple(dims)}")
    for coord in ("lat", "lon"):
        values = ds[coord].values
        if values.size > 1 and not np.all(np.diff(values) > 0):
            raise SchemaError(f"'{coord}' must be strictly increasing")
    precip = ds[PRECIP_VAR].values
    if np.all(np.isnan(precip)):
        raise SchemaError(f"'{PRECIP_VAR}' is entirely missing")
    if np.nanmin(precip) < 0:
        raise SchemaError(f"'{PRECIP_VAR}' contains negative rainfall")


def validate_forecast(ds: xr.Dataset) -> None:
    """Check a forecast dataset against the canonical layout."""
    _validate(ds, FORECAST_DIMS)


def validate_observation(ds: xr.Dataset) -> None:
    """Check an observation dataset against the canonical layout."""
    _validate(ds, OBSERVATION_DIMS)
