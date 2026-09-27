"""Provider-independent data source interfaces.

Model code depends only on these interfaces and the canonical schema, never on a
provider's raw format. Concrete adapters (IMD, GEFS, GFS, NCMRWF, demo) implement ``_load``;
the public ``load`` always validates the result before handing it on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from typing import ClassVar

import xarray as xr

from rainpp.data.schema import (
    DATA_KIND_ATTR,
    DataKind,
    ProvenanceError,
    data_kind,
    validate_forecast,
    validate_observation,
)


def _check_declared_kind(ds: xr.Dataset, declared: DataKind, source: str) -> None:
    if data_kind(ds) is not declared:
        raise ProvenanceError(
            f"{source} declares {declared.value} data but returned attrs['{DATA_KIND_ATTR}']="
            f"{ds.attrs.get(DATA_KIND_ATTR)!r}"
        )


class ForecastSource(ABC):
    """Raw NWP rainfall forecasts aligned to the observation rain day."""

    name: ClassVar[str]
    data_kind: ClassVar[DataKind]

    def load(
        self, init_dates: Sequence[date], lead_days: Sequence[int], members: Sequence[str]
    ) -> xr.Dataset:
        ds = self._load(init_dates, lead_days, members)
        validate_forecast(ds)
        _check_declared_kind(ds, self.data_kind, self.name)
        return ds

    @abstractmethod
    def _load(
        self, init_dates: Sequence[date], lead_days: Sequence[int], members: Sequence[str]
    ) -> xr.Dataset:
        """Return dims (init_time, lead_day, member, lat, lon) with variable precip_mm."""


class ObservationSource(ABC):
    """Observed daily rainfall used as training target and verification truth."""

    name: ClassVar[str]
    data_kind: ClassVar[DataKind]

    def load(self, start: date, end: date) -> xr.Dataset:
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        ds = self._load(start, end)
        validate_observation(ds)
        _check_declared_kind(ds, self.data_kind, self.name)
        return ds

    @abstractmethod
    def _load(self, start: date, end: date) -> xr.Dataset:
        """Return dims (time, lat, lon) with variable precip_mm."""
