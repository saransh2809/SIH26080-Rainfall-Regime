from __future__ import annotations

from datetime import date

import pytest

from rainpp.data.schema import DataKind, ProvenanceError, SchemaError
from rainpp.data.sources.base import ObservationSource
from tests.conftest import make_observation


class _StubObservations(ObservationSource):
    name = "stub"
    data_kind = DataKind.SYNTHETIC

    def __init__(self, dataset):
        self._dataset = dataset

    def _load(self, start, end):
        return self._dataset


def test_load_validates_adapter_output() -> None:
    bad = make_observation().rename({"precip_mm": "rain"})
    with pytest.raises(SchemaError):
        _StubObservations(bad).load(date(2018, 8, 1), date(2018, 8, 2))


def test_adapter_cannot_mislabel_its_data_kind() -> None:
    mislabelled = make_observation(kind="real")
    with pytest.raises(ProvenanceError, match="declares synthetic"):
        _StubObservations(mislabelled).load(date(2018, 8, 1), date(2018, 8, 2))


def test_reversed_date_range_rejected() -> None:
    with pytest.raises(ValueError, match="after end"):
        _StubObservations(make_observation()).load(date(2018, 8, 2), date(2018, 8, 1))
