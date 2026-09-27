from __future__ import annotations

import numpy as np
import pytest

from rainpp.data.schema import (
    DATA_KIND_ATTR,
    DataKind,
    ProvenanceError,
    SchemaError,
    assert_same_kind,
    validate_forecast,
    validate_observation,
)
from tests.conftest import make_forecast, make_observation


def test_valid_datasets_pass(observation, forecast) -> None:
    validate_observation(observation)
    validate_forecast(forecast)


def test_missing_data_kind_rejected(observation) -> None:
    del observation.attrs[DATA_KIND_ATTR]
    with pytest.raises(ProvenanceError):
        validate_observation(observation)


def test_real_and_synthetic_cannot_be_combined() -> None:
    with pytest.raises(ProvenanceError, match="different kinds"):
        assert_same_kind(make_observation("real"), make_forecast("synthetic"))


def test_same_kind_returns_kind() -> None:
    assert assert_same_kind(make_observation("real"), make_forecast("real")) is DataKind.REAL


def test_negative_rainfall_rejected() -> None:
    values = np.full((2, 3, 2), 1.0)
    values[0, 0, 0] = -0.5
    with pytest.raises(SchemaError, match="negative"):
        validate_observation(make_observation(values=values))


def test_partial_missing_values_allowed_but_all_missing_rejected() -> None:
    values = np.full((2, 3, 2), 1.0)
    values[0, 0, 0] = np.nan
    validate_observation(make_observation(values=values))
    with pytest.raises(SchemaError, match="entirely missing"):
        validate_observation(make_observation(values=np.full((2, 3, 2), np.nan)))


def test_wrong_dimension_order_rejected(observation) -> None:
    with pytest.raises(SchemaError, match="dims"):
        validate_observation(observation.transpose("lat", "time", "lon"))


def test_descending_latitude_rejected(observation) -> None:
    with pytest.raises(SchemaError, match="lat"):
        validate_observation(observation.isel(lat=slice(None, None, -1)))
