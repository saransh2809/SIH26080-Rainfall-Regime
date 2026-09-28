from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from rainpp.config import load_settings
from rainpp.data.sources.gefs_operational import (
    GEFSOperational,
    _find_labelled,
    file_path,
    upsample_half_degree,
)

# Verbatim lines from gec00.t00z.pgrb2s.0p25.f015.idx (2026-09-28 run)
IDX_F015 = """\
17:8113226:d=2026092800:CPOFP:surface:15 hour fcst:ENS=low-res ctl
18:8714653:d=2026092800:APCP:surface:12-15 hour acc fcst:ENS=low-res ctl
19:9154494:d=2026092800:CSNOW:surface:12-15 hour ave fcst:ENS=low-res ctl
"""


def test_file_path_matches_bucket_layout() -> None:
    assert file_path(date(2026, 9, 28), "s", 15) == \
        "noaa-gefs-pds/gefs.20260928/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f015"
    assert file_path(date(2026, 9, 28), "a", 3).endswith("pgrb2ap5/gec00.t00z.pgrb2a.0p50.f003")


def test_find_labelled_returns_byte_range_of_the_accumulation() -> None:
    assert _find_labelled(IDX_F015, "APCP", "surface", "12-15 hour acc fcst") == (8714653, 9154494)
    with pytest.raises(KeyError):
        _find_labelled(IDX_F015, "APCP", "surface", "9-15 hour acc fcst")


def test_upsample_keeps_original_nodes_and_averages_between() -> None:
    rng = np.random.default_rng(0)
    coarse = rng.normal(size=(361, 720))
    fine = upsample_half_degree(coarse)
    assert fine.shape == (721, 1440)
    np.testing.assert_allclose(fine[::2, ::2], coarse)
    np.testing.assert_allclose(fine[1, 0], 0.5 * (coarse[0, 0] + coarse[1, 0]))
    np.testing.assert_allclose(fine[0, 1439], 0.5 * (coarse[0, 719] + coarse[0, 0]))  # wraps at 360°


def test_upsample_rejects_other_grids() -> None:
    with pytest.raises(ValueError):
        upsample_half_degree(np.zeros((721, 1440)))


class FakeFetch:
    """Uniform 1 mm per 3 h, stored in the operational bucket layout: (6k, 6k+3) and (6k, 6k+6)."""

    def __init__(self) -> None:
        self.labels = []

    def message(self, init, product, hour, variable, level, label=None):
        self.labels.append(label)
        start, end = (int(x) for x in label.split(" ")[0].split("-"))
        return np.full((721, 1440), float(end - start) / 3.0), 0.01


def test_operational_rain_day_totals_from_bucketed_accumulations() -> None:
    src = GEFSOperational(load_settings().domain)
    src.fetch = FakeFetch()
    totals = src._one(date(2025, 8, 1), [1, 2])
    np.testing.assert_allclose(totals, 8.0)  # 24 h at 1 mm / 3 h
    assert "0-3 hour acc fcst" in src.fetch.labels and "48-51 hour acc fcst" in src.fetch.labels


def test_operational_refuses_dates_before_gefs_v12() -> None:
    with pytest.raises(ValueError, match="starts"):
        GEFSOperational(load_settings().domain)._one(date(2019, 8, 1), [1])
