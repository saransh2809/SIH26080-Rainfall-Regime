from __future__ import annotations

import numpy as np
import pytest

from rainpp.data.sources.gefs_fields import (
    SynopticBox,
    block_mean,
    crop_box,
    find_message,
    mid_window_hour,
    relative_vorticity,
)

IDX = """\
5:3461212:d=2012071000:PRES:mean sea level:15 hour fcst:ENS=low-res ctl
6:4067235:d=2012071000:UGRD:850 mb:3 hour fcst:ENS=low-res ctl
34:26971451:d=2012071000:UGRD:850 mb:15 hour fcst:ENS=low-res ctl
35:27900000:d=2012071000:UGRD:700 mb:15 hour fcst:ENS=low-res ctl
"""


def test_mid_window_hours_match_rain_days() -> None:
    assert [mid_window_hour(n) for n in (1, 2, 3)] == [15, 39, 63]


def test_find_message_returns_byte_range() -> None:
    assert find_message(IDX, "UGRD", "850 mb", 15) == (26971451, 27900000)
    assert find_message(IDX, "UGRD", "700 mb", 15) == (27900000, None)
    with pytest.raises(KeyError):
        find_message(IDX, "UGRD", "850 mb", 39)


def test_vorticity_of_solid_body_rotation() -> None:
    """u = -Ω·y, v = Ω·x gives ζ = 2Ω; checked on a small near-equatorial patch."""
    lats = np.linspace(-1.0, 1.0, 41)
    lons = np.linspace(0.0, 2.0, 41)
    y = np.deg2rad(lats)[:, None] * 6.371e6
    x = np.deg2rad(lons)[None, :] * 6.371e6 * np.cos(np.deg2rad(lats))[:, None]
    omega = 1e-5
    u, v = -omega * y * np.ones_like(x), omega * x
    zeta = relative_vorticity(u, v, lats, lons[1] - lons[0])
    np.testing.assert_allclose(zeta[5:-5, 5:-5], 2 * omega, rtol=0.02)


def test_crop_tiles_into_one_degree_blocks() -> None:
    sub, lats, lons = crop_box(np.zeros((721, 1440)), SynopticBox())
    assert sub.shape == (160, 280)
    assert lats[0] == 0.0 and lats[-1] == 39.75 and lons[0] == 40.0
    assert block_mean(sub).shape == (40, 70)


def test_block_mean_rejects_ragged_field() -> None:
    with pytest.raises(ValueError):
        block_mean(np.zeros((5, 8)))
