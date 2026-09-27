from __future__ import annotations

import numpy as np
import pytest

from rainpp.config import load_settings
from rainpp.data.sources.gefs import (
    crop_global_field,
    packing_step,
    parse_idx,
    rain_day_window,
    three_hour_increments,
    window_total,
)

# Verbatim lines from apcp_sfc_2018080100_c00.grib2.idx
IDX_SAMPLE = """\
1:0:d=2018080100:APCP:surface:0-3 hour acc fcst:ENS=low-res ctl
2:458682:d=2018080100:APCP:surface:0-6 hour acc fcst:ENS=low-res ctl
3:996236:d=2018080100:APCP:surface:6-9 hour acc fcst:ENS=low-res ctl
4:1431369:d=2018080100:APCP:surface:6-12 hour acc fcst:ENS=low-res ctl
"""


def test_parse_idx_reads_offsets_and_windows() -> None:
    entries = parse_idx(IDX_SAMPLE)
    assert [(e.start_hour, e.end_hour) for e in entries] == [(0, 3), (0, 6), (6, 9), (6, 12)]
    assert entries[2].offset == 996236
    assert entries[0].variable == "APCP"


def test_parse_idx_skips_instantaneous_messages() -> None:
    assert parse_idx("6:4064763:d=2018080100:UGRD:850 mb:3 hour fcst:ENS=low-res ctl") == []


@pytest.mark.parametrize(
    ("lead", "init_hour", "end_hour", "expected"),
    [(1, 0, 3, (3, 27)), (2, 0, 3, (27, 51)), (3, 0, 3, (51, 75)), (1, 0, 0, (0, 24)), (1, 12, 3, (15, 39))],
)
def test_rain_day_window(lead, init_hour, end_hour, expected) -> None:
    assert rain_day_window(lead, init_hour, end_hour) == expected


def test_rain_day_window_rejects_lead_zero() -> None:
    with pytest.raises(ValueError):
        rain_day_window(0, 0, 3)


def _bucket_accumulations(rate_per_3h: list[float]) -> dict[tuple[int, int], np.ndarray]:
    """Build GEFS-style 6-hour bucket messages from known 3-hourly amounts."""
    acc = {}
    for k in range(0, len(rate_per_3h), 2):
        start = 3 * k
        first = rate_per_3h[k]
        acc[(start, start + 3)] = np.array([first])
        if k + 1 < len(rate_per_3h):
            acc[(start, start + 6)] = np.array([first + rate_per_3h[k + 1]])
    return acc


def test_three_hour_increments_undo_buckets() -> None:
    truth = [1.0, 2.0, 0.0, 4.0, 5.0, 0.5, 3.0, 0.0, 7.0, 1.0]  # hours 0-30
    inc = three_hour_increments(_bucket_accumulations(truth), max_hour=30)
    assert [float(inc[h][0]) for h in range(3, 31, 3)] == truth


def test_day1_total_is_hours_3_to_27() -> None:
    truth = [100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]  # 0-3 and 27-30 excluded
    inc = three_hour_increments(_bucket_accumulations(truth), max_hour=30)
    assert float(window_total(inc, 3, 27)[0]) == pytest.approx(8.0)


def test_packing_noise_clipped_but_real_negatives_rejected() -> None:
    noisy = {(0, 3): np.array([2.0]), (0, 6): np.array([1.99])}
    assert float(three_hour_increments(noisy, max_hour=6)[6][0]) == 0.0
    broken = {(0, 3): np.array([2.0]), (0, 6): np.array([1.0])}
    with pytest.raises(ValueError, match="accumulation difference"):
        three_hour_increments(broken, max_hour=6)


def test_packing_step_formula() -> None:
    assert packing_step(0, 2) == pytest.approx(0.01)
    assert packing_step(1, 1) == pytest.approx(0.2)


def test_tolerance_follows_message_packing() -> None:
    acc = {(0, 3): np.array([5.0]), (0, 6): np.array([4.8])}  # -0.2 mm after differencing
    coarse = {(0, 3): 0.1, (0, 6): 0.1}
    assert float(three_hour_increments(acc, 6, coarse)[6][0]) == 0.0
    fine = {(0, 3): 0.01, (0, 6): 0.01}
    with pytest.raises(ValueError, match="packing tolerance"):
        three_hour_increments(acc, 6, fine)


def test_missing_message_raises() -> None:
    with pytest.raises(KeyError):
        three_hour_increments({(0, 3): np.array([1.0])}, max_hour=6)


def test_crop_matches_imd_grid() -> None:
    domain = load_settings().domain
    field = np.broadcast_to(np.linspace(90.0, -90.0, 721)[:, None], (721, 1440))
    sub, lats, lons = crop_global_field(field, domain)
    assert sub.shape == (129, 135)
    assert (lats[0], lats[-1], lons[0], lons[-1]) == (6.5, 38.5, 66.5, 100.0)
    assert np.all(np.diff(lats) > 0)
    np.testing.assert_allclose(sub[:, 0], lats)  # values follow their latitude after flipping
