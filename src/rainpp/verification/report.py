"""Side-by-side verification of several forecasts on the same table of rows.

Every model is scored on exactly the same (valid_date, lead_day, cell) rows, so comparisons
are like for like. Undefined scores stay NaN and are rendered as "n/a" — never as 0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rainpp.config import Domain
from rainpp.verification.metrics import (
    calculate_bias,
    calculate_correlation,
    calculate_fss,
    calculate_mae,
    calculate_rmse,
    contingency,
)

OBS = "obs_precip_mm"


def to_fields(table: pd.DataFrame, column: str, domain: Domain) -> np.ndarray:
    """Rebuild 2-D maps (one per valid_date) on the full domain grid; cells without rows are NaN."""
    nlat = round((domain.lat_max - domain.lat_min) / domain.resolution_deg) + 1
    nlon = round((domain.lon_max - domain.lon_min) / domain.resolution_deg) + 1
    dates, date_idx = np.unique(table["valid_date"].to_numpy(), return_inverse=True)
    i = np.rint((table["lat"].to_numpy() - domain.lat_min) / domain.resolution_deg).astype(int)
    j = np.rint((table["lon"].to_numpy() - domain.lon_min) / domain.resolution_deg).astype(int)
    fields = np.full((dates.size, nlat, nlon), np.nan)
    fields[date_idx, i, j] = table[column].to_numpy()
    return fields


def score(table: pd.DataFrame, column: str, domain: Domain, thresholds: list[float],
          fss_thresholds: list[float], fss_sizes: list[int]) -> dict:
    f, o = table[column].to_numpy(), table[OBS].to_numpy()
    result: dict = {
        "n": len(table),
        "rmse": calculate_rmse(f, o),
        "mae": calculate_mae(f, o),
        "bias": calculate_bias(f, o),
        "correlation": calculate_correlation(f, o),
        "categorical": {str(t): contingency(f, o, t).as_dict() for t in thresholds},
        "fss": {},
    }
    if fss_thresholds:
        ff, of = to_fields(table, column, domain), to_fields(table, OBS, domain)
        for t in fss_thresholds:
            result["fss"][str(t)] = {str(n): calculate_fss(ff, of, t, n) for n in fss_sizes}
    return result


def compare(table: pd.DataFrame, models: dict[str, str], domain: Domain, thresholds: list[float],
            fss_thresholds: list[float], fss_sizes: list[int]) -> dict:
    """{lead_day: {model: scores}} for each lead, computed on identical rows."""
    missing = [c for c in [OBS, *models.values()] if c not in table.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return {
        int(lead): {name: score(group, col, domain, thresholds, fss_thresholds, fss_sizes)
                    for name, col in models.items()}
        for lead, group in table.groupby("lead_day")
    }


def _fmt(value: float) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def to_markdown(results: dict, title: str) -> str:
    lines = [f"## {title}", ""]
    for lead, by_model in results.items():
        names = list(by_model)
        lines += [f"### Lead day {lead}", "", "| Metric | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        first = by_model[names[0]]
        rows = [(m.upper() + (" (mm)" if m != "correlation" else ""), lambda s, m=m: s[m])
                for m in ("rmse", "mae", "bias", "correlation")]
        for t in first["categorical"]:
            for m in ("pod", "far", "csi", "ets", "frequency_bias"):
                rows.append((f"{m.upper()} ≥{t} mm", lambda s, t=t, m=m: s["categorical"][t][m]))
        for t, by_size in first["fss"].items():
            for n in by_size:
                rows.append((f"FSS ≥{t} mm, {n}x{n} cells", lambda s, t=t, n=n: s["fss"][t][n]))
        for label, get in rows:
            lines.append(f"| {label} | " + " | ".join(_fmt(get(by_model[n])) for n in names) + " |")
        events = {t: first["categorical"][t]["hits"] + first["categorical"][t]["misses"] for t in first["categorical"]}
        lines += ["", f"Rows: {first['n']:,}. Observed events per threshold: "
                  + ", ".join(f"≥{t} mm: {e:,}" for t, e in events.items()), ""]
    return "\n".join(lines)
