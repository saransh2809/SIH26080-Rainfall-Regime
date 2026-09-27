# Methodology — SIH 26080

Living document; each section states what was done and the evidence behind it.
This system **post-processes** an existing NWP forecast. It does not forecast weather from scratch.

## 1. Data pairing (Phase 3)

| Step | Method | Evidence |
|---|---|---|
| Forecast | NOAA GEFSv12 reforecast, control member `c00`, 00 UTC runs, JJAS 2010–2019, lead days 1–3 | DATA_SOURCES §2.1 |
| Observation | IMD 0.25° gridded daily rainfall | DATA_SOURCES §3.1 |
| Rain-day window | Lead N = forecast hours (24N−21, 24N+3], matching IMD's 24 h ending 03 UTC | IMD definition |
| Date label | `valid_date = init_date + N` (IMD labels a day by its window END) | Empirical: best spatial match at +1 day in 30/40 cases |
| Accumulations | GEFS stores 6-h buckets; 3-h increments are rebuilt as `(s→s+6) − (s→s+3)` | Verified against an independent calculation on 3 training-year runs (max difference 0.1 mm = packing step) |
| Packing noise | Differences may go negative by up to the two messages' quantisation steps `2^E/10^D`; within that tolerance they are clipped to 0, beyond it the run stops | Measured −0.10 and −0.20 mm on real files |
| Grid | GEFS and IMD 0.25° nodes coincide; no regridding | Coordinates compared exactly |
| Domain mask | Only the 4,964 IMD land cells are kept; forecasts outside India are dropped after features are computed | |
| Provenance | Every dataset carries `data_kind` (`real`/`synthetic`); combining different kinds raises an error | `rainpp.data.schema` |

### Raw forecast characteristics (training years 2010–2015 only)

| | Observed | GEFS lead 1 | lead 2 | lead 3 |
|---|---|---|---|---|
| Cell-days ≥ 2.5 mm | 37.4% | 51.6% | 48.9% | 48.8% |
| Cell-days ≥ 64.5 mm (heavy) | 1.45% | 0.97% | 0.59% | 0.48% |
| Cell-days ≥ 115.6 mm (very heavy) | 0.30% | 0.16% | 0.10% | 0.07% |
| Mean (mm/day) | 7.0 | 8.6 | 7.3 | 7.0 |
| Pearson r | — | 0.44 | 0.40 | 0.35 |

GEFS rains on too many days and produces too few heavy-rain days, so an additive or multiplicative
mean correction cannot fix it. The extra Day-1 rainfall is **model spin-up**: land-mean 3-hourly rain is
about 2 mm in forecast hours 3–12, falling to about 1–1.4 mm later (checked on three training-year runs).
Both correctors are therefore lead-specific.

## 2. Features (forecast-time only)

| Feature | Why |
|---|---|
| `nwp_precip_mm`, `nwp_log1p` | The raw forecast being corrected |
| 3×3 and 7×7 neighbourhood mean / max of forecast rain | NWP rain is often displaced by tens of km; nearby forecast rain is informative |
| `obs_climatology_mm` | Mean observed rain for that cell and day of year, ±15-day smoothing, **fitted on training years only** |
| `doy_sin`, `doy_cos` | Seasonal cycle within the monsoon |
| `lat`, `lon` | Location-dependent model errors |
| `lead_day` | Error grows with lead time; spin-up affects lead 1 |

Observations of the valid day are never features.

## 3. Experiment design

- Chronological split: train 2010–2015, validation 2016–2017, test 2018–2019.
- The test years are read only for the final evaluation. Phase 3 inspected the 2018 IMD file and 40
  GEFS 2018 Day-1 forecasts for the date-convention question; no model or parameter was tuned on them.

## 4. Baselines (Phase 4)

| Tag | Model | Fitting protocol |
|---|---|---|
| A | Raw GEFS | none |
| B1 | Empirical quantile mapping per cell and lead | Fitted on 2010–2015. Mid-rank CDF handles tied zeros; above the training maximum the excess is added to the largest observation. Non-negative by construction. |
| B2 | Global LightGBM, Tweedie loss, regime-blind | Number of rounds chosen by early stopping on 2015, then refit on 2010–2015. Validation years are not used for fitting or tuning. |

Full scores: `reports/phase4_validation.md` (validation 2016–2017, real data, reproducible across runs).

### Findings (validation years; no confidence intervals yet, so small differences are not claimed as real)

| Lead 1 | A raw | B1 QM | B2 LightGBM |
|---|---|---|---|
| RMSE (mm) | 15.68 | 16.82 | **13.29** |
| Correlation | 0.47 | 0.47 | **0.56** |
| Frequency bias ≥ 2.5 mm | 1.39 | **1.00** | 1.72 |
| ETS ≥ 64.5 mm | 0.103 | **0.134** | 0.045 |
| POD ≥ 64.5 mm | 0.168 | **0.252** | 0.048 |
| FSS ≥ 64.5 mm, 9×9 | 0.529 | **0.572** | 0.132 |

1. **No baseline wins on every metric.** The metric families disagree, so all of them must be reported.
2. **B2 minimises squared error but erases heavy rain.** It predicts the conditional mean, which smooths
   extremes: it almost never forecasts ≥ 115.6 mm and inflates light-rain frequency. Lower RMSE here does
   not mean a more useful heavy-rain forecast.
3. **B1 fixes the rain-frequency bias and improves heavy-rain detection** (ETS ≥ 64.5 mm rises by 30–95%
   across leads) but raises RMSE, because matching the observed spread incurs the double penalty.
4. **Spatial context dominates B2**: the 7×7 neighbourhood mean of forecast rain carries 60% of the split
   gain and the grid-point value about 0.1%, consistent with displacement errors.

Consequences for the design: the amount forecast and the heavy-rain probability must be separate
products (Phase 7); regime-aware models (Phase 6) are compared against B1 on event and spatial scores and
against B2 on continuous scores; block-bootstrap confidence intervals are needed before any improvement
is claimed (Phase 8).
