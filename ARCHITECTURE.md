# Architecture — Regime-Aware Rainfall Post-Processing (SIH 26080)

This is a **post-processing** system. It does not forecast weather from scratch: it takes an existing
NWP rainfall forecast and corrects it using what it has learned about that model's past errors
in each weather regime.

## 1. Judge-facing pipeline

```
DATA → IDENTIFY REGIME → CORRECT NWP → PREDICT HEAVY RAIN → VERIFY → VISUALIZE
```

## 2. Full pipeline

```
 PUBLIC DATA
   GEFSv12 reforecast (forecast)   IMD 0.25° gridded (observed)   IMD tracks / ERA5 LPS (labels)   DEM, districts (static)
        │                                   │                               │
        ▼                                   ▼                               ▼
 DATA PROCESSING   stream → crop to India → align rain day (03 UTC) → align grid → validate schema + provenance
        │
        ▼
 FEATURE ENGINEERING   forecast rainfall, lead, winds, humidity, MSLP, terrain, coast distance, season
        │
        ▼
 WEATHER REGIME CLASSIFICATION
   synoptic (per day):  ACTIVE / BREAK / DEPRESSION / NORMAL   ← ML on forecast fields
   local (per cell):    OROGRAPHIC / COASTAL / INLAND          ← rules on terrain + forecast wind
        │
        ▼
 RAW NWP FORECAST ──► REGIME-AWARE BIAS CORRECTION ──► corrected rainfall (mm, ≥ 0)
        │
        ▼
 HEAVY RAINFALL PROBABILITY   P(rain ≥ 64.5 mm), P(rain ≥ 115.6 mm), calibrated
        │
        ▼
 GRID FORECAST ──► area-weighted aggregation ──► DISTRICT FORECAST
        │
        ▼
 VERIFICATION   RMSE · MAE · Bias · ETS · CSI · POD · FAR · FSS   (Raw vs Global vs Regime-aware)
        │
        ▼
 VISUALIZATION   FastAPI → React + Leaflet dashboard
```

## 3. Components

| Package | Responsibility | Phase |
|---|---|---|
| `rainpp.config` | Load `config/*.yaml`, env overrides, validate chronological split | 2 ✅ |
| `rainpp.data.schema` | Canonical layout + provenance guard (`real` vs `synthetic`) | 2 ✅ |
| `rainpp.data.sources` | Adapters: `base` ✅, `imd`, `gefs`, `gfs`, `ncmrwf` (stub), `demo` | 2–3 |
| `rainpp.data.align` / `features` | Rain-day and grid alignment; feature engineering | 3 |
| `rainpp.regimes` | Label generation (`labels.py`) and classifier (`classifier.py`) | 5 |
| `rainpp.models` | Quantile mapping, global corrector, regime corrector, heavy-rain model, model registry | 4, 6, 7 |
| `rainpp.spatial` | Grid → district area-weighted aggregation | 6 |
| `rainpp.verification` | Metric functions and comparison report | 8 |
| `rainpp.api` | FastAPI app | 9 |
| `frontend/` | React + Vite + TypeScript + react-leaflet dashboard | 10 |

## 4. Data contracts

All adapters return xarray Datasets in one canonical layout (`rainpp/data/schema.py`):

| Kind | Dims of `precip_mm` | Required attrs |
|---|---|---|
| Forecast | `(init_time, lead_day, member, lat, lon)` | `data_kind`, `source` |
| Observation | `(time, lat, lon)` | `data_kind`, `source` |

`ForecastSource.load()` / `ObservationSource.load()` validate every result: dims, ascending coords,
non-negative rainfall, not all-missing, and that the declared `data_kind` matches. Any step combining
datasets calls `assert_same_kind`, so synthetic data cannot enter real training or evaluation.

Training table (Phase 3), one row per (valid date, lead day, grid cell):

```
valid_date, init_date, lead_day, lat, lon, district_id,
nwp_precip_mm, nwp_<weather fields>..., elevation_m, coast_distance_km, month, doy,
synoptic_regime, local_regime, label_evidence,   ← labels (training only)
obs_precip_mm                                    ← target
```

## 5. Model ladder (the core comparison)

| Tag | Model | Purpose |
|---|---|---|
| A | Raw NWP | Reference |
| B1 | Quantile mapping per cell/lead | Standard interpretable correction |
| B2 | Global LightGBM (Tweedie loss, no regime input) | Strong regime-blind ML |
| C1 | LightGBM with regime as input | Regime-aware, pooled data |
| C2 | One LightGBM per regime | Regime-aware, separate models (PS wording) |
| CNN | Regime classifier experiment | Deep-learning comparison against the tree classifier |

The claim "regime-aware correction helps" is only made if C beats B on the held-out test set,
with block-bootstrap confidence intervals.

## 6. Experiment design

- Screening prototype: JJAS 2010–2019, GEFSv12 control member, lead days 1–3.
- Chronological split: train 2010–2015, validation 2016–2017, test 2018–2019 (`config/settings.yaml`).
- Test years are used once, for final numbers. The Aug 2018 Kerala floods fall in the test period.

## 7. Modes

| Mode | Data | Label shown in UI |
|---|---|---|
| `demo` | Small bundled slice of **real** processed data (a test-period case), for offline judging | "OFFLINE DEMONSTRATION — archived real data" |
| `real` | Full pipeline output from downloaded data | "REAL DATA" |
| tests | Tiny synthetic fixtures only | never shown in UI |

Switch via `mode` in `config/settings.yaml` or `RAINPP_MODE`. The core application code is the same.

## 8. Planned API

`GET /health` · `GET /metadata` · `GET /model-info` · `GET /districts` · `GET /regimes` ·
`GET /forecast/{date}` · `POST /classify-regime` · `POST /correct-rainfall` ·
`POST /heavy-rainfall-probability` · `POST /district-forecast` · `POST /verify`

Every prediction response carries `data_kind`, `model_version`, and the regime with its confidence.

## 9. Verification items

| # | Item | Status |
|---|---|---|
| 1 | IMD date convention | ✅ Resolved: labelled by window END date (empirical, DATA_SOURCES §7) |
| 2 | GEFSv12 layout, APCP accumulation, grid offsets | ✅ Resolved: 6-h buckets, exact grid match (DATA_SOURCES §2.1) |
| 3 | Download size/time | ✅ Resolved: ≈2 s and ≈10 MB per init for Days 1–3 APCP |
| 4 | Exact core-monsoon-zone polygon (Rajeevan et al. 2010) | Open — Phase 5 |
| 5 | Survey-of-India-compliant district boundaries | Open — Phase 6 |

Test-period hygiene: the 2018 IMD file and 40 GEFS 2018 Day-1 forecasts were inspected in Phase 3 to establish the
date convention (a data-format question). No model or parameter was tuned on them. All further sanity checks
use training years only.
