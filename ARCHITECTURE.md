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
   GEFSv12 reforecast (training archive)   GEFSv12 operational (live)   IMD 0.25° gridded (observed)
   IMD best track (labels)                 GEFS terrain / land, districts (static)
        │                                        │
        ▼                                        ▼
 DATA PROCESSING   stream → crop to India → align rain day (03 UTC) → align grid → validate schema + provenance
        │          (operational 0.5° wind / pressure interpolated to 0.25°, then identical processing)
        ▼
 FEATURE ENGINEERING   forecast rainfall + neighbourhoods, lead, season, 850 hPa wind / vorticity, MSLP, PWAT,
        │              terrain, coast distance, observed climatology (training years only)
        ▼
 WEATHER REGIME CLASSIFICATION
   synoptic (per day):  ACTIVE / BREAK / DEPRESSION / NORMAL   ← LightGBM on forecast fields
                        (archive: + IMD rainfall published before the forecast; live: forecast fields only)
   local (per cell):    OROGRAPHIC / COASTAL / INLAND          ← rules on terrain + forecast wind
        │
        ▼
 RAW NWP FORECAST ──► REGIME-AWARE BIAS CORRECTION (C1) ──► corrected rainfall (mm, ≥ 0)
        │
        ▼
 HEAVY RAINFALL PROBABILITY   P(rain ≥ 64.5 mm), P(rain ≥ 115.6 mm), calibrated
        │
        ▼
 GRID FORECAST ──► area-weighted aggregation ──► DISTRICT FORECAST (641 districts) ──► CSV / print bulletin
        │
        ▼
 VERIFICATION   RMSE · MAE · Bias · ETS · CSI · POD · FAR · FSS · Brier · AUC · block bootstrap
        │
        ▼
 VISUALIZATION   FastAPI → React + Leaflet dashboard (English / Hindi)
```

## 3. Components

| Package / script | Responsibility |
|---|---|
| `rainpp.config` | Load `config/*.yaml`, env overrides, validate the chronological split |
| `rainpp.data.schema` | Canonical layout + provenance guard (`real` vs `synthetic`) |
| `rainpp.data.sources` | Adapters: `imd`, `gefs` (reforecast rainfall), `gefs_fields` (synoptic fields), `gefs_operational` (live) |
| `rainpp.data.align` / `features` | Rain-day and grid alignment; training tables and forecast-only tables |
| `rainpp.regimes` | Labels, forecast-time classifier features, classifiers (rule, RF, LightGBM, CNN), local regimes |
| `rainpp.models` | Quantile mapping, LightGBM correctors, regime-split corrector, heavy-rain model, U-Net experiment |
| `rainpp.spatial` | Grid → district area-weighted aggregation |
| `rainpp.verification` | Metrics, probabilistic scores, block bootstrap, comparison reports |
| `rainpp.products` | One code path from forecast to product; also used by the one-time evaluation |
| `rainpp.api` | FastAPI app (`app.py`) and single-origin server for deployment (`serve.py`) |
| `frontend/` | React + Vite + TypeScript + react-leaflet dashboard |
| `scripts/run_pipeline.py` | Downloads → tables → baselines → regimes → classifier → correctors → heavy rain |
| `scripts/run_live.py` | Build today's product from the newest operational GEFS run |
| `scripts/run_final_evaluation.py` | One-time scoring of the frozen models on test and operational years |

## 4. Data contracts

All adapters return xarray Datasets in one canonical layout (`rainpp/data/schema.py`):

| Kind | Dims of `precip_mm` | Required attrs |
|---|---|---|
| Forecast | `(init_time, lead_day, member, lat, lon)` | `data_kind`, `source` |
| Observation | `(time, lat, lon)` | `data_kind`, `source` |

`ForecastSource.load()` / `ObservationSource.load()` validate every result: dims, ascending coords,
non-negative rainfall, not all-missing, and that the declared `data_kind` matches. Any step combining
datasets calls `assert_same_kind`, so synthetic data cannot enter real training or evaluation.

Cell table, one row per (init, lead day, grid cell) on IMD land cells: forecast features, observed climatology
(training years), `valid_date`, and `obs_precip_mm` (target in training; verification only in products). Regime
columns (probabilities, local regime, terrain) are added by `rainpp.regimes.augment.add_regime_columns`.

## 5. Model ladder (the core comparison)

| Tag | Model | Purpose |
|---|---|---|
| A | Raw NWP | Reference |
| B1 | Quantile mapping per cell/lead | Standard interpretable correction |
| B2 | Global LightGBM (Tweedie loss, no regime input) | Strong regime-blind ML |
| C1 | LightGBM with regime probabilities and local regime as inputs | Regime-aware, pooled data (product corrector) |
| C2 | One LightGBM per predicted regime | Regime-aware, separate models (PS wording) |
| C3 | Quantile mapping per regime | Regime-aware, distribution-preserving (rejected on validation) |
| D | U-Net on whole forecast maps | Deep-learning comparison against the tree correctors |
| P | LightGBM heavy-rain probability with regime inputs | Warnings for rare extremes |

The claim "regime-aware correction helps" is only made where C beats B with block-bootstrap confidence intervals.

## 6. Experiment design

- Forecast archive: GEFSv12 reforecast, control member, 00 UTC, lead days 1–3, JJAS 2000–2019.
- Chronological split (`config/settings.yaml`): train 2000–2015, validation 2016–2017, test 2018–2019.
- Operational test: GEFSv12 operational forecasts, JJAS 2021–2025, run through the live pipeline.
- All model choices use validation years only. Test and operational years are scored once, by
  `scripts/run_final_evaluation.py`, which records the hashes of the evaluated models and refuses silent re-runs.

## 7. Modes and product sources

| Mode | Data | Label shown in UI |
|---|---|---|
| `demo` | Saved products of **real** data, served offline | "Offline demonstration · real data" |
| `real` | Full pipeline output from downloaded data | "Real data" |
| tests | Tiny synthetic fixtures only | never shown in UI |

Each product also carries its source: **archived** (reforecast, verifiable against IMD) or **live** (operational
run; verification pending until IMD publishes the days). Live products valid outside June–September are refused
unless explicitly requested, and then carry a warning.

## 8. API

`GET /health` · `/metadata` · `/model-info` · `/regimes` · `/districts` · `/geo/{states|districts}` ·
`/verification/{phase}` · `/products` · `/forecast/{init}` · `/forecast/{init}/districts?lead=` ·
`/forecast/{init}/districts.csv?lead=` · `/forecast/{init}/grid?var=&lead=`

Inputs are validated (dates, whitelisted variables and layers, lead range); files are resolved from fixed
directories plus validated identifiers, never from client paths. In deployment `rainpp.api.serve` mounts the API
at `/api` and the built dashboard at `/` (same paths as the Vite dev proxy).

## 9. Deployment

`docker compose up --build` builds the dashboard (Node) and the API image (Python 3.12) and serves both on
http://localhost:8080. `data/`, `models/` and `reports/` are mounted read-only from the host; the image contains no
data. Live products are built on the host with `scripts/run_live.py` (e.g. a daily scheduled task).

## 10. Verification items

| # | Item | Status |
|---|---|---|
| 1 | IMD date convention | ✅ Resolved: labelled by window END date (empirical, DATA_SOURCES §7) |
| 2 | GEFSv12 layout, APCP accumulation, grid offsets | ✅ Resolved: 6-h buckets, exact grid match (DATA_SOURCES §2.1) |
| 3 | Download size/time | ✅ Resolved: ≈2 s and ≈10 MB per init for Days 1–3 APCP |
| 4 | Core-monsoon-zone definition | ✅ Resolved: IITM operational description 18–28°N, 65–88°E over IMD land cells (config/regimes.yaml) |
| 5 | Survey-of-India-based district boundaries | ✅ Resolved with caveats: DataMeet SoI index maps, Census-2011 districts (DATA_SOURCES §5) |
| 6 | Operational GEFS layout and variables | ✅ Resolved: per-hour files; APCP buckets as reforecast; 850 hPa wind and "PRES:mean sea level" at 0.5° only (`gefs_operational.py`) |

Test-period hygiene: the 2018 IMD file and 40 GEFS 2018 Day-1 forecasts were inspected in Phase 3 to establish the
date convention (a data-format question). No model or parameter was tuned on them. All further sanity checks
use training years only.
