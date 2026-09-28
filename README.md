# Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**Smart India Hackathon — Problem Statement 26080** · Ministry of Earth Sciences · NCMRWF

This system **post-processes** numerical weather prediction (NWP) rainfall forecasts. It does not forecast the
weather from scratch: it learns how a model's rainfall forecast differs from what IMD later observed, conditioned on
the weather regime, and corrects new forecasts accordingly.

## The problem

Rainfall forecast errors over India depend on the weather regime — active and break monsoon spells, monsoon
depressions, orographic rain on the Western Ghats, coastal rain. A single bias correction treats them all alike.

## The solution

```
DATA → IDENTIFY REGIME → CORRECT NWP → PREDICT HEAVY RAIN → VERIFY → VISUALIZE
```

| Step | What it does | Where |
|---|---|---|
| Data | NOAA GEFSv12 reforecast 2000–2019 (training archive) and operational runs (live), paired with IMD 0.25° gridded rainfall, aligned to IMD's 03 UTC rain day | `src/rainpp/data` |
| Regime | Synoptic regime per day (active / break / depression / normal) predicted from forecast fields; local regime per cell (orographic / coastal / inland) | `src/rainpp/regimes` |
| Correction | Quantile mapping and LightGBM correctors, regime-blind and regime-aware | `src/rainpp/models` |
| Heavy rain | Probability of ≥ 64.5 mm and ≥ 115.6 mm (IMD heavy / very heavy), calibrated | `src/rainpp/models/heavy_rain.py` |
| Verification | RMSE, MAE, bias, ETS, CSI, POD, FAR, FSS, Brier skill, ROC AUC, block-bootstrap intervals | `src/rainpp/verification` |
| Products | Grids, district table (641 districts, CSV), printable bulletin, regime probabilities, SHAP-based explanation per forecast | `src/rainpp/products.py` |
| Live mode | Newest NOAA GEFS operational run → product in ≈ 50 s, marked live / verification pending | `scripts/run_live.py` |
| Visualisation | FastAPI backend + offline React/Leaflet dashboard, English / Hindi, case studies | `src/rainpp/api`, `frontend/` |

## Results (validation 2016–2017, real data, models trained on 16 seasons 2000–2015)

- Post-processing reduces lead-1 RMSE by 17% (15.68 → 13.07 mm, regime-aware C1) and raises heavy-rain discrimination
  from AUC 0.55 (raw forecast) to 0.88 (probability model; 0.91 at lead 1).
- Regime information adds **small but statistically significant** gains (C1 RMSE, heavy-rain Brier score), is clearest
  over the Western Ghats, and **hurts** regime-split quantile mapping. Raw GEFS keeps the best heavy-rain placement in
  monsoon depressions.
- Held-out results — test years 2018–2019 and real operational forecasts JJAS 2021–2025, scored once:
  [reports/final_evaluation.md](reports/final_evaluation.md).
- Every table: [VERIFICATION.md](VERIFICATION.md). Methods: [METHODOLOGY.md](METHODOLOGY.md). Caveats:
  [LIMITATIONS.md](LIMITATIONS.md). Model card: [MODEL_CARD.md](MODEL_CARD.md).

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Pipeline, components, data contracts, API |
| [DATA_SOURCES.md](DATA_SOURCES.md) | Every dataset, how it was verified, alignment rules |
| [METHODOLOGY.md](METHODOLOGY.md) | Methods, experiments and results with evidence |
| [VERIFICATION.md](VERIFICATION.md) | Every score, generated from `reports/*.json` |
| [MODEL_CARD.md](MODEL_CARD.md) | What the models are for, their data and failure modes |
| [LIMITATIONS.md](LIMITATIONS.md) | What the results do and do not show |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | Running the offline demonstration |
| `reports/` | Machine-readable and Markdown validation reports per phase |

## Quick start (Windows, Python 3.12, Node 24)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,grib,ml,geo,api]"
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python scripts/run_pipeline.py --download
cd frontend && npm install
```

The first full run downloads ~20 monsoon seasons (a few hours on a home connection) and trains every model in about
an hour on a laptop CPU. The held-out evaluation additionally needs the operational forecasts and is run once:

```bash
.venv/Scripts/python scripts/download_data.py --source gefs_op fields_op imd --years 2021 2022 2023 2024 2025
.venv/Scripts/python scripts/run_final_evaluation.py
```

Run the dashboard (two terminals):

```bash
.venv/Scripts/python -m uvicorn rainpp.api.app:app --port 8000
```

```bash
node frontend/node_modules/vite/bin/vite.js frontend --port 5173
```

Or, after the pipeline has produced `data/`, `models/` and `reports/`, one command serves everything on
http://localhost:8080:

```bash
docker compose up --build
```

Live forecast for today (schedule daily after 11:30 IST):

```bash
.venv/Scripts/python scripts/run_live.py --latest
```

Tests: `.venv/Scripts/python -m pytest` and `cd frontend && npx vitest run`.

## Principles

No fabricated data or metrics; synthetic data only in unit tests and enforced by a provenance guard; observations
never used as model inputs; chronological train / validation / test split; honest reporting when a model is worse
than the raw forecast. See [CLAUDE.md](CLAUDE.md).

## Data credits

NOAA GEFSv12 reforecast and operational forecasts (AWS Open Data) · IMD gridded rainfall (Pai et al. 2014) · IMD RSMC New Delhi best track ·
ERA5-derived LPS catalogue (K. Hunt, CC BY 4.0) · district boundaries: DataMeet, Survey of India index maps.
