# SIH 26080 — Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

Post-processing system (NOT a weather model): learns `raw NWP forecast → observed rainfall`, conditioned on weather regime.

## Non-negotiable rules
- Never fabricate weather data, model accuracy, or verification metrics. Unevaluated metrics show "Not evaluated".
- Synthetic data is for unit tests / UI dev only. Every xarray Dataset carries `attrs["data_kind"]` = `real` | `synthetic`; `rainpp.data.schema.assert_same_kind` must guard any combination of datasets. Never mix them in training or evaluation.
- Observation/label data (IMD rainfall, ERA5, LPS tracks) must never be a model input at inference time. The classifier sees forecast fields only.
- Chronological train/val/test split (see `config/settings.yaml`). The test period is touched only for final evaluation.
- Rainfall is non-negative; any transform/clipping is documented in METHODOLOGY.md.
- Thresholds, regions, radii, periods live in `config/*.yaml`, not in code. Heuristic parameters are labelled as heuristic.
- No LLM generates rainfall values. No deep learning without a baseline comparison.
- Report honestly when a model is worse than raw NWP for a metric/regime.

## Layout
- `src/rainpp/` Python package (data, regimes, models, spatial, verification, api)
- `config/` YAML config; env var `RAINPP_DATA_DIR` / `RAINPP_MODE` override paths/mode
- `frontend/` React + Vite + TS + react-leaflet
- Docs: DATA_SOURCES.md, ARCHITECTURE.md (+ METHODOLOGY, MODEL_CARD, VERIFICATION, DEMO_GUIDE, LIMITATIONS later)

## Commands
- Env: `py -3.12 -m venv .venv` then `.venv/Scripts/python -m pip install -e ".[dev]"`
- Tests: `.venv/Scripts/python -m pytest`
