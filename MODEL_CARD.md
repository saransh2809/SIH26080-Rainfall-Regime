# Model Card — RegimeCast rainfall post-processing (SIH 26080)

## What it is

A set of models that **post-process** NWP rainfall forecasts for India. Given a raw forecast from NOAA GEFSv12
(control member, 00 UTC, lead days 1–3, June–September), it outputs, for every 0.25° IMD land cell and every Census-2011
district:

| Output | Model | Files |
|---|---|---|
| Synoptic regime probabilities (NORMAL / ACTIVE / BREAK / MONSOON_DEPRESSION) | LightGBM multiclass (`lgbm.txt`; `lgbm_live.txt` for live runs) | `models/regime_classifier/` |
| Local regime (OROGRAPHIC / COASTAL / INLAND) | Rules on forecast 850 hPa wind, GEFS terrain and coastline | `config/regimes.yaml` |
| Corrected rainfall amount (mm) | **C1**: LightGBM Tweedie regressor with regime probabilities and local regime as inputs | `models/regime_bc/C1/` |
| Heavy-rain-frequency-preserving amount (mm) | **B1**: quantile mapping per cell and lead | `models/quantile_mapping/` |
| P(rain ≥ 64.5 mm), P(rain ≥ 115.6 mm) | LightGBM binary classifiers with regime inputs, uncalibrated | `models/heavy_rain/P_regime_*` |

It is **not a weather model**, does not use any observation of the forecast day, and is not an official IMD warning.

## Training data

- Forecasts: NOAA GEFSv12 reforecast, JJAS **2000–2015** (16 seasons; 29.1 M forecast–observation cell-days).
- Targets: IMD 0.25° gridded daily rainfall (Pai et al. 2014), rain day ending 03 UTC.
- Regime labels: IMD core-zone rainfall anomaly (Rajeevan et al. 2010 criterion) and IMD RSMC best track.
- Model choices: validation years 2016–2017 only. Held-out: test 2018–2019, operational forecasts JJAS 2021–2025.
- All data are real; synthetic data exist only in unit tests and are blocked from real datasets by a provenance check.

## Inputs at forecast time

Forecast rainfall and its 3×3 / 7×7 neighbourhoods, lead day, season, location, observed climatology (training years),
terrain elevation, distance to coast, forecast 850 hPa wind / vorticity, MSLP and precipitable water, and — for the
archive classifier only — IMD core-zone rainfall observed on the two days before the forecast was issued.

## Validation performance (2016–2017, real data; details in VERIFICATION.md)

| | Raw GEFS | Product |
|---|---|---|
| RMSE, lead 1 / 2 / 3 (mm) | 15.68 / 15.76 / 16.06 | C1: 13.07 / 13.66 / 14.06 |
| Correlation, lead 1 | 0.471 | C1: 0.575 |
| ETS ≥ 64.5 mm, lead 1 | 0.103 | QM: 0.134 · C1: 0.078 |
| Heavy-rain ROC AUC (≥ 64.5 mm), all leads | 0.549 (0/1 exceedance) | P_regime: 0.883 (lead 1: 0.907) |
| Heavy-rain Brier skill vs climatology | −0.34 | P_regime: +0.077 |
| Very-heavy ROC AUC (≥ 115.6 mm) | 0.525 | P_regime: 0.886 |
| Regime classifier macro F1 | — | 0.554 (live variant 0.532; simple rule 0.542) |

Held-out results (test years and operational forecasts): `reports/final_evaluation.md` and VERIFICATION.md §4.

## Intended use

- Guidance for forecasters and disaster managers alongside the raw NWP forecast: where the model is likely to be
  biased, and where heavy rain is more likely than climatology.
- District-level screening of heavy-rain risk for the next three days during the monsoon.

## Not intended for

- Replacing IMD warnings or NCMRWF forecasts.
- Forecasts outside June–September, beyond day 3, or from a different NWP model without retraining.
- Point (station) forecasts: outputs are 0.25° cell and district averages.

## Known failure modes

- Corrected amounts under-forecast extremes; heavy rain must be read from the probability layer.
- In monsoon depressions the raw forecast's heavy-rain placement beats the corrector's.
- Very-heavy-rain probabilities have little skill by day 3.
- The regime classifier is right about two times in three; regime probabilities are guidance, not certainty.
- Live forecasts: operational GEFS differs from the reforecast in its initial conditions and supplies some fields only
  at 0.5°; see METHODOLOGY §9.

## Ethical and operational considerations

- Probabilities are shown with their validation scores and data source on every screen; unverified live products are
  labelled as such.
- Place names and boundaries follow Census-2011 / Survey of India index maps via DataMeet; new districts are missing.
- Hindi interface text awaits native-speaker review.
