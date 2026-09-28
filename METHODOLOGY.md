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

Consequences for the design (continued below): the amount forecast and the heavy-rain probability must be separate
products (Phase 7); regime-aware models (Phase 6) are compared against B1 on event and spatial scores and
against B2 on continuous scores; block-bootstrap confidence intervals are needed before any improvement
is claimed (Phase 8).

## 5. Weather regimes (Phase 5)

### 5.1 Labels (observation-derived; training targets and scoring only)

| Regime | Rule | Evidence |
|---|---|---|
| ACTIVE / BREAK | cos(lat)-weighted IMD rainfall over the core monsoon zone (18–28°N, 65–88°E, IITM operational description), standardised by day-of-year mean and SD of 1981–2010 (±15-day smoothing); > +1 / < −1 for ≥ 3 consecutive days (Rajeevan et al. 2010) | **published** in July–August; **derived** in June and September |
| MONSOON_DEPRESSION | IMD RSMC best-track fix of depression grade or stronger inside 12–30°N, 68–92°E during the rain day | **heuristic** (official track, our box) |
| NORMAL | none of the above | — |

Priority: MONSOON_DEPRESSION > ACTIVE > BREAK > NORMAL. Base-period July–August anomalies have mean 0.03 and SD 1.00.

Training years (2010–2015, 732 rain days): NORMAL 72.5%, BREAK 10.7%, ACTIVE 9.7%, MONSOON_DEPRESSION 7.1%.
Counts agree with known seasons: the 2015 El Niño drought year has 33 break days; 2013 (good monsoon) has 21 active days.

Why IMD's best track and not the ERA5 low-pressure-system catalogue for depressions: in JJAS 2010–2015 the two
agree on only 22 of ~53 depression days, yet the catalogue contains *some* low-pressure system on 50 of IMD's 52
depression days. The disagreement is intensity grading, not detection, and IMD grading is the operational standard.
Monsoon lows (any intensity) are present on 68% of days, too often to form a useful regime of their own.

### 5.2 Local regimes (per cell; computed from forecast-time data, heuristic)

- **OROGRAPHIC**: terrain-forced ascent w = V₈₅₀·∇h ≥ 0.05 m/s, from the FORECAST 850 hPa wind and GEFS model terrain.
  A slope threshold (10 m/km) was tried first and rejected: GEFS 0.25° terrain smooths the Western Ghats to a
  90th-percentile slope of 7.7 m/km, so only 5% of the Ghats qualified. The forced-ascent rule gives ~31% of the Ghats
  box at the July peak (the rest is lee slope) and ~0% on the Deccan interior; its seasonal cycle (100 → 174 → 73 → 47
  cells from June to September 2011) follows the monsoon westerlies. The 0.05 m/s value was first chosen on one training
  date; the sensitivity analysis below (training years 2010–2015, lead 1, 3.6 M cell-days) supports it:

  | Threshold (m/s) | Share of cell-days | Mean obs. rain in / out (mm) | Heavy-rain frequency in / out | Share of all heavy events |
  |---|---|---|---|---|
  | 0.02 | 10.1% | 14.9 / 6.1 | 4.2× | 32% |
  | 0.03 | 6.0% | 18.2 / 6.3 | 5.3× | 25% |
  | **0.05** | **2.7%** | **24.3 / 6.5** | **7.4×** | **17%** |
  | 0.08 | 1.0% | 33.4 / 6.7 | 11.2× | 10% |
  | 0.12 | 0.2% | 50.7 / 6.9 | 19.9× | 4% |

  Heavy-rain frequency rises monotonically with forecast forced ascent, so the class is physically meaningful; 0.05 m/s
  balances contrast against coverage. Using forced ascent as a *continuous* predictor is a candidate improvement.
- **COASTAL**: ≤ 100 km from the GEFS coastline. **INLAND**: otherwise.
- Limitation: over terrain above ~1.5 km the 850 hPa wind is an extrapolation below ground.

### 5.3 Classifier inputs (forecast-time only)

GEFS mid-rain-day fields at 1° over 0–40°N, 40–110°E (850 hPa u, v and relative vorticity computed on the native 0.25°
grid, MSLP, PWAT) summarised over boxes in `config/regimes.yaml`; forecast core-zone and west-coast rainfall; forecast
core anomaly standardised with the model's own training statistics per lead and month; observed core anomaly on the two
days before initialisation (already published by IMD at issue time). Class means on training years separate as expected
physically (e.g. break days: weak central-India westerlies 3.3 m/s and a high trough pressure 1002 hPa; depression days:
the largest vorticity maximum and lowest trough pressure).

The rule baseline R0 applies the ±1 threshold to the FORECAST anomaly day by day: the 3-day persistence of the published
definition cannot be verified inside a 3-day forecast whose first rain day is unfinished at issue time.

### 5.4 Classifier results (validation 2016–2017, real data; `reports/phase5_validation.md`)

| | R0 rule | RandomForest | LightGBM | CNN | always NORMAL |
|---|---|---|---|---|---|
| Macro F1 | **0.535** | 0.491 | 0.522 | 0.466 | 0.207 |
| Balanced accuracy | 0.523 | 0.454 | 0.563 | **0.587** | 0.250 |
| Depression F1 | **0.417** | 0.220 | 0.297 | 0.365 | 0 |
| Macro F1, lead 1 / 2 / 3 | 0.61 / 0.53 / 0.45 | 0.57 / 0.46 / 0.43 | 0.56 / 0.54 / 0.46 | 0.53 / 0.46 / 0.41 | — |

- Every classifier carries real regime information (macro F1 ≈ 0.47–0.54 vs 0.21 for always-NORMAL).
- **No ML model beats the interpretable rule baseline** on macro F1 or depression F1. With 2,160 highly
  correlated training samples, physically motivated thresholds are as good as learned models.
- **The CNN is not justified** on this data: lowest macro F1; it over-predicts ACTIVE (recall 0.86, precision 0.35).
  It is kept as a reported experiment, not used downstream.
- Differences between models are within the likely sampling uncertainty (≈240 correlated validation days); no
  ranking is claimed among R0, RF and LightGBM.
- LightGBM supplies the regime probabilities for Phase 6–7: the rule gives only hard labels, LightGBM is the most
  stable at leads 2–3, and its macro F1 is close to the rule's. This choice was made on validation scores, so test
  years provide the unbiased estimate.

## 6. Regime-aware correction (Phase 6)

- Training rows receive **out-of-fold** regime probabilities (classifier refit without that year), so the corrector
  learns from regime information as imperfect as it will be in operation. `C1_oracle` (observed regimes) is reported
  only as a diagnostic upper bound.
- Ablation `B2s` (B2 + elevation + distance to coast, no regime) separates the value of regime information from the
  value of static terrain features, which the local regime partly encodes.
- Claims of improvement require a paired block-bootstrap 95% interval (5-day blocks of valid dates, all cells and
  leads of a day kept together) that excludes zero.

### 6.1 Results (validation 2016–2017, real data; `reports/phase6_validation.md`)

| Lead 1 | Raw | QM (B1) | B2 | B2s | **C1** | **C2** | C1-oracle* |
|---|---|---|---|---|---|---|---|
| RMSE (mm) | 15.68 | 16.82 | 13.29 | 13.28 | 13.25 | 13.31 | 13.22 |
| Correlation | 0.471 | 0.465 | 0.556 | 0.558 | 0.560 | 0.554 | 0.562 |
| ETS ≥ 15.6 mm | 0.206 | 0.205 | 0.224 | 0.227 | 0.228 | 0.224 | 0.229 |
| ETS ≥ 64.5 mm | 0.103 | **0.134** | 0.045 | 0.044 | 0.044 | 0.064 | 0.048 |
| FSS ≥ 64.5 mm, 9×9 | 0.529 | **0.572** | 0.132 | 0.130 | 0.124 | 0.225 | 0.144 |

*Oracle uses observed regimes — not an operational forecast.

Paired 5-day block bootstrap, all leads pooled (positive = second model better):

| Comparison | RMSE improvement (95% CI) | ETS ≥ 64.5 improvement (95% CI) |
|---|---|---|
| C1 vs B2s | **+0.059 mm [+0.033, +0.088] — significant** | −0.001 [−0.005, +0.002] — not significant |
| C2 vs B2s | −0.023 [−0.079, +0.036] — not significant | **+0.007 [+0.002, +0.013] — significant** |
| B2s vs B2 (terrain only) | +0.012 [+0.000, +0.021] — significant | +0.001 [−0.002, +0.003] — not significant |

Findings:
1. **Regime information helps, but only slightly, within the ML-corrector family.** C1 lowers RMSE significantly
   but by 0.06 mm (≈ 0.4%). C2 (one model per regime) significantly improves heavy-rain ETS over the regime-blind
   model and nearly doubles FSS ≥ 64.5 mm at lead 1 (0.130 → 0.225).
2. **The ceiling is low even with perfect regime knowledge**: C1-oracle is only marginally better than C1, so better
   regime *classification* would not change the picture much for mean-regression correctors.
3. **Where regimes matter most**: in OROGRAPHIC cells, heavy-rain ETS rises from 0.083 (raw) to 0.128 (B2s),
   0.137 (C1) and 0.145 (oracle) — the clearest regime signal in the data (subgroup, no interval computed).
4. **In MONSOON_DEPRESSION regimes raw GEFS has the best heavy-rain skill** (ETS 0.129 vs ≤ 0.044 for every ML
   corrector): the correctors wash out correctly placed extremes. Reported as found.
5. **No ML corrector fixes the heavy-rain problem found in Phase 4**; quantile mapping remains best for heavy rain.
   The heavy-rain product must come from the probability model (Phase 7) or distribution-preserving correction.

### 6.2 Regime-conditioned quantile mapping (Phase 6b; `reports/phase6b_validation.md`)

Tested because quantile mapping is the only family here that preserves heavy rain. C3 fits one mapping per
predicted synoptic regime, per cell and lead; compared with global QM (B1) on identical rows.

| C3 vs B1, all leads (positive = C3 better) | Improvement | 95% CI |
|---|---|---|
| ETS ≥ 64.5 mm | −0.0039 | [−0.0061, −0.0018] — significantly worse |
| ETS ≥ 115.6 mm | −0.0038 | [−0.0076, −0.0014] — significantly worse |
| RMSE | −0.51 mm | [−0.62, −0.36] — significantly worse |

The oracle version (observed regimes) is no better, so the cause is not classifier error but **sample size**:
splitting six seasons by regime leaves ~50–70 training days per regime per cell, the tails become noisy, and heavy
rain is over-forecast (frequency bias ≥ 64.5 mm rises from 1.03 to 1.12–1.18). **Regime-conditioned QM is rejected
in this configuration.** The GEFSv12 reforecast starts in 2000, so extending training to 16 seasons (≈ 2.7× the data)
is the first thing to try before revisiting regime-split models.

## 6.3 Heavy-rain probability (Phase 7; `reports/phase7_validation.md`)

LightGBM binary classifiers, unweighted log-loss, rounds chosen on 2015 and refit on 2010–2015. Reference: climatological
event frequency (training years). Only the GEFS control member was downloaded, so raw and QM forecasts can only be
scored as 0/1 exceedances.

| ≥ 64.5 mm (51,423 events; base rate 1.42%) | CLIM | RAW 0/1 | QM 0/1 | P_blind | P_regime |
|---|---|---|---|---|---|
| Brier skill vs climatology | 0 | −0.34 | −0.71 | 0.062 | **0.067** |
| ROC AUC | 0.765 | 0.549 | 0.596 | 0.855 | **0.858** |
| BSS lead 1 / 2 / 3 | — | — | — | 0.092 / 0.056 / 0.037 | 0.095 / 0.061 / 0.044 |

- The probability models discriminate heavy-rain cells far better than the raw forecast (AUC 0.86 vs 0.55) and beat
  climatology on Brier score at every lead.
- Deterministic yes/no exceedances of rare events score worse than climatology — the case for issuing heavy rain as
  a probability.
- **Regime information significantly improves the heavy-rain Brier score** (P_regime vs P_blind: +0.000065,
  95% CI [+0.000004, +0.000121]); small in absolute terms, as expected for a 1.4% base rate.
- Reliability ≥ 64.5 mm is good up to 0.5 (e.g. forecast 0.243 → observed 0.226), slightly over-confident at 0.5–0.6.
- **≥ 115.6 mm probabilities were over-confident** (forecast 0.43 → observed 0.20) despite AUC 0.87, and skill is ~0 by
  lead 3; the regime difference is not significant.

**Recalibration.** Isotonic regression fitted on leave-one-year-out (out-of-fold) predictions over the six training
years, applied to the final model; validation years untouched by fitting.

| | Brier skill | ROC AUC | Reliability example |
|---|---|---|---|
| Heavy, uncalibrated (**used**) | 0.0666 | 0.858 | 0.34 → 0.33 |
| Heavy, calibrated | 0.0664 (difference n.s.) | 0.848 | — |
| Very heavy, uncalibrated | 0.0133 | 0.873 | 0.43 → 0.19 (over-confident) |
| Very heavy, calibrated (**used**) | 0.0227 (significantly better) | 0.782 | 0.24 → 0.19 |

Heavy probabilities were already reliable, so calibration only cost discrimination (isotonic steps create ties) and is
not applied. Very-heavy probabilities are shown calibrated: a probability on a warning map must mean what it says, and
the price — lower ranking skill — is reported, not hidden. Setting: `config/models.yaml → heavy_rain.calibrate`.

### 6.4 Overall answer to the problem statement's hypothesis (validation years)

- **Post-processing itself delivers the large gains**: RMSE −15% (raw 15.7 → 13.2 mm at lead 1); heavy-rain
  discrimination AUC 0.55 → 0.86; rain-frequency bias fixed by quantile mapping.
- **Regime-awareness adds statistically significant but small gains** where the method family can use it (continuous
  RMSE via C1, heavy-rain ETS via C2, heavy-rain Brier via P_regime), is clearest in orographic cells, and **hurts**
  when it fragments scarce data (C3). With six training seasons, data volume — not regime classification — is the
  limiting factor.

## 7. District product

Area-weighted mean over the 0.25° IMD land cells overlapping each Census-2011 district (exact polygon overlap in an
equal-area projection, EPSG:6933); the maximum over overlapping cells for heavy-rain probability. 608 of 641 districts are
≥ 95% covered; the island districts (Andaman & Nicobar, Lakshadweep) have no IMD cells and return no value; Chennai is
4.5% covered and flagged.

## 8. Forecast products and dashboard

- One product per 00 UTC initialisation (`data/products/`), built only from saved artifacts: grids (raw, C1 corrected,
  quantile-mapped, heavy/very-heavy probability, local regime), the district table, regime probabilities, and an
  explanation assembled from the corrector's own SHAP contributions and validation scores — no hand-written text.
- Observed IMD rainfall is stored as a separate, labelled variable for verification and is never an input.
- `scripts/build_products.py` refuses test-year dates without `--final-test`. Products so far: 24–31 Aug 2017
  (validation). The Aug 2018 Kerala case will be produced together with the one-time test evaluation.
- Map colours: IMD rainfall categories on a single-hue blue ramp and rare-event probability bins on a single-hue orange
  ramp, each validated for light and dark surfaces (ordinal checks: monotone lightness, visible steps, contrast). The
  lowest class is unfilled so colour appears only where something happens.

Validation case (not test): Mumbai, rain day ending 08:30 IST 30 Aug 2017, observed district mean 162 mm. Lead-1 raw 55 mm,
C1 58 mm, QM 81 mm — every amount forecast under-estimates the extreme — while P(≥ 64.5 mm) = 0.51, about 35× the
climatological rate, and the predicted regime was ACTIVE (0.54), matching the observed label.
