# Methodology — SIH 26080

Living document; each section states what was done and the evidence behind it. Every score quoted here is in a
JSON report under `reports/`; the full tables are in [VERIFICATION.md](VERIFICATION.md), generated from those files.
This system **post-processes** an existing NWP forecast. It does not forecast weather from scratch.

## 1. Data pairing (Phase 3)

| Step | Method | Evidence |
|---|---|---|
| Forecast | NOAA GEFSv12 reforecast, control member `c00`, 00 UTC runs, JJAS 2000–2019, lead days 1–3 | DATA_SOURCES §2.1 |
| Live forecast | NOAA GEFSv12 operational runs, same member and processing (§9) | DATA_SOURCES §2.1b |
| Observation | IMD 0.25° gridded daily rainfall | DATA_SOURCES §3.1 |
| Rain-day window | Lead N = forecast hours (24N−21, 24N+3], matching IMD's 24 h ending 03 UTC | IMD definition |
| Date label | `valid_date = init_date + N` (IMD labels a day by its window END) | Empirical: best spatial match at +1 day in 30/40 cases |
| Accumulations | GEFS stores 6-h buckets; 3-h increments are rebuilt as `(s→s+6) − (s→s+3)` | Verified against an independent calculation on 3 training-year runs (max difference 0.1 mm = packing step) |
| Packing noise | Differences may go negative by up to the two messages' quantisation steps `2^E/10^D`; within that tolerance they are clipped to 0, beyond it the run stops | Measured −0.10 and −0.20 mm on real files |
| Grid | GEFS and IMD 0.25° nodes coincide; no regridding | Coordinates compared exactly |
| Domain mask | Only the 4,964 IMD land cells are kept; forecasts outside India are dropped after features are computed | |
| Provenance | Every dataset carries `data_kind` (`real`/`synthetic`); combining different kinds raises an error | `rainpp.data.schema` |

### Raw forecast characteristics (measured on training years 2010–2015)

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

- Chronological split: **train 2000–2015 (16 seasons, 29.1 M cell-days), validation 2016–2017, test 2018–2019**.
  Training began with 2010–2015; it was extended once the 2000–2009 reforecast was downloaded, because Phase 6b showed
  data volume limiting the regime-split models. Every phase was re-run; all numbers below are from the 16-season models.
- **Operational test: JJAS 2021–2025** — real GEFSv12 operational forecasts run through the live pipeline (§9).
- All model choices (classifier, corrector, calibration) use validation years only. Test and operational-test years
  are scored once, by `scripts/run_final_evaluation.py` (§11). Phase 3 inspected the 2018 IMD file and 40 GEFS 2018
  Day-1 forecasts for the date-convention question; no model or parameter was tuned on them.
- LightGBM correctors are fitted on a checkerboard of half the cells (memory; neighbouring cells are strongly correlated).
  Quantile mapping and every evaluation use all cells.

## 4. Baselines (Phase 4)

| Tag | Model | Fitting protocol |
|---|---|---|
| A | Raw GEFS | none |
| B1 | Empirical quantile mapping per cell and lead | Fitted on the training years. Mid-rank CDF handles tied zeros; above the training maximum the excess is added to the largest observation. Non-negative by construction. |
| B2 | Global LightGBM, Tweedie loss, regime-blind | Rounds chosen by early stopping on the last training year (376), then refit on all training years. |

### Findings (validation 2016–2017; `reports/phase4_validation.md`)

| Lead 1 | A raw | B1 QM | B2 LightGBM |
|---|---|---|---|
| RMSE (mm) | 15.68 | 16.57 | **13.16** |
| Correlation | 0.471 | 0.472 | **0.567** |
| Frequency bias ≥ 2.5 mm | 1.39 | **0.98** | 1.65 |
| ETS ≥ 64.5 mm | 0.103 | **0.134** | 0.069 |
| POD ≥ 64.5 mm | 0.168 | **0.248** | 0.076 |
| FSS ≥ 64.5 mm, 9×9 | 0.529 | **0.576** | 0.229 |

1. **No baseline wins on every metric.** The metric families disagree, so all of them must be reported.
2. **B2 minimises squared error but smooths heavy rain away.** It predicts the conditional mean: its frequency bias at
   ≥ 64.5 mm is 0.14 (it forecasts one heavy-rain cell-day for every seven observed).
3. **B1 fixes the rain-frequency bias and improves heavy-rain detection** (ETS ≥ 64.5 mm +30%, +56%, +85% at leads
   1–3) but raises RMSE, because matching the observed spread incurs the double penalty.
4. **Spatial context dominates B2**: the 7×7 neighbourhood mean of forecast rain carries 62% of the split gain and the
   grid-point value 0.2%, consistent with displacement errors.
5. **More training data helps the ML corrector**: going from 6 to 16 seasons lowered B2's lead-1 RMSE from 13.29 to
   13.16 mm and raised its ETS ≥ 64.5 mm from 0.045 to 0.069.

Consequences: the amount forecast and the heavy-rain probability are separate products (Phase 7); regime-aware models
(Phase 6) are compared against B1 on event and spatial scores and against B2 on continuous scores; improvements are
claimed only with block-bootstrap confidence intervals.

## 5. Weather regimes (Phase 5)

### 5.1 Labels (observation-derived; training targets and scoring only)

| Regime | Rule | Evidence |
|---|---|---|
| ACTIVE / BREAK | cos(lat)-weighted IMD rainfall over the core monsoon zone (18–28°N, 65–88°E, IITM operational description), standardised by day-of-year mean and SD of 1981–2010 (±15-day smoothing); > +1 / < −1 for ≥ 3 consecutive days (Rajeevan et al. 2010) | **published** in July–August; **derived** in June and September |
| MONSOON_DEPRESSION | IMD RSMC best-track fix of depression grade or stronger inside 12–30°N, 68–92°E during the rain day | **heuristic** (official track, our box) |
| NORMAL | none of the above | — |

Priority: MONSOON_DEPRESSION > ACTIVE > BREAK > NORMAL. Base-period July–August anomalies have mean 0.03 and SD 1.00.

Training years (2000–2015, 1,952 rain days): NORMAL 71.3%, BREAK 12.0%, ACTIVE 8.6%, MONSOON_DEPRESSION 8.0%.
Counts agree with known seasons: the drought years 2002, 2009 and 2015 have 27, 36 and 33 break days; 2013 (good
monsoon) has 21 active days; 2005–2006 have 24 depression days each.

Why IMD's best track and not the ERA5 low-pressure-system catalogue for depressions: in JJAS 2010–2015 the two
agree on only 22 of ~53 depression days, yet the catalogue contains *some* low-pressure system on 50 of IMD's 52
depression days. The disagreement is intensity grading, not detection, and IMD grading is the operational standard.
Monsoon lows (any intensity) are present on 68% of days, too often to form a useful regime of their own.

### 5.2 Local regimes (per cell; computed from forecast-time data, heuristic)

- **OROGRAPHIC**: terrain-forced ascent w = V₈₅₀·∇h ≥ 0.05 m/s, from the FORECAST 850 hPa wind and GEFS model terrain.
  A slope threshold (10 m/km) was tried first and rejected: GEFS 0.25° terrain smooths the Western Ghats to a
  90th-percentile slope of 7.7 m/km, so only 5% of the Ghats qualified. The forced-ascent rule gives ~31% of the Ghats
  box at the July peak (the rest is lee slope) and ~0% on the Deccan interior; its seasonal cycle (100 → 174 → 73 → 47
  cells from June to September 2011) follows the monsoon westerlies. Sensitivity (training years 2010–2015, lead 1,
  3.6 M cell-days):

  | Threshold (m/s) | Share of cell-days | Mean obs. rain in / out (mm) | Heavy-rain frequency in / out | Share of all heavy events |
  |---|---|---|---|---|
  | 0.02 | 10.1% | 14.9 / 6.1 | 4.2× | 32% |
  | 0.03 | 6.0% | 18.2 / 6.3 | 5.3× | 25% |
  | **0.05** | **2.7%** | **24.3 / 6.5** | **7.4×** | **17%** |
  | 0.08 | 1.0% | 33.4 / 6.7 | 11.2× | 10% |
  | 0.12 | 0.2% | 50.7 / 6.9 | 19.9× | 4% |

  Heavy-rain frequency rises monotonically with forecast forced ascent, so the class is physically meaningful; 0.05 m/s
  balances contrast against coverage.
- **COASTAL**: ≤ 100 km from the GEFS coastline. **INLAND**: otherwise.
- Limitation: over terrain above ~1.5 km the 850 hPa wind is an extrapolation below ground.

### 5.3 Classifier inputs (forecast-time only)

GEFS mid-rain-day fields at 1° over 0–40°N, 40–110°E (850 hPa u, v and relative vorticity computed on the native 0.25°
grid, MSLP, PWAT) summarised over boxes in `config/regimes.yaml`; forecast core-zone and west-coast rainfall; forecast
core anomaly standardised with the model's own training statistics per lead and month; observed core anomaly on the two
days before initialisation (already published by IMD at issue time — used only by the archive classifier; see §9).

The rule baseline R0 applies the ±1 threshold to the FORECAST anomaly day by day: the 3-day persistence of the published
definition cannot be verified inside a 3-day forecast whose first rain day is unfinished at issue time.

### 5.4 Classifier results (validation 2016–2017, 720 forecasts; `reports/phase5_validation.md`)

| | R0 rule | RandomForest | **LightGBM** | LightGBM live* | CNN | always NORMAL |
|---|---|---|---|---|---|---|
| Macro F1 | 0.542 | 0.495 | **0.554** | 0.532 | 0.341 | 0.207 |
| Balanced accuracy | 0.537 | 0.468 | **0.607** | 0.577 | 0.398 | 0.250 |
| Macro F1, July–August | 0.566 | 0.462 | **0.575** | 0.557 | 0.397 | — |
| Depression F1 | **0.425** | 0.129 | 0.344 | 0.350 | 0.227 | 0 |

*Without the observed-persistence inputs, for live forecasts (§9).

- Every classifier carries real regime information (macro F1 0.34–0.55 vs 0.21 for always-NORMAL).
- With 16 training seasons (5,760 labelled forecasts, up from 2,160) **LightGBM now edges the interpretable rule** on
  macro F1 and balanced accuracy; the rule still detects depressions best. The difference is within the sampling
  uncertainty of ~240 correlated validation days, so no strict ranking is claimed.
- **The CNN classifier is not justified** on this data: lowest macro F1 (it over-predicts MONSOON_DEPRESSION). It is
  kept as a reported experiment, not used downstream.
- LightGBM supplies regime probabilities for Phases 6–7 (the rule gives only hard labels). Dropping observed
  persistence costs 0.022 macro F1 — the price of running live.

## 6. Regime-aware correction (Phase 6)

- Training rows receive **out-of-fold** regime probabilities (classifier refit without that year), so the corrector
  learns from regime information as imperfect as it will be in operation. `C1_oracle` (observed regimes) is reported
  only as a diagnostic upper bound.
- Ablation `B2s` (B2 + elevation + distance to coast, no regime) separates the value of regime information from the
  value of static terrain features, which the local regime partly encodes.
- Claims of improvement require a paired block-bootstrap 95% interval (5-day blocks of valid dates, all cells and
  leads of a day kept together) that excludes zero.

### 6.1 Results (validation 2016–2017; `reports/phase6_validation.md`)

| Lead 1 | Raw | QM (B1) | B2 | B2s | **C1** | C2 | C1-oracle* |
|---|---|---|---|---|---|---|---|
| RMSE (mm) | 15.68 | 16.57 | 13.16 | 13.16 | **13.07** | 13.10 | 13.09 |
| Correlation | 0.471 | 0.472 | 0.567 | 0.567 | **0.575** | 0.573 | 0.574 |
| ETS ≥ 64.5 mm | 0.103 | **0.134** | 0.069 | 0.070 | 0.078 | 0.083 | 0.090 |
| FSS ≥ 64.5 mm, 9×9 | 0.529 | **0.576** | 0.229 | 0.234 | 0.273 | 0.298 | 0.312 |

C1 has the lowest RMSE at every lead (lead 2: 13.66 vs raw 15.76; lead 3: 14.06 vs 16.06).
*Oracle uses observed regimes — not an operational forecast.

Paired 5-day block bootstrap, all leads pooled (positive = second model better):

| Comparison | RMSE improvement (95% CI) | ETS ≥ 64.5 improvement (95% CI) |
|---|---|---|
| C1 vs B2s | **+0.077 mm [+0.040, +0.122] — significant** | +0.003 [−0.002, +0.007] — not significant |
| C2 vs B2s | +0.028 [−0.018, +0.079] — not significant | +0.007 [−0.003, +0.014] — not significant |
| B2s vs B2 (terrain only) | +0.006 [−0.003, +0.014] — not significant | +0.001 [−0.001, +0.003] — not significant |

Findings:
1. **Regime information helps, modestly, within the ML-corrector family.** C1 lowers RMSE significantly (0.08 mm,
   ≈ 0.6%) and every heavy-rain score of C1/C2 is above the regime-blind models at lead 1 (ETS 0.070 → 0.078/0.083,
   FSS 0.234 → 0.273/0.298). Terrain features alone add nothing significant, so the gain is the regime information.
   With six seasons C2 had a significant ETS gain; with sixteen neither ETS gain is significant — reported as found.
2. **The ceiling is low even with perfect regime knowledge**: C1-oracle is no better than C1 on RMSE, so better regime
   *classification* would not change the picture much for mean-regression correctors.
3. **Where regimes matter most — orographic cells**: heavy-rain ETS 0.083 (raw) → 0.155 (B2s) → 0.179 (C1) → 0.186
   (C2); RMSE 29.97 → 26.11 mm (C1). The clearest regime signal in the data (subgroup, no interval computed).
4. **In MONSOON_DEPRESSION regimes raw GEFS keeps the best heavy-rain skill** (ETS 0.129 vs 0.054 for C1 and 0.067 for
   C2): the correctors wash out correctly placed extremes. Reported as found — the heavy-rain probability (§6.3) and
   quantile mapping are the products for extremes.
5. **Product choice: C1** (lowest RMSE at every lead, significant gain) is the dashboard's corrected-rainfall layer.

### 6.2 Regime-conditioned quantile mapping (Phase 6b; `reports/phase6b_validation.md`)

Tested because quantile mapping is the only family here that preserves heavy rain. C3 fits one mapping per
predicted synoptic regime, per cell and lead; compared with global QM (B1) on identical rows.

| C3 vs B1, all leads (positive = C3 better) | Improvement | 95% CI |
|---|---|---|
| ETS ≥ 64.5 mm | −0.0027 | [−0.0041, −0.0007] — significantly worse |
| ETS ≥ 115.6 mm | −0.0038 | [−0.0073, −0.0010] — significantly worse |
| RMSE | −0.38 mm | [−0.48, −0.27] — significantly worse |

With six seasons the cause looked like sample size. **Extending training to 16 seasons narrowed the gap (RMSE penalty
0.51 → 0.38 mm) but did not close it**, and the oracle version (observed regimes) is still worse than B1. Splitting the
distribution by regime adds sampling noise to the tails faster than it adds signal. **Regime-conditioned QM is
rejected.**

### 6.3 Heavy-rain probability (Phase 7; `reports/phase7_validation.md`)

LightGBM binary classifiers, unweighted log-loss, rounds chosen on the last training year and refit on all training
years. Reference: climatological event frequency (training years). Only the GEFS control member is used, so raw and QM
forecasts can only be scored as 0/1 exceedances.

| ≥ 64.5 mm (51,423 events; base rate 1.42%) | CLIM | RAW 0/1 | QM 0/1 | P_blind | **P_regime** |
|---|---|---|---|---|---|
| Brier skill vs climatology | 0 | −0.34 | −0.66 | 0.071 | **0.077** |
| ROC AUC | 0.778 | 0.549 | 0.590 | 0.880 | **0.883** |
| BSS lead 1 / 2 / 3 | — | — | — | — | 0.109 / 0.072 / 0.049 |
| AUC lead 1 / 2 / 3 | — | — | — | — | 0.907 / 0.880 / 0.860 |

| ≥ 115.6 mm (11,036 events; base rate 0.30%) | CLIM | RAW 0/1 | QM 0/1 | P_blind | **P_regime** |
|---|---|---|---|---|---|
| Brier skill vs climatology | 0 | −0.28 | −0.80 | 0.032 | **0.038** |
| ROC AUC | 0.788 | 0.525 | 0.566 | 0.880 | **0.886** |

- The probability models discriminate heavy-rain cells far better than the raw forecast (AUC 0.88 vs 0.55) and beat
  climatology on Brier score at every lead.
- Deterministic yes/no exceedances of rare events score worse than climatology — the case for issuing heavy rain as
  a probability.
- **Regime information significantly improves the heavy-rain Brier score** (P_regime vs P_blind: +0.000080,
  95% CI [+0.000027, +0.000124]); for ≥ 115.6 mm the gain (+0.000015) is not significant.
- Reliability ≥ 64.5 mm is close to the diagonal up to 0.7 (e.g. forecast 0.343 → observed 0.344).

**Recalibration.** Isotonic regression fitted on leave-one-year-out predictions over the training years was tested.
With 16 seasons it no longer helps: Brier changes by −0.000003 (heavy) and +0.000007 (very heavy), both not significant,
while very-heavy AUC drops 0.886 → 0.854. **Probabilities are used as fitted** (`config/models.yaml →
heavy_rain.calibrate`). With six seasons very-heavy probabilities were over-confident and calibration was needed —
more data fixed the model rather than the symptom.

### 6.4 Overall answer to the problem statement's hypothesis (validation years)

- **Post-processing delivers the large gains**: lead-1 RMSE −17% (raw 15.68 → 13.07 mm with C1), correlation
  0.47 → 0.58; heavy-rain discrimination AUC 0.55 → 0.88 (0.91 at lead 1); rain-frequency bias fixed by quantile mapping.
- **Regime-awareness adds statistically significant but small gains** where the method family can use it (continuous
  RMSE via C1, heavy-rain Brier via P_regime), is clearest in orographic cells, and **hurts** when it fragments the
  distribution (C3). In depressions the raw model's extremes are better than any corrector's.
- These are validation results. The unbiased estimate is the one-time evaluation in §11.

## 7. District product

Area-weighted mean over the 0.25° IMD land cells overlapping each Census-2011 district (exact polygon overlap in an
equal-area projection, EPSG:6933); the maximum over overlapping cells for heavy-rain probability. 608 of 641 districts are
≥ 95% covered; the island districts (Andaman & Nicobar, Lakshadweep) have no IMD cells and return no value; Chennai is
4.5% covered and flagged. District tables download as CSV; a printable bulletin ranks districts by heavy-rain chance.

## 8. Forecast products and dashboard

- One product per 00 UTC initialisation (`data/products/`), built only from saved artifacts by `ProductBuilder`: grids
  (raw, C1 corrected, quantile-mapped, heavy/very-heavy probability, local regime), the district table, regime
  probabilities, and an explanation assembled from the corrector's own SHAP contributions and validation scores.
- Observed IMD rainfall is stored as a separate, labelled variable for verification and is never an input.
- Each product records its source (archive / live), the classifier used and that classifier's validation score, and
  whether it can be verified yet. Held-out dates are refused without `--final-test`.
- Case studies (`config/events.yaml`) are chosen from the observations: the days with most IMD cells ≥ 115.6 mm per
  period and region.
- English and Hindi interface; Hindi rainfall terms follow IMD usage (review by a native speaker pending).
- Map colours: IMD rainfall categories on a single-hue blue ramp and rare-event probability bins on a single-hue orange
  ramp, each validated for light and dark surfaces. The lowest class is unfilled so colour appears only where something
  happens.

## 9. Live mode (operational GEFSv12)

- `scripts/run_live.py` downloads the newest complete 00 UTC operational run (≈ 45 s for days 1–3) and builds a product
  through the same `ProductBuilder` code, marked **live**. The GEFSv12 reforecast and operational system share the model
  version (operational since 2020-09-23); the initial-condition analysis differs, which is why the operational test
  (§11) scores the live pipeline separately.
- **Resolution**: operational 850 hPa wind and MSL pressure are published at 0.5° only. They are interpolated to 0.25°
  and processed identically. On 41 validation days (123 forecasts) the degraded inputs change the classifier's regime in
  3.3% of forecasts and its probabilities by 1.8 points on average; the largest feature effect is on the trough
  vorticity maximum (−0.13 SD, coarser winds smooth vorticity peaks) (`reports/operational_fields_check.json`).
- **Observations**: IMD publishes a year's grid only after the year ends, so today's forecasts cannot use observed
  persistence. Live runs use `LGBM_live`, trained without those inputs (macro F1 0.532 vs 0.554), rather than feeding
  the archive model inputs it never saw missing. Verification is marked pending until IMD publishes the days.
- **Season**: models are trained on June–September. Out-of-season runs are refused unless explicitly requested, and
  then carry a warning. (The SIH finale falls outside the monsoon; live demonstrations then show this warning.)

## 10. Deep-learning corrector (Phase 8; `reports/phase8_validation.md`)

A U-Net sees the whole forecast map (log rain, observed climatology, terrain, coast distance, land) plus lead, season and
regime probabilities (out-of-fold for training years, as for C1) as constant planes, and predicts rain with the same
Tweedie deviance as the LightGBM correctors, on observed land cells only. ~0.1 M parameters; 7 epochs chosen by early
stopping on 2015, refit on 2000–2015; 35 min on a laptop CPU. Scored on the same validation rows as A, B2 and C1.

| Lead 1 (validation 2016–2017) | Raw | B2 | C1 | **U-Net** |
|---|---|---|---|---|
| RMSE (mm) | 15.68 | 13.16 | 13.07 | **12.93** |
| MAE (mm) | 7.83 | 6.44 | **6.40** | 6.77 |
| Bias (mm) | +1.63 | −0.22 | −0.22 | +1.07 |
| Correlation | 0.471 | 0.567 | 0.575 | **0.593** |
| ETS ≥ 64.5 mm | 0.103 | 0.069 | 0.078 | **0.121** |
| FSS ≥ 64.5 mm, 9×9 | **0.529** | 0.229 | 0.273 | 0.460 |

U-Net vs C1, all leads pooled (5-day block bootstrap): RMSE **+0.126 mm [+0.004, +0.263]**, ETS ≥ 64.5 mm
**+0.031 [+0.018, +0.046]** — both significant. The U-Net has the lowest RMSE and highest correlation at every lead, and
is the only corrector whose heavy-rain ETS beats the raw forecast at lead 1 (0.121 vs 0.103; quantile mapping 0.134).
Seeing the whole map lets it move displaced rain rather than only smooth it. Its costs are a **wet bias** (+0.9 to
+1.3 mm) and a higher MAE than C1, so it does not dominate on every metric.

Decision (validation only): C1 stays the main corrected layer — it is unbiased and has the lowest MAE — and the U-Net is
added as a separate, clearly labelled layer and scored in the one-time evaluation alongside the other models. The CNN
*classifier* (§5.4) remains unused.

## 11. One-time evaluation on held-out data (`reports/final_evaluation.md`)

`scripts/run_final_evaluation.py` scored the frozen models once (2026-09-29, 04:05–04:14 IST) through the same
`ProductBuilder` code that makes dashboard products: the **test years 2018–2019** through the archive pipeline, and
**610 real GEFSv12 operational forecasts, JJAS 2021–2025**, through the live pipeline (forecast-only classifier, 0.5°
inputs interpolated). No model, threshold or setting was changed after seeing these numbers. The lock file records the
hashes of the evaluated model files; the U-Net weights (`models/cnn_bc/unet.pt`, saved 03:44 IST, SHA-256 `47f84876…`)
were omitted from that list by a file-type filter, since fixed.

| Lead 1 | Test 2018–2019 (3.6 M cell-days) | | | Operational 2021–2025 (9.1 M cell-days) | | |
|---|---|---|---|---|---|---|
| | RMSE (mm) | Corr. | ETS ≥ 64.5 | RMSE (mm) | Corr. | ETS ≥ 64.5 |
| Raw GEFS | 16.73 | 0.487 | 0.108 | 15.01 | 0.483 | 0.088 |
| Quantile mapping | 17.99 | 0.496 | 0.151 | 15.67 | 0.481 | **0.138** |
| Global LightGBM (B2) | 13.49 | 0.599 | 0.099 | 13.68 | 0.579 | 0.067 |
| **Regime-aware C1** | **13.39** | 0.607 | 0.098 | 13.64 | 0.585 | 0.070 |
| Regime-split C2 | 13.47 | 0.601 | 0.100 | 13.69 | 0.579 | 0.068 |
| **U-Net** | 13.51 | **0.613** | **0.155** | **13.27** | **0.602** | 0.096 |

| Heavy-rain probability (all leads) | Test AUC | Test BSS | Operational AUC | Operational BSS |
|---|---|---|---|---|
| ≥ 64.5 mm (raw 0/1 AUC: 0.563 / 0.543) | **0.897** (lead 1: 0.914) | +0.101 | **0.880** (lead 1: 0.901) | +0.084 |
| ≥ 115.6 mm | 0.908 | +0.046 | 0.877 | +0.038 |

Paired 5-day block bootstrap, all leads pooled (positive = second better):

| Comparison | Test RMSE | Test ETS ≥ 64.5 | Operational RMSE | Operational ETS ≥ 64.5 |
|---|---|---|---|---|
| C1 vs raw | **+2.56 [2.31, 2.79]** | −0.003 (n.s.) | **+1.78 [1.62, 1.97]** | **−0.014 [−0.027, −0.002]** |
| C1 vs regime-blind B2 | **+0.082 [0.025, 0.138]** | +0.004 (n.s.) | **+0.074 [0.046, 0.103]** | **+0.004 [0.001, 0.007]** |
| C2 vs B2 | +0.005 (n.s.) | +0.002 (n.s.) | +0.017 (n.s.) | +0.003 (n.s.) |
| U-Net vs C1 | +0.020 (n.s.) | **+0.044 [0.029, 0.062]** | **+0.236 [0.157, 0.331]** | **+0.029 [0.021, 0.039]** |
| P_regime vs climatology, Brier (≥ 64.5 mm) | **+0.0016** | | **+0.0013** | |

Findings, stated as found:
1. **The validation conclusions hold on unseen data.** Post-processing cuts lead-1 RMSE by 20% on the test years
   (16.73 → 13.39 mm) and by 9–12% on real operational forecasts; heavy-rain AUC rises from ~0.55 to 0.88–0.90.
2. **Regime-awareness helps, significantly, on both held-out sets**: C1 beats the regime-blind B2 on RMSE in both, and on
   heavy-rain ETS operationally. The gains remain small (≈ 0.6% RMSE). Regime-split C2 shows no significant gain.
3. **The U-Net is the most useful amount forecast for heavy rain**: significantly better heavy-rain ETS than C1 on both
   sets, better than raw GEFS and on the test years better than quantile mapping at lead 1; significantly lower RMSE
   than C1 operationally, not on the test years.
4. **Operational forecasts are not the reforecast**: C1 and B2 are **1.1–1.2 mm too dry** at lead 1 on operational runs
   (raw GEFS operational bias +0.2 mm vs +2.3 mm on the reforecast test years), and C1's heavy-rain ETS falls
   significantly below raw. The U-Net is nearly unbiased there (−0.13 mm). Retraining on operational seasons is the
   fix (LIMITATIONS).
5. **Monsoon depressions**: raw GEFS keeps better heavy-rain ETS than the LightGBM correctors on both sets (test: raw
   0.127 vs C1 0.113; operational: raw 0.113 vs C1 0.047). The U-Net beats raw on the test years (0.162) but not
   operationally (0.107). Orographic cells again gain most (operational
   ETS raw 0.086 → C1 0.177 → U-Net 0.196).
