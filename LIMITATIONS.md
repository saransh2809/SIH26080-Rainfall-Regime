# Limitations — SIH 26080

Stated plainly so that results are read for what they are.

## Data

| Limitation | Consequence | What would fix it |
|---|---|---|
| The NWP model is **NOAA GEFSv12 (reforecast)**, not NCMRWF's NCUM/NEPS; no NCMRWF forecast archive is publicly downloadable | The *method* transfers; the *trained models* do not — every model must be retrained on NCUM forecasts | NCMRWF forecast archive (requested via SIH); the adapter interface (`ForecastSource`) is ready |
| Only the GEFS **control member** is used | No ensemble spread; heavy-rain probability comes from post-processing, not ensemble counting | Download the 4 perturbed members (same pipeline) |
| Lead days 1–3 only, JJAS only | No medium-range (day 4–10) or winter (western disturbance) products | Extend `lead_days` / `season.months`; WD labels would need a WD catalogue |
| IMD 0.25° gridded rainfall is a gauge **analysis**, not truth | Sparse-gauge regions (NE hills, Himalaya) carry analysis error into both training and verification | Merged gauge–satellite products for cross-checks |
| IMD date convention was established **empirically** (window-end labelling; DATA_SOURCES §7) | If wrong, every pairing would be shifted by a day | An authoritative IMD statement; the evidence (0.43 vs 0.34 correlation) is strong but indirect |
| District boundaries are **Census 2011** (641 districts; ~780 today) | New districts (e.g. in Telangana) are not shown separately | LGD-based current boundaries, Survey of India compliant |
| Island districts (Andaman & Nicobar, Lakshadweep) have no IMD grid cells | No district values there; shown as "no data" | A product covering the islands |

## Regimes

- Active/break labels use the **published** Rajeevan et al. (2010) criterion only for July–August; June and
  September labels apply it outside its original scope (**derived**).
- The core-zone box is IITM's approximate operational description, not the exact published polygon.
- Depression labels use IMD's official best track with a **heuristic** box; the ERA5 LPS catalogue disagrees on
  intensity grading for about half of depression days.
- Orographic/coastal regimes are **heuristic** rules. The forced-ascent threshold (0.05 m/s) is supported by a
  training-year sensitivity analysis (heavy rain 7.4× more frequent inside the class; METHODOLOGY §5.2), but the
  100 km coastal distance has not been tested.
- Over terrain above ~1.5 km the 850 hPa wind used for orographic forcing is below ground (model extrapolation).
- The regime classifier is imperfect (validation macro F1 ≈ 0.5); ML did not beat a simple rule.

## Models and results

- **Every rainfall-amount corrector under-forecasts extremes.** Mean-regression correctors (LightGBM) wash out heavy
  rain; quantile mapping keeps its frequency but not its location. Heavy rain must be read from the probability layer.
- **Regime-awareness gives small gains** (e.g. RMSE −0.06 mm, heavy-rain Brier +0.00007, both significant) and
  **harms** regime-split quantile mapping; with limited training seasons, data volume dominates.
- In MONSOON_DEPRESSION regimes the raw forecast has better heavy-rain ETS than every ML corrector.
- Very-heavy (≥ 115.6 mm) probabilities are calibrated at the cost of discrimination (AUC 0.87 → 0.78) and have
  near-zero skill by day 3.
- Validation results are from two monsoon seasons (2016–2017); some model choices (classifier, calibration) were made
  on validation scores, so validation figures are mildly optimistic. The test years (2018–2019) give the unbiased
  estimate and are evaluated once.
- Confidence intervals use a 5-day block bootstrap; longer spells may make them slightly too narrow.
- Training LightGBM on 16 seasons uses a 50% checkerboard of cells (memory); quantile mapping and all evaluation use
  every cell.

## Engineering

- Forecast products are **pre-computed** per date; the dashboard does not run models live.
- Real-data mode for current forecasts (operational GEFS / NCUM) is designed but not yet connected.
