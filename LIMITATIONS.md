# Limitations — SIH 26080

Stated plainly so that results are read for what they are.

## Data

| Limitation | Consequence | What would fix it |
|---|---|---|
| The NWP model is **NOAA GEFSv12**, not NCMRWF's NCUM/NEPS; no NCMRWF forecast archive is publicly downloadable | The *method* transfers; the *trained models* do not — every model must be retrained on NCUM forecasts | NCMRWF forecast archive (request via SIH); the adapter interface (`ForecastSource`) is ready |
| Training uses the GEFSv12 **reforecast**; live mode uses the **operational** runs, whose initial conditions differ | Measured (METHODOLOGY §11): raw operational GEFS is less wet than the reforecast, so the LightGBM correctors are ~1.2 mm too dry at lead 1 and their heavy-rain ETS falls below raw; the U-Net is nearly unbiased | Retrain or recalibrate on the 2021–2025 operational seasons now downloaded (they would then no longer be a held-out test) |
| Operational 850 hPa wind and MSL pressure exist only at **0.5°** | Interpolated to 0.25°; the regime call changes in 3.3% of validation forecasts; trough vorticity maxima read ≈ 0.13 SD low | Use 0.25° fields if NOAA publishes them, or retrain the classifier on 0.5° inputs |
| Only the GEFS **control member** is used | No ensemble spread; heavy-rain probability comes from post-processing, not ensemble counting | Download the perturbed members (same pipeline) |
| Lead days 1–3 only, **JJAS only** | No medium-range or winter products; live runs outside June–September are refused by default and carry a warning otherwise | Extend `lead_days` / `season.months` and retrain |
| IMD 0.25° gridded rainfall is a gauge **analysis**, not truth | Sparse-gauge regions (NE hills, Himalaya) carry analysis error into both training and verification | Merged gauge–satellite products for cross-checks |
| IMD publishes each year's grid only **after the year ends** | Live forecasts cannot use observed persistence and remain unverified until then | IMD / NCMRWF real-time merged rainfall as a provisional verification source |
| IMD date convention was established **empirically** (window-end labelling; DATA_SOURCES §7) | If wrong, every pairing would be shifted by a day | An authoritative IMD statement; the evidence (0.43 vs 0.34 correlation) is strong but indirect |
| District boundaries are **Census 2011** (641 districts; ~780 today) | New districts and states (e.g. Telangana) are not shown separately | LGD-based current boundaries, Survey of India compliant |
| Island districts (Andaman & Nicobar, Lakshadweep) have no IMD grid cells | No district values there; shown as "no data" | A product covering the islands |

## Regimes

- Active/break labels use the **published** Rajeevan et al. (2010) criterion only for July–August; June and
  September labels apply it outside its original scope (**derived**).
- The core-zone box is IITM's approximate operational description, not the exact published polygon.
- Depression labels use IMD's official best track with a **heuristic** box; the ERA5 LPS catalogue disagrees on
  intensity grading for about half of depression days.
- Orographic/coastal regimes are **heuristic** rules. The forced-ascent threshold (0.05 m/s) is supported by a
  training-year sensitivity analysis (METHODOLOGY §5.2); the 100 km coastal distance has not been tested.
- Over terrain above ~1.5 km the 850 hPa wind used for orographic forcing is below ground (model extrapolation).
- The regime classifier is imperfect (validation macro F1 0.554; 0.532 for the live variant) and only marginally
  better than a simple rule, which still detects depressions best.

## Models and results

- **Every rainfall-amount corrector under-forecasts extremes.** Mean-regression correctors (LightGBM) wash out heavy
  rain; quantile mapping keeps its frequency but not its location. Heavy rain must be read from the probability layer.
- **Regime-awareness gives small gains** (C1 RMSE −0.08 mm and heavy-rain Brier +0.00008, both significant) and
  **harms** regime-split quantile mapping even with 16 training seasons.
- In MONSOON_DEPRESSION regimes the raw forecast has better heavy-rain ETS than every ML corrector.
- Very-heavy (≥ 115.6 mm) probabilities have Brier skill of only 0.02–0.05 by lead; skill falls with lead time.
- Validation covers two monsoon seasons (2016–2017) and all model choices were made on it, so validation figures are
  mildly optimistic. The held-out evaluation (test 2018–2019, operational 2021–2025) gives the unbiased estimate and
  is run once.
- Confidence intervals use a 5-day block bootstrap; longer spells may make them slightly too narrow.
- LightGBM correctors are fitted on a 50% checkerboard of cells (memory); quantile mapping and all evaluation use
  every cell.

## Engineering

- Products are computed per forecast run (≈ 45 s live, a few seconds from the archive), not on request; the dashboard
  serves saved products.
- Live mode needs a daily scheduled run (`scripts/run_live.py --latest`); no scheduler is installed automatically.
- The Hindi interface text has not yet been reviewed by a native speaker.
