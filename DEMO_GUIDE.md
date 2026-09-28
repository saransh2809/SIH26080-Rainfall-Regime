# Demo Guide — SIH 26080

The demonstration runs **offline** on archived **real** data: no internet is needed once the data and products are
built. The dashboard header always shows the data kind ("Offline demonstration · archived real data").

## Before the demo

1. Products exist in `data/products/` (built with `scripts/build_products.py`). Current set: 24–31 Aug 2017
   (validation period).
2. Start the backend and the dashboard:

   ```bash
   .venv/Scripts/python -m uvicorn rainpp.api.app:app --port 8000
   ```

   ```bash
   node frontend/node_modules/vite/bin/vite.js frontend --port 5173
   ```

3. Open http://localhost:5173.

## Suggested walkthrough (≈ 4 minutes)

| Step | Action | What to say |
|---|---|---|
| 1 | Point to the header badge | "Everything here is real archived data; this is post-processing, not a weather model." |
| 2 | Select **2017-08-29**, **Day 1**, layer **Raw NWP** | "This is the raw GEFS forecast for the rain day ending 08:30 IST on 30 August 2017." |
| 3 | Regime card | "The classifier reads the forecast's own wind, pressure and moisture fields and calls this an active monsoon day (54%). The observed label was active too." |
| 4 | Layer **P(≥64.5 mm)** | "Heavy-rain probability lights up the Konkan coast. Mumbai reaches 51% — about 35 times the climatological rate." |
| 5 | Layer **Observed (IMD)** | "This is what IMD observed afterwards — shown for verification only, never used as an input." |
| 6 | Type "Mumbai Suburban, Maharashtra" in **Find district** | "Raw 60 mm, corrected 61 mm, quantile-mapped 94 mm, observed 156 mm: amount forecasts under-estimate the extreme, which is why the probability product matters — 50% chance of heavy rain here." |
| 7 | "Why this correction" panel | "These are the model's own feature contributions (SHAP), not written text." |
| 8 | Model comparison table | "Validation 2016–2017. Post-processing cuts RMSE ~15%; regime information adds small but significant gains." |

## Honest answers to likely questions

- *Is this NCMRWF's model?* No — NOAA GEFSv12, because no NCMRWF archive is public. The pipeline retrains on NCUM data.
- *Does regime-awareness help?* A little, with confidence intervals; and in one variant it hurts. See LIMITATIONS.md.
- *Why is the corrected rainfall lower than observed in Mumbai?* Correctors predict expected values and smooth
  extremes; the heavy-rain probability is the product designed for extremes.

## Test-period case

The August 2018 Kerala floods fall in the held-out test period. Their products are built only together with the
one-time final evaluation (`build_products.py --final-test`).
