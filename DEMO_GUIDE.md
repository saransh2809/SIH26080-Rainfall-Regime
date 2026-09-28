# Demo Guide — SIH 26080

The demonstration runs **offline** on saved products of **real** data; no internet is needed once data and products
are built. The header always states the data kind, and every forecast shows whether it is an archived forecast
(verifiable against IMD) or a live one (verification pending).

## Before the demo

1. Products exist in `data/products/` (`scripts/build_products.py`, `scripts/run_live.py`). The **Case study** menu lists
   every event in `config/events.yaml` whose products are built.
2. Start the dashboard — either one command (Docker Desktop running):

   ```bash
   docker compose up --build
   ```

   and open http://localhost:8080 — or two terminals:

   ```bash
   .venv/Scripts/python -m uvicorn rainpp.api.app:app --port 8000
   ```

   ```bash
   node frontend/node_modules/vite/bin/vite.js frontend --port 5173
   ```

   and open http://localhost:5173.
3. Every view is a shareable link, e.g. the Mumbai case:
   `/?init=2017-08-29&lead=1&layer=p_heavy_max&district=518` (add `&lang=hi` for Hindi).

## Suggested walkthrough (≈ 5 minutes)

| Step | Action | What to say |
|---|---|---|
| 1 | Point to the header badge and the source card | "Everything here is real data. This is post-processing of a weather model's forecast, not a new weather model. This card says where the forecast came from and whether it can be checked yet." |
| 2 | **Case study → Mumbai, 30 Aug 2017**; layer **Raw NWP** | "The raw GEFS forecast issued at 00 UTC on 29 August for the rain day ending 08:30 IST on the 30th." |
| 3 | Regime card | "The classifier reads the forecast's own wind, pressure and moisture fields: active monsoon, 56%. Its validation score is printed right here — guidance, not certainty." |
| 4 | Layer **P(≥64.5 mm)** | "Heavy-rain probability lights up the Konkan coast. Mumbai Suburban reaches 47% — about 33 times the 1.4% base rate." |
| 5 | Layer **Observed (IMD)** | "What IMD observed afterwards — shown for verification only, never an input." |
| 6 | District panel (Mumbai Suburban is selected) | "Raw 60 mm, regime-aware correction 61 mm, quantile-mapped 94 mm, observed 156 mm. Amount forecasts under-estimate extremes; that is why the probability product exists." |
| 6b | Layer **U-Net (deep learning)** | "A deep-learning model that sees the whole forecast map. On validation it beat our LightGBM corrector significantly on error and on heavy rain — it is wetter on average, so it is shown alongside, not instead." |
| 7 | "Why this correction" | "These are the model's own feature contributions (SHAP) for this day, not written text." |
| 8 | **Case study → Kerala floods, 16 Aug 2018** (test year) | "Never used to train or choose a model. Kottayam: raw 95 mm, corrected 62, observed 210 — but heavy-rain chance 61%." |
| 8b | **Case study → Meghalaya, 16 Jun 2022** (live-mode test) | "A real NOAA operational forecast through our live pipeline. West Khasi Hills observed 360 mm; every amount forecast was under 110 mm, the heavy-rain chance was 72%." |
| 9 | Model comparison table, then `reports/final_evaluation.md` | "On the unseen test years error drops 20% and heavy-rain AUC goes 0.56 → 0.90; on 610 real daily forecasts error drops up to 12% and AUC reaches 0.88. We also report where it is worse: on operational runs the tree correctors are ~1 mm too dry." |
| 10 | **Export → District table (CSV)** and **Print / save PDF** | "District tables for disaster managers, and a one-page bulletin ranking districts by heavy-rain chance." |
| 11 | **हिन्दी** button | "The whole interface switches to Hindi; the link keeps the language." |
| 12 | (Optional) terminal: `.venv/Scripts/python scripts/run_live.py --latest --allow-out-of-season` | "Live mode pulls today's NOAA run and builds this product in under a minute. Outside the monsoon it warns that the models were trained on June–September." |

## Honest answers to likely questions

- *Is this NCMRWF's model?* No — NOAA GEFSv12, because no NCMRWF archive is public. The pipeline retrains on NCUM data
  through the same adapter interface.
- *Does regime-awareness help?* Yes, a little, with confidence intervals: C1's RMSE gain and the heavy-rain Brier gain
  are significant; regime-split quantile mapping is significantly worse; in monsoon depressions raw GEFS places heavy
  rain better than any corrector. See LIMITATIONS.md.
- *Why is corrected rainfall lower than observed in Mumbai?* Correctors predict expected values and smooth extremes; the
  heavy-rain probability is the product designed for extremes.
- *Did you tune on the test data?* No. Test and operational-test years are scored once by a script that records the
  model hashes and refuses to re-run silently.
- *Deep learning?* A CNN regime classifier and a U-Net rain corrector were built and compared against the tree models
  on the same data; results are in VERIFICATION.md, reported whether they win or lose.
