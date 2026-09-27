# Data Sources — SIH 26080 Regime-Aware Rainfall Post-Processing

Status: **Phase 1 research draft** (2026-09-27). Nothing here has been downloaded yet.
Every entry is marked with how it was checked:

- **Verified** — confirmed on the provider's own page during research.
- **Secondary** — confirmed only via papers/third-party pages; must be re-checked against the provider during Phase 3 download tests.

## 1. Why the data is split into roles

This system is a **post-processor**: it learns `NWP forecast → observed rainfall`. The data therefore falls into four roles that must never be mixed up:

| Role | Used for | Must be available at forecast time? |
|---|---|---|
| **A. Forecast / predictor** | Model inputs (raw NWP rainfall + forecast weather fields) | Yes |
| **B. Observation / target** | Training target and verification truth | No (arrives after the fact) |
| **C. Regime labelling** | Creating the "true" regime labels for training the classifier | No |
| **D. Static** | Terrain, coastline, district boundaries | Yes (never changes) |

Rule: role **B** and **C** data must never appear as model *inputs* at inference time. The regime classifier predicts the regime from **forecast** fields (role A), and is trained against labels derived from observations/reanalysis (roles B/C).

## 2. Forecast / predictor data (role A)

### 2.1 NOAA GEFSv12 Reforecast — **recommended primary training source**

| Field | Value |
|---|---|
| Provider | NOAA (NCEP / PSL), via NOAA Open Data Dissemination on AWS |
| URL | https://registry.opendata.aws/noaa-gefs-reforecast/ · bucket `s3://noaa-gefs-retrospective` · doc: https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf |
| Type | **Forecast** (retrospective runs of a fixed model version) |
| Period | 2000–2019 (**Verified**) |
| Runs | Daily 00 UTC, 5 members; weekly 11 members to +35 days (**Verified**) |
| Resolution | Global 0.25° (721 × 1440) for days 1–10 (**Verified**, Phase 3). Grid nodes coincide exactly with the IMD grid. |
| Variables | APCP (accumulated precipitation) plus pressure-level / surface fields (winds, humidity, PWAT, MSLP, etc.) |
| Format | GRIB2, one file per variable/member/init: `GEFSv12/reforecast/{YYYY}/{YYYYMMDD}00/{member}/Days:1-10/{var}_{level}_{YYYYMMDD}00_{member}.grib2`, each with a `.idx` byte-offset index (**Verified**) |
| APCP storage | 6-hour buckets: messages alternate `6k→6k+3` and `6k→6k+6` hour totals. Packed to 0.01–0.1 mm steps, so differencing leaves negatives down to −0.10 mm, which are clipped to 0 (**Verified**) |
| Measured cost | Days 1–3 of APCP ≈ 10 MB per init via byte-range requests; ≈ 2 s per init with 8 parallel requests; ≈ 70 kB per init-lead after cropping to India |
| Access | Anonymous HTTPS/S3, no login — programmatic (**Verified**) |
| License | NOAA open data (public domain, US Gov) |
| India suitable | Yes (global) |
| Why chosen | Only free source with a **long, consistent** archive of real NWP forecasts. 20 monsoon seasons from one frozen model version is exactly what post-processing needs. Operational GEFS has run this same v12 system since Sept 2020, so a model trained here can be applied to live GEFS forecasts. |

### 2.2 NOAA GFS 0.25° operational archive — secondary / independent test

| Field | Value |
|---|---|
| Provider | NOAA NCEP; archived by NCAR RDA (ds084.1) and AWS |
| URL | https://gdex.ucar.edu/datasets/d084001/ · https://registry.opendata.aws/noaa-gfs-bdp-pds/ |
| Type | **Forecast** (operational, deterministic) |
| Period | 2015-01-15 → present (**Verified**); RDA copy stops updating in early 2026, AWS copy continues (**Verified**) |
| Runs / steps | 00/06/12/18 UTC; 3-hourly to 240 h (**Verified**) |
| Caveat | Model version changed several times since 2015 (e.g. FV3 upgrade 2019), so biases are **not stationary**. Useful for a recent independent test, not as the main training set. |

### 2.3 NCMRWF NCUM / NEPS forecasts — **the real target system, not publicly available**

| Field | Value |
|---|---|
| Provider | NCMRWF, MoES |
| URL | https://www.ncmrwf.gov.in/data/ (portal refused connections during research) |
| Status | **No public, downloadable historical forecast archive was found.** The PS "Dataset Link" field is also empty. |
| Action | Request an NCUM-G / NEPS rainfall forecast archive through the SIH nodal contact / NCMRWF mentors. The code uses a dataset adapter (`NCMRWFDataset`) so this can drop in without changing model code. The *method* transfers across models; a *trained model* does not — it must be retrained on NCUM data. |

### 2.4 ERA5 — **not a forecast predictor** (correction to the earlier plan)

ERA5 precipitation comes from short-range (≤18 h) forecasts that are heavily constrained by data assimilation ([ECMWF docs](https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790)). Its errors do not resemble day-1 to day-5 medium-range forecast errors, so training a "bias corrector" on ERA5 would correct the wrong thing. ERA5 is kept only for **regime labelling** (role C, section 4).

## 3. Observation / target data (role B)

### 3.1 IMD 0.25° gridded daily rainfall — **primary target**

| Field | Value |
|---|---|
| Provider | India Meteorological Department, Pune |
| URL | https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html (NetCDF) · https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html (binary) |
| Type | **Observation** (gauge-based analysis, Pai et al. 2014, MAUSAM) |
| Period | 1901–2024 (**Verified**) |
| Grid | 0.25°, 135 × 129 points, 6.5°N–38.5°N, 66.5°E–100.0°E (**Verified**) |
| Rain day | 24 h ending 03 UTC (08:30 IST), **labelled by the date the window ends** (**Verified empirically**, Phase 3 — see §7) |
| Access | Free. Yearly NetCDF via `POST https://imdpune.gov.in/cmpg/Griddata/RF25.php` with form field `RF25=<year>` (≈25 MB/year). The server drops connections intermittently; the downloader retries. (**Verified**) |
| File layout | `RAINFALL` (mm) on `TIME`, `LATITUDE`, `LONGITUDE`; cells outside India are NaN; no negative values (**Verified**, 2010–2018 files) |
| District suitable | Yes (0.25° ≈ 27 km; aggregate by area weighting) |

### 3.2 NCMRWF–IMD merged satellite-gauge rainfall (GPM) — recent / real-time truth

| Field | Value |
|---|---|
| Provider | IMD + NCMRWF (Mitra et al. algorithm) |
| URL | https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF_Merged.html · real-time: https://imdpune.gov.in/cmpg/Realtimedata/gpm/Rain_Download.html |
| Type | **Observation** (gauges merged with GPM satellite first guess) |
| Period | 1 Oct 2015 → near real time (**Secondary**, from paper) |
| Grid | 0.25° |
| Use | Verification for REAL DATA mode on recent dates where the gauge-only product is not yet released. Not mixed with 3.1 in the same training set. |

### 3.3 NASA GPM IMERG Final V07 — optional cross-check

| Field | Value |
|---|---|
| URL | https://www.earthdata.nasa.gov/data/catalog/ges-disc-gpm-3imergdf-07 · https://registry.opendata.aws/nasa-gpm3imergdf/ |
| Type | Satellite **observation estimate** |
| Resolution / period | 0.1°, daily; 2000 → present, ~3.5-month latency (**Verified**) |
| Access | Free, NASA Earthdata login |
| Use | Sensitivity check only. IMD gauge analysis remains the official truth for India. |

## 4. Regime-labelling data (role C)

| Dataset | Provider / URL | Use | Check |
|---|---|---|---|
| IMD gridded rainfall (3.1) | IMD | Active/break labels via Rajeevan et al. (2010) criterion | Verified |
| IMD best-track (depressions and stronger) | RSMC New Delhi — https://rsmcnewdelhi.imd.gov.in/ (Excel, 1982→); parsed copy: https://github.com/syedhamidali/imdtrack | Depression labels (official IMD) | Secondary |
| ERA5-derived South Asian LPS catalogue | Zenodo, e.g. https://zenodo.org/records/22151110 (v5.6, 1940–2025); explorer: https://github.com/kieranmrhunt/monsoon-low-atlas | Monsoon **lows** (weaker than depressions, not in IMD best track) | Secondary — confirm authors, licence and citation on the Zenodo record |
| Global monsoon LPS track dataset | https://zenodo.org/records/3890646 | Cross-check of LPS catalogue | Secondary |
| ERA5 reanalysis (850 hPa wind, MSLP, vorticity) | Copernicus CDS, free registration | Physical diagnostics for labels (e.g. monsoon low-level jet strength) | Standard, not re-checked |
| IMDAA 12 km regional reanalysis | NCMRWF — https://rds.ncmrwf.gov.in/ (1979–2018, hourly; NGFS 25 km 1999–2018) | India-specific alternative to ERA5 | Secondary; registration required |

## 5. Static data (role D)

| Dataset | Source | Use | Check / caveat |
|---|---|---|---|
| District boundaries (Survey of India based) | DataMeet `Survey-of-India-Index-Maps/Boundaries/India-Districts-2011Census.shp` — https://github.com/datameet/maps (project page states CC BY 2.5 India; GitHub reports MIT; attribute DataMeet) | District aggregation and map | **Verified** (Phase 6): 641 Census-2011 districts, EPSG:4326, unique census codes, 1 invalid polygon repaired. India now has ~780 districts (e.g. Telangana is still inside Andhra Pradesh). Coverage by IMD land cells: 608 districts ≥ 95%; none for Andaman & Nicobar (3) and Lakshadweep (islands not in the IMD grid); Chennai 4.5%. |
| geoBoundaries ADM2 India | https://data.humdata.org/dataset/geoboundaries-admin-boundaries-for-india | Fallback | International boundary may not match the official Survey of India depiction — **do not use for a MoES-facing map** |
| Elevation (DEM) | e.g. NOAA ETOPO 2022 or SRTM / Copernicus GLO-90 | Orography features (elevation, slope, upslope flow) | To choose in Phase 3 |
| Land–sea mask / coastline | Derived from DEM or Natural Earth | Coastal features (distance to coast) | To choose in Phase 3 |

## 6. Operational thresholds

IMD 24-hour rainfall categories (**Secondary** — confirm against IMD's official terminology document before final release). Stored in `config/thresholds.yaml`, never hard-coded:

| Category | 24 h rainfall |
|---|---|
| Heavy | 64.5 – 115.5 mm |
| Very heavy | 115.6 – 204.4 mm |
| Extremely heavy | ≥ 204.5 mm |

## 7. Alignment rules (critical)

1. **Rain-day window.** IMD rainfall is the 24 h ending 03 UTC. From a 00 UTC forecast, lead day N covers forecast hours (24N−21, 24N+3]: Day 1 = hours 3→27. Because APCP is stored in 6-hour buckets, the total is built from 3-hour increments: Day 1 = [(0–6) − (0–3)] + (6–12) + (12–18) + (18–24) + (24–27).
2. **Date label.** IMD labels each rain day by its END date, so `valid_date = init_date + N`. Evidence: across 40 JJAS-2018 forecasts, GEFS rainfall for 03 UTC D → 03 UTC D+1 matched IMD date D+1 best (mean spatial correlation 0.43 vs 0.34 for D and 0.32 for D+2; D+1 beat D in 30/40 cases). No authoritative IMD statement was found, so this is recorded as an empirical finding.
3. **Grid.** GEFS and IMD 0.25° nodes coincide exactly (6.5–38.5°N, 66.5–100.0°E). Pairing refuses mismatched grids rather than silently regridding.
4. **Only fields available at forecast time go into the model.**

## 8. Practical constraints found during research

- **Disk:** GRIB data is streamed with byte-range requests and cropped to India in memory; only India subsets are stored (APCP for 10 seasons × 3 leads ≈ 250 MB).
- **Download volume:** measured, see §2.1. Pressure-level fields (winds) are 20–50× larger per init than APCP; only selected levels/steps will be fetched (Phase 5).
- **GRIB on Windows:** ecCodes 2.48 installs from pip wheels and works without conda (**Verified**).

## 9. Not available / open requests

| Missing | Impact | Plan |
|---|---|---|
| NCMRWF NCUM/NEPS forecast archive | Cannot report results on NCMRWF's own model | Request via SIH; adapter interface ready |
| Authoritative daily labels for "orographic" and "coastal" rainfall | These are not synoptic regimes with an official catalogue | Derived from terrain + forecast wind, documented as **heuristic** labels |
| Western disturbance catalogue in monsoon season | WDs are mainly a winter phenomenon; rare in JJAS | Grouped into OTHER for the prototype; revisit if scope expands beyond JJAS |
