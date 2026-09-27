## Phase 7 heavy-rain probability — validation 2016–2017 (REAL DATA)

### heavy (≥ 64.5 mm) — 51,423 events in 3,633,648 cell-days (base rate 0.0142)

| Metric | CLIM | RAW01 | QM01 | P_blind | P_regime |
|---|---|---|---|---|---|
| brier | 0.0135 | 0.0181 | 0.0232 | 0.0127 | 0.0126 |
| bss_vs_climatology | 0.0000 | -0.3396 | -0.7101 | 0.0618 | 0.0666 |
| roc_auc | 0.7654 | 0.5491 | 0.5959 | 0.8554 | 0.8580 |
| mean_probability | 0.0144 | 0.0069 | 0.0148 | 0.0136 | 0.0135 |

Brier improvement P_regime vs P_blind: 0.000065, 95% CI [0.000004, 0.000121] — significant

Reliability of P_regime (forecast bin → mean forecast / observed frequency / count):

- 0.0–0.1: 0.009 / 0.010 / 3,545,749
- 0.1–0.2: 0.137 / 0.131 / 55,846
- 0.2–0.3: 0.243 / 0.226 / 16,514
- 0.3–0.4: 0.344 / 0.326 / 7,068
- 0.4–0.5: 0.448 / 0.431 / 4,867
- 0.5–0.6: 0.543 / 0.474 / 2,733
- 0.6–0.7: 0.632 / 0.603 / 794
- 0.7–0.8: 0.729 / 0.720 / 75
- 0.8–0.9: 0.819 / 1.000 / 2

### very_heavy (≥ 115.6 mm) — 11,036 events in 3,633,648 cell-days (base rate 0.0030)

| Metric | CLIM | RAW01 | QM01 | P_blind | P_regime |
|---|---|---|---|---|---|
| brier | 0.0030 | 0.0038 | 0.0057 | 0.0029 | 0.0030 |
| bss_vs_climatology | 0.0000 | -0.2708 | -0.8903 | 0.0142 | 0.0133 |
| roc_auc | 0.7511 | 0.5249 | 0.5683 | 0.8651 | 0.8731 |
| mean_probability | 0.0029 | 0.0011 | 0.0035 | 0.0027 | 0.0026 |

Brier improvement P_regime vs P_blind: -0.000002, 95% CI [-0.000018, 0.000013] — not significant

Reliability of P_regime (forecast bin → mean forecast / observed frequency / count):

- 0.0–0.1: 0.002 / 0.002 / 3,618,899
- 0.1–0.2: 0.137 / 0.091 / 7,506
- 0.2–0.3: 0.251 / 0.161 / 3,951
- 0.3–0.4: 0.340 / 0.212 / 2,726
- 0.4–0.5: 0.433 / 0.195 / 508
- 0.5–0.6: 0.539 / 0.180 / 50
- 0.6–0.7: 0.649 / 0.286 / 7
- 0.7–0.8: 0.717 / 0.000 / 1
