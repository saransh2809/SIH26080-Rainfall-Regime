## Phase 7 heavy-rain probability — validation 2016–2017 (REAL DATA)

### heavy (≥ 64.5 mm) — 51,423 events in 3,633,648 cell-days (base rate 0.0142)

| Metric | CLIM | RAW01 | QM01 | P_blind | P_regime | P_regime_cal |
|---|---|---|---|---|---|---|
| brier | 0.0135 | 0.0181 | 0.0232 | 0.0127 | 0.0126 | 0.0126 |
| bss_vs_climatology | 0.0000 | -0.3396 | -0.7101 | 0.0618 | 0.0666 | 0.0664 |
| roc_auc | 0.7654 | 0.5491 | 0.5959 | 0.8554 | 0.8580 | 0.8479 |
| mean_probability | 0.0144 | 0.0069 | 0.0148 | 0.0136 | 0.0135 | 0.0141 |

Brier improvement P_regime vs P_blind: 0.000065, 95% CI [0.000004, 0.000121] — significant

Brier improvement P_regime_cal vs P_regime: -0.000002, 95% CI [-0.000017, 0.000011] — not significant

Reliability of P_regime_cal (forecast bin → mean forecast / observed frequency / count):

- 0.0–0.1: 0.009 / 0.009 / 3,540,127
- 0.1–0.2: 0.133 / 0.123 / 56,493
- 0.2–0.3: 0.234 / 0.215 / 21,485
- 0.3–0.4: 0.340 / 0.341 / 8,691
- 0.4–0.5: 0.455 / 0.456 / 5,858
- 0.5–0.6: 0.556 / 0.587 / 842
- 0.6–0.7: 0.650 / 0.627 / 75
- 0.7–0.8: 0.721 / 0.720 / 75
- 0.8–0.9: 0.891 / 1.000 / 1
- 0.9–1.0: 1.000 / 1.000 / 1

### very_heavy (≥ 115.6 mm) — 11,036 events in 3,633,648 cell-days (base rate 0.0030)

| Metric | CLIM | RAW01 | QM01 | P_blind | P_regime | P_regime_cal |
|---|---|---|---|---|---|---|
| brier | 0.0030 | 0.0038 | 0.0057 | 0.0029 | 0.0030 | 0.0029 |
| bss_vs_climatology | 0.0000 | -0.2708 | -0.8903 | 0.0142 | 0.0133 | 0.0227 |
| roc_auc | 0.7511 | 0.5249 | 0.5683 | 0.8651 | 0.8731 | 0.7818 |
| mean_probability | 0.0029 | 0.0011 | 0.0035 | 0.0027 | 0.0026 | 0.0029 |

Brier improvement P_regime vs P_blind: -0.000002, 95% CI [-0.000018, 0.000013] — not significant

Brier improvement P_regime_cal vs P_regime: 0.000028, 95% CI [0.000014, 0.000042] — significant

Reliability of P_regime_cal (forecast bin → mean forecast / observed frequency / count):

- 0.0–0.1: 0.002 / 0.003 / 3,621,723
- 0.1–0.2: 0.127 / 0.101 / 5,659
- 0.2–0.3: 0.241 / 0.192 / 6,266
