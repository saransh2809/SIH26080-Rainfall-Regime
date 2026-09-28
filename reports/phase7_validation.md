## Phase 7 heavy-rain probability — validation 2016–2017 (REAL DATA)

### heavy (≥ 64.5 mm) — 51,423 events in 3,633,648 cell-days (base rate 0.0142)

| Metric | CLIM | RAW01 | QM01 | P_blind | P_regime | P_regime_cal |
|---|---|---|---|---|---|---|
| brier | 0.0135 | 0.0181 | 0.0224 | 0.0126 | 0.0125 | 0.0125 |
| bss_vs_climatology | 0.0000 | -0.3430 | -0.6598 | 0.0710 | 0.0769 | 0.0767 |
| roc_auc | 0.7783 | 0.5491 | 0.5901 | 0.8798 | 0.8832 | 0.8825 |
| mean_probability | 0.0144 | 0.0069 | 0.0137 | 0.0140 | 0.0140 | 0.0142 |

Brier improvement P_regime vs P_blind: 0.000080, 95% CI [0.000027, 0.000124] — significant

Brier improvement P_regime_cal vs P_regime: -0.000003, 95% CI [-0.000008, 0.000002] — not significant

Reliability of P_regime_cal (forecast bin → mean forecast / observed frequency / count):

- 0.0–0.1: 0.009 / 0.009 / 3,539,705
- 0.1–0.2: 0.138 / 0.136 / 62,168
- 0.2–0.3: 0.245 / 0.237 / 15,014
- 0.3–0.4: 0.343 / 0.344 / 10,216
- 0.4–0.5: 0.433 / 0.464 / 3,278
- 0.5–0.6: 0.533 / 0.547 / 2,194
- 0.6–0.7: 0.631 / 0.599 / 968
- 0.7–0.8: 0.717 / 0.924 / 105

### very_heavy (≥ 115.6 mm) — 11,036 events in 3,633,648 cell-days (base rate 0.0030)

| Metric | CLIM | RAW01 | QM01 | P_blind | P_regime | P_regime_cal |
|---|---|---|---|---|---|---|
| brier | 0.0030 | 0.0038 | 0.0054 | 0.0029 | 0.0029 | 0.0029 |
| bss_vs_climatology | 0.0000 | -0.2754 | -0.7976 | 0.0323 | 0.0375 | 0.0397 |
| roc_auc | 0.7878 | 0.5249 | 0.5657 | 0.8803 | 0.8860 | 0.8542 |
| mean_probability | 0.0032 | 0.0011 | 0.0031 | 0.0030 | 0.0030 | 0.0031 |

Brier improvement P_regime vs P_blind: 0.000015, 95% CI [-0.000010, 0.000037] — not significant

Brier improvement P_regime_cal vs P_regime: 0.000007, 95% CI [-0.000007, 0.000020] — not significant

Reliability of P_regime_cal (forecast bin → mean forecast / observed frequency / count):

- 0.0–0.1: 0.003 / 0.002 / 3,621,448
- 0.1–0.2: 0.145 / 0.129 / 8,677
- 0.2–0.3: 0.236 / 0.218 / 2,468
- 0.3–0.4: 0.333 / 0.340 / 936
- 0.4–0.5: 0.411 / 0.782 / 119
