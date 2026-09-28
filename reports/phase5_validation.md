## Phase 5 regime classifier — validation 2016–2017 (REAL DATA)

| Metric | R0_rule | RF | LGBM | LGBM_live | CNN | reference_always_NORMAL |
|---|---|---|---|---|---|---|
| accuracy | 0.707 | 0.749 | 0.685 | 0.671 | 0.425 | 0.704 |
| balanced_accuracy | 0.537 | 0.468 | 0.607 | 0.577 | 0.398 | 0.250 |
| macro_f1 | 0.542 | 0.495 | 0.554 | 0.532 | 0.341 | 0.207 |
| NORMAL precision | 0.812 | 0.792 | 0.863 | 0.839 | 0.779 | 0.704 |
| NORMAL recall | 0.817 | 0.931 | 0.736 | 0.732 | 0.438 | 1.000 |
| NORMAL f1 | 0.814 | 0.856 | 0.794 | 0.782 | 0.561 | 0.826 |
| ACTIVE precision | 0.468 | 0.590 | 0.438 | 0.435 | 0.306 | n/a |
| ACTIVE recall | 0.462 | 0.462 | 0.679 | 0.641 | 0.385 | 0.000 |
| ACTIVE f1 | 0.465 | 0.518 | 0.533 | 0.518 | 0.341 | 0.000 |
| BREAK precision | 0.556 | 0.595 | 0.466 | 0.402 | 0.692 | n/a |
| BREAK recall | 0.397 | 0.397 | 0.651 | 0.587 | 0.143 | 0.000 |
| BREAK f1 | 0.463 | 0.476 | 0.543 | 0.477 | 0.237 | 0.000 |
| MONSOON_DEPRESSION precision | 0.386 | 0.286 | 0.329 | 0.352 | 0.139 | n/a |
| MONSOON_DEPRESSION recall | 0.472 | 0.083 | 0.361 | 0.347 | 0.625 | 0.000 |
| MONSOON_DEPRESSION f1 | 0.425 | 0.129 | 0.344 | 0.350 | 0.227 | 0.000 |

Support (validation rows): NORMAL: 507, ACTIVE: 78, BREAK: 63, MONSOON_DEPRESSION: 72

### July–August only (published active/break definition)

| Metric | R0_rule | RF | LGBM | LGBM_live | CNN |
|---|---|---|---|---|---|
| accuracy | 0.661 | 0.664 | 0.664 | 0.659 | 0.468 |
| balanced_accuracy | 0.554 | 0.440 | 0.583 | 0.549 | 0.429 |
| macro_f1 | 0.566 | 0.462 | 0.575 | 0.557 | 0.397 |

### Confusion matrices (rows = true, columns = predicted; order NORMAL, ACTIVE, BREAK, MONSOON_DEPRESSION)

**R0_rule**: `[[414, 36, 19, 38], [28, 36, 0, 14], [36, 0, 25, 2], [32, 5, 1, 34]]`
**RF**: `[[472, 17, 14, 4], [31, 36, 0, 11], [38, 0, 25, 0], [55, 8, 3, 6]]`
**LGBM**: `[[373, 56, 41, 37], [9, 53, 0, 16], [22, 0, 41, 0], [28, 12, 6, 26]]`
**LGBM_live**: `[[371, 56, 51, 29], [11, 50, 0, 17], [26, 0, 37, 0], [34, 9, 4, 25]]`
**CNN**: `[[222, 66, 3, 216], [0, 30, 0, 48], [39, 0, 9, 15], [24, 2, 1, 45]]`
**reference_always_NORMAL**: `[[507, 0, 0, 0], [78, 0, 0, 0], [63, 0, 0, 0], [72, 0, 0, 0]]`
