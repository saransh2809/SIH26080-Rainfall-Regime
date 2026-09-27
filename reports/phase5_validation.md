## Phase 5 regime classifier — validation 2016–2017 (REAL DATA)

| Metric | R0_rule | RF | LGBM | CNN | reference_always_NORMAL |
|---|---|---|---|---|---|
| accuracy | 0.706 | 0.739 | 0.667 | 0.536 | 0.704 |
| balanced_accuracy | 0.523 | 0.454 | 0.563 | 0.587 | 0.250 |
| macro_f1 | 0.535 | 0.491 | 0.522 | 0.466 | 0.207 |
| NORMAL precision | 0.800 | 0.785 | 0.835 | 0.808 | 0.704 |
| NORMAL recall | 0.822 | 0.927 | 0.738 | 0.499 | 1.000 |
| NORMAL f1 | 0.811 | 0.850 | 0.783 | 0.617 | 0.826 |
| ACTIVE precision | 0.422 | 0.571 | 0.430 | 0.354 | n/a |
| ACTIVE recall | 0.487 | 0.308 | 0.513 | 0.859 | 0.000 |
| ACTIVE f1 | 0.452 | 0.400 | 0.468 | 0.502 | 0.000 |
| BREAK precision | 0.571 | 0.619 | 0.448 | 0.282 | n/a |
| BREAK recall | 0.381 | 0.413 | 0.683 | 0.587 | 0.000 |
| BREAK f1 | 0.457 | 0.495 | 0.541 | 0.381 | 0.000 |
| MONSOON_DEPRESSION precision | 0.433 | 0.324 | 0.277 | 0.333 | n/a |
| MONSOON_DEPRESSION recall | 0.403 | 0.167 | 0.319 | 0.403 | 0.000 |
| MONSOON_DEPRESSION f1 | 0.417 | 0.220 | 0.297 | 0.365 | 0.000 |

Support (validation rows): NORMAL: 507, ACTIVE: 78, BREAK: 63, MONSOON_DEPRESSION: 72

### July–August only (published active/break definition)

| Metric | R0_rule | RF | LGBM | CNN |
|---|---|---|---|---|
| accuracy | 0.664 | 0.637 | 0.605 | 0.629 |
| balanced_accuracy | 0.547 | 0.419 | 0.508 | 0.589 |
| macro_f1 | 0.563 | 0.445 | 0.509 | 0.562 |

### Confusion matrices (rows = true, columns = predicted; order NORMAL, ACTIVE, BREAK, MONSOON_DEPRESSION)

**R0_rule**: `[[417, 47, 15, 28], [31, 38, 0, 9], [38, 0, 24, 1], [35, 5, 3, 29]]`
**RF**: `[[470, 14, 14, 9], [38, 24, 0, 16], [37, 0, 26, 0], [54, 4, 2, 12]]`
**LGBM**: `[[374, 47, 48, 38], [17, 40, 0, 21], [19, 0, 43, 1], [38, 6, 5, 23]]`
**CNN**: `[[253, 110, 87, 57], [10, 67, 0, 1], [26, 0, 37, 0], [24, 12, 7, 29]]`
**reference_always_NORMAL**: `[[507, 0, 0, 0], [78, 0, 0, 0], [63, 0, 0, 0], [72, 0, 0, 0]]`
