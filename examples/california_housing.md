# Example: California Housing — medium regression with intervals

All-numeric regression at medium scale, where the deliverable is a price estimate
with an honest interval attached.

**Dataset:** 20,640 rows, 8 numeric features. Target `MedHouseVal`, median house
value per block group in $100,000s.

---

## 1. Inspect

```
rows: 20640   cols: 9   task: regression
target: MedHouseVal   min 0.15   median 1.797   max 5.00001   std 1.154

MedInc      numeric   median income
HouseAge    numeric
AveRooms    numeric   max 141.9   <- extreme outliers
AveBedrms   numeric   max 34.1    <- extreme outliers
Population  numeric
AveOccup    numeric   max 1243.3  <- extreme outliers
Latitude    numeric
Longitude   numeric

missing: none
duplicate_rows: 0
leakage_suspects: none
```

Two things to flag:

**Target is censored at 5.00001.** 965 block groups (4.7 percent) sit exactly at the
cap. The model cannot predict above it and errors there are structurally biased.
This must appear in the final report.

**Ratio features have extreme tails.** `AveOccup` of 1243 is a data artifact, not a
neighborhood. Do not clip silently — flag them and let the tree model handle them.

Latitude/Longitude present spatial structure. Random splitting means the model can
interpolate between neighboring block groups — geographic interpolation, not
generalization to new regions. State the assumption; if the deployment target is
*new* regions, split spatially instead.

---

## 2. Split

`tabular-split` → regression, no group or time column → random split.

20,640 rows is above `cv_below_rows`, so a single holdout is adequate, but we keep
5-fold CV on the training portion for model selection.

```
train: 13209   validation: 3303   test: 4128
random_state: 42
```

Three-way, because CatBoost needs a validation set for early stopping and
`tabular-uncertainty` needs a calibration set that never touched training.

---

## 3. Baseline

Regression → `Ridge`.

```
best alpha: 1.0
CV RMSE: 0.7284 ± 0.0121   ($72,840)
dummy (median) RMSE: 1.1540 ($115,400)
R²: 0.602
```

Top standardized coefficients: `MedInc` +0.83, `Latitude` −0.89, `Longitude` −0.87,
`AveRooms` −0.29.

Latitude and Longitude carrying large opposite-signed coefficients is a linear model
straining to express "coastal California is expensive." A tree model will do this far
better — a good prior for what comes next.

---

## 4. CatBoost

```
iterations requested: 2000
best_iteration: 1436  (early stopping fired)
validation RMSE: 0.4519  ($45,190)
```

Importance: `MedInc` 41%, `Longitude` 16%, `Latitude` 15%, `AveOccup` 11%,
`HouseAge` 7%.

A 38 percent RMSE reduction against the baseline. Interactions and spatial
non-linearity are doing real work here — this is the case where the complex model
clearly earns its keep.

---

## 5. TabICL

20,640 rows is within scale limits but on the larger side for CPU.

```
context_rows: 13209
device: cpu
fit + predict: 412s
validation RMSE: 0.4783  ($47,830)
```

Competitive with zero tuning, but slower than CatBoost and slightly worse. On a GPU
this would run in seconds; on CPU the cost is real.

---

## 6. Evaluate

Untouched 4,128-row test set:

| Model | RMSE | 95% CI | MAE | R² | Median abs err |
| --- | --- | --- | --- | --- | --- |
| Dummy (median) | 1.1540 | — | 0.9122 | 0.000 | 0.7810 |
| Ridge | 0.7301 | [0.7089, 0.7518] | 0.5324 | 0.599 | 0.4187 |
| CatBoost | 0.4497 | [0.4302, 0.4698] | 0.2951 | 0.848 | 0.1962 |
| TabICL | 0.4771 | [0.4566, 0.4983] | 0.3143 | 0.829 | 0.2104 |

In target units: typical CatBoost error ≈ **$44,970**, median absolute error ≈
**$19,620**, against a median home value of $179,700.

Baseline and CatBoost intervals are far apart — this difference is real, unlike the
Titanic case. CatBoost vs TabICL intervals barely overlap; CatBoost is preferred on
both accuracy and cost.

**Subgroup check:** RMSE on capped-value blocks (target = 5.00001) is 0.9124, double
the overall figure. Expected, given censoring, and worth stating plainly.

---

## 7. Ensemble

CatBoost~TabICL prediction correlation: 0.974. High.

```
weighted average (0.62 / 0.38) RMSE: 0.4471
best single (CatBoost):             0.4497
relative gain: 0.0058
threshold: 0.005
```

Clears the bar, barely. **Adopted with a caveat:** the gain is $260 of typical error
and requires running TabICL at scoring time — 412s on CPU versus 0.02s for CatBoost
alone.

Recommend CatBoost alone unless batch scoring is infrequent and accuracy is
paramount. Documented, decision handed to the user.

---

## 8. Uncertainty

Split conformal on the 3,303-row calibration set, α = 0.1.

```
conformal quantile q: 0.7218
nominal coverage:   90%
empirical coverage: 89.7%   <- guarantee holding
mean interval width: 1.4436  ($144,360)
```

Example: predicted $312,000, 90 percent interval **$240,000 to $384,000**.

The interval is wide because constant-width split conformal must accommodate the
hardest predictions. Locally adaptive (normalized) conformal narrows intervals for
easy cases:

```
normalized conformal, empirical coverage: 90.2%
mean width: 1.2914 ($129,140)
width on easiest quintile:   $71,200
width on hardest quintile:  $238,600
```

Prefer the normalized variant here — same coverage, materially more informative.

**Subgroup coverage:** 91.2 percent on non-capped blocks, 68.4 percent on capped
blocks. Marginal coverage holds overall while failing badly on a known subgroup —
the textbook illustration of why marginal is not conditional coverage. Do not use
these intervals for individual high-value properties near the cap.

---

## 9. Recommendation

**Ship CatBoost with normalized conformal intervals.**

- Typical error $44,970; median absolute error $19,620.
- 90 percent intervals with verified 90.2 percent empirical coverage.
- 38 percent RMSE improvement over the linear baseline, statistically clear.

**Stated limitations**

1. Target censored at $500,001; predictions near the cap are biased low and interval
   coverage there is only 68 percent.
2. Random split assumes deployment on the same geography. New regions require a
   spatial split to estimate honestly.
3. 1990 census data; coefficients and levels do not transfer to current prices.
4. Block-group aggregates — do not apply to individual homes.

---

## 10. Checkpoint

```
artifacts/models/cahousing_catboost_20260728_151140/
├── model.joblib
├── metadata.json
├── feature_names.json
├── metrics.json
└── conformal.json      # normalized conformal scaler + quantile
```
