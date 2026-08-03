# Example: Titanic — small binary classification

Small, mixed-type, missing values, interpretability matters. The canonical case for
"run all three models and compare."

**Dataset:** ~891 rows, 11 features. Target `Survived` (0/1).

---

## 1. Inspect

`tabular-inspect` reports:

```
rows: 891   cols: 12   task: binary
class_shares: {0: 0.6162, 1: 0.3838}   imbalanced: False

Age         numeric      missing 19.9%
Cabin       high_card    missing 77.1%   687 unique
Embarked    categorical  missing 0.2%    3 unique
Ticket      identifier   681 unique
Name        text         891 unique
PassengerId identifier   891 unique
Sex, Pclass, SibSp, Parch, Fare  usable

duplicate_rows: 0
no temporal structure, no group structure
leakage_suspects: none
```

Decisions:

- Drop `PassengerId`, `Ticket` — identifiers.
- Drop `Name` — text, out of scope for v1. (Title extraction is a known winner here;
  note it as a future feature-engineering step rather than doing it silently.)
- Keep `Cabin` as a derived `has_cabin` flag; 77 percent missing is itself signal
  (deck recorded mostly for first class).
- 891 rows is **very small** → cross-validation, not a single holdout. Compare all
  three model families.

---

## 2. Split

`tabular-split` → stratified, no group or time structure.

Below `cv_below_rows` (5,000), so `StratifiedKFold(n_splits=10, shuffle=True,
random_state=42)`, plus a 20 percent stratified holdout touched once at the end.

```
train: 712    test: 179
class balance preserved: train 38.3% positive, test 38.5% positive
minority class in test: 69 rows — above the 10-row floor
```

---

## 3. Baseline

`tabular-baseline` → binary task → `LogisticRegression`.

Preprocessing: median impute + standardize numerics; most-frequent impute + one-hot
for `Sex`, `Embarked`, `Pclass`. `Cabin` dropped, replaced by `has_cabin`.

```
best C: 1.0
CV ROC AUC: 0.857 ± 0.038
dummy (prior) AUC: 0.500
```

Top standardized coefficients:

| Feature | Coef | Odds ratio | Reading |
| --- | --- | --- | --- |
| Sex=male | −1.27 | 0.28 | men ~72% lower odds of survival |
| Pclass=3 | −0.71 | 0.49 | third class roughly halves the odds |
| Age | −0.42 | 0.66 | each SD of age lowers odds ~34% |
| has_cabin | +0.31 | 1.36 | recorded cabin associated with survival |
| SibSp | −0.28 | 0.76 | larger sibling/spouse count, lower odds |

Associational, not causal — `has_cabin` is largely a proxy for class.

---

## 4. CatBoost

`tabular-catboost`, categoricals passed natively (`Sex`, `Embarked`, `Pclass`).

```
iterations requested: 2000
best_iteration: 187  (early stopping fired — good)
CV ROC AUC: 0.874 ± 0.035
```

Importance: `Sex` 34%, `Fare` 19%, `Age` 16%, `Pclass` 14%, `SibSp` 7%.

Gain over baseline: +0.017 AUC. Confidence intervals overlap substantially.

---

## 5. TabICL

891 rows, 8 features — comfortably inside the sweet spot, and exactly the regime
where a foundation model shines.

```
context_rows: 712
device: cpu
fit + predict: 24s
CV ROC AUC: 0.881 ± 0.033
```

Best single CV score, marginally.

---

## 6. Evaluate

`tabular-evaluate` on the untouched 179-row holdout:

| Model | ROC AUC | 95% CI | AP | Brier | Fit time |
| --- | --- | --- | --- | --- | --- |
| Dummy | 0.500 | — | 0.385 | 0.236 | 0s |
| Baseline | 0.851 | [0.789, 0.905] | 0.821 | 0.147 | 0.4s |
| CatBoost | 0.869 | [0.812, 0.919] | 0.845 | 0.139 | 3.1s |
| TabICL | 0.876 | [0.820, 0.924] | 0.851 | 0.135 | 24s |

**All three intervals overlap heavily.** On 179 test rows, an AUC spread of 0.025
is not a real difference.

Calibration: baseline Brier 0.147, well calibrated. CatBoost slightly overconfident
at the extremes.

---

## 7. Ensemble

Prediction correlations: baseline~CatBoost 0.91, CatBoost~TabICL 0.94,
baseline~TabICL 0.89. Decent diversity.

```
simple average AUC: 0.878
best single (TabICL): 0.876
relative gain: 0.0023
threshold: 0.005
```

**Rejected.** Below the adoption threshold. Not worth running three models at
scoring time for 0.002 AUC.

---

## 8. Recommendation

**Ship the baseline.**

The reasoning: on 891 rows, no model is statistically distinguishable from logistic
regression. The baseline is 60× faster, produces odds ratios a stakeholder can read,
is the best calibrated of the three, and has no checkpoint-download dependency.

Choose CatBoost instead if raw accuracy is the only objective and the ~0.02 AUC —
unproven at this sample size — is worth the interpretability loss.

Report verbatim: *the more complex models did not demonstrably help.* That is a
finding, not a failure, and it is exactly what the baseline exists to establish.

---

## 9. Checkpoint

```
artifacts/models/titanic_logreg_20260728_143022/
├── model.joblib
├── metadata.json
├── feature_names.json
└── metrics.json
```

Metadata records the 10-fold stratified split, seed 42, sklearn version, and the
dropped-column decisions above.
