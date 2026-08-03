# Tabular modeling pitfalls

The failure modes that actually ruin tabular projects, in rough order of how often
they do real damage. Every skill in this pack references this file.

Most of these produce **excellent** validation numbers. That is what makes them
dangerous — the model looks great right up until it reaches production.

---

## 1. Target leakage

A feature contains information unavailable at prediction time.

**How it shows up:** implausibly high performance. AUC of 0.99 on a messy business
problem is a leak until proven otherwise.

**Common sources**

- Post-outcome fields: `cancellation_reason`, `refund_amount`, `resolved_at`,
  `discharge_code`. Populated only *because* the outcome happened.
- Aggregates computed over the full dataset, including the future.
- Identifiers that encode the target — a case number issued only to churned accounts.
- Timestamps whose presence implies the outcome.
- Preprocessing fit on all data before splitting (see #2).

**Detection:** correlations above 0.98 with a numeric target. A single categorical
that perfectly separates classes. Drop the top feature and see if performance
collapses — real signal is usually distributed.

**Fix:** for every feature ask, "would this value be known, in this form, at the
moment I need the prediction?" If no, drop it.

---

## 2. Preprocessing before splitting

Fitting a scaler, imputer, or encoder on the whole dataset leaks test statistics
into training.

**Wrong**

```python
X_scaled = StandardScaler().fit_transform(X)   # sees the test set
X_train, X_test = train_test_split(X_scaled)
```

**Right**

```python
X_train, X_test = train_test_split(X)
pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
pipe.fit(X_train, y_train)   # scaler fit on train only, inside CV folds too
```

The effect is small for a mean and standard deviation, and large for target
encoding or feature selection. Use pipelines. Always.

---

## 3. Temporal leakage

Random splitting time-ordered data trains on the future to predict the past.

Validation looks great. Production does not, because in production the future is
genuinely unavailable.

**Fix:** sort by time, train on earlier, test on later. Use `TimeSeriesSplit`.
Verify `train[time].max() < test[time].min()`.

Watch for subtler versions: rolling features with a window that peeks forward, a
"days since last purchase" computed against a fixed date after the test period, or
records back-dated on update.

---

## 4. Group leakage

The same entity appears in both train and test. The model memorizes the entity
rather than learning the pattern.

Common with repeated measures: multiple visits per patient, sessions per user,
readings per device, transactions per account.

**Fix:** `GroupKFold` / `GroupShuffleSplit` on the entity id. Verify zero overlap.
Duplicate rows are the degenerate case — deduplicate before splitting.

---

## 5. Resampling before splitting

SMOTE or oversampling applied to the whole dataset places synthetic copies derived
from test rows into training.

**Fix:** resample inside the training fold only, using `imblearn.pipeline.Pipeline`
so it participates correctly in cross-validation. Better still: use class weights
(`auto_class_weights="Balanced"` in CatBoost) and skip resampling. It usually helps
less than people expect once metrics are chosen sensibly.

---

## 6. The wrong metric

Accuracy on imbalanced data is the classic. 95 percent accuracy with a 95 percent
majority class means the model learned to say "no."

- Rare positives → average precision over ROC AUC.
- Probabilities used as probabilities → Brier score, log loss, calibration curve.
- Asymmetric costs → threshold chosen from the cost matrix, not 0.5.
- Regression with outliers → MAE alongside RMSE; RMSE is dominated by tails.
- Target near zero → never MAPE.

Always report against a reference: base rate for classification, target standard
deviation or a median predictor for regression.

---

## 7. Test set contamination through repeated use

Every peek at the test set is a small fit. Twenty models compared on the same test
set, and the winner's score is optimistically biased by selection.

**Fix:** validation set (or CV) for all decisions; test set touched once, at the
end. If you must re-use it, say so and stop calling the estimate unbiased.

---

## 8. Overinterpreting a small test set

On 200 test rows, a 2-point AUC difference is noise.

**Fix:** bootstrap confidence intervals. Compare intervals, not point estimates. If
they overlap, the difference is not established — prefer the simpler model. Use
cross-validation instead of a single holdout below a few thousand rows.

---

## 9. Miscalibrated probabilities

Gradient boosted trees are typically overconfident: predictions cluster near 0 and 1.
Fine for ranking, wrong for anything where 0.8 must mean "80 percent of the time."

**Detection:** reliability curve, Brier score.

**Fix:** Platt scaling (small calibration sets) or isotonic regression (larger ones),
fit on held-out calibration data. See `tabular-uncertainty`.

---

## 10. Ignoring distribution shift

The model was fit on last year. The world moved.

Covariate shift (features change), label shift (base rate changes), and concept
shift (the relationship itself changes) all degrade performance, and only the last
is visible without labels.

**Fix:** monitor input distributions, keep a delayed-label evaluation loop, and set
retraining triggers up front. Conformal coverage guarantees assume exchangeability
and quietly fail under drift.

---

## 11. High-cardinality categoricals handled badly

One-hot encoding a 10,000-level column produces a sparse mess. Naive target encoding
on the full dataset leaks.

**Fix:** let CatBoost handle them natively with ordered target statistics. If you must
encode manually, fit the encoding inside the CV fold. Group rare levels into
`__other__` with a minimum frequency.

---

## 12. Silent dtype and schema drift

A column that was numeric in training arrives as a string in production, or column
order changes and a positional model mismatches everything.

**Fix:** persist the feature schema with the model and validate on every scoring
call. See `tabular-checkpoint`.

---

## 13. Missingness treated as noise

Missing values are often informative — the field is blank *because* of something.
Imputing to the median destroys that signal.

**Fix:** add a missingness indicator when the pattern looks structural. CatBoost
handles `NaN` natively; let it. For categoricals, an explicit `"__missing__"` level
usually beats mode imputation.

Also check whether missingness differs between train and test — that is a shift
signal.

---

## 14. Duplicate rows

Exact duplicates inflate performance when they straddle the split, and quietly
reweight the training distribution when they do not.

**Fix:** count them during inspection. Deduplicate before splitting unless duplicates
are genuinely meaningful (legitimate repeat events).

---

## 15. Tuning into the validation set

Hundreds of hyperparameter trials scored on one validation split will find the
configuration that best fits that split's noise.

**Fix:** modest search spaces, nested cross-validation when the budget allows, and a
final untouched test set. This pack deliberately keeps tuning small for that reason.

---

## 16. Confusing importance with causality

Feature importance says the model used a feature. It says nothing about what happens
if you intervene on it.

"Number of support tickets is the top churn predictor" does not mean suppressing
tickets reduces churn. Correlated features also split credit arbitrarily — two
near-duplicates each look half as important.

**Fix:** state importance as associational. Reserve causal claims for designs that
support them.

---

## 17. Not having a baseline

Without a linear/logistic baseline and a dummy reference, there is no way to know
whether the complex model earned its complexity.

This is why `tabular-baseline` runs on every task, even when a stronger model is
expected to win. It is the cheapest insurance in the pack.

---

## Quick pre-flight checklist

- [ ] Every feature available at prediction time?
- [ ] Preprocessing inside the pipeline, fit on train only?
- [ ] Split respects time ordering?
- [ ] Split respects group structure?
- [ ] Duplicates checked?
- [ ] Metric appropriate to task, balance, and cost structure?
- [ ] Metric reported against a base rate or reference?
- [ ] Confidence interval around the headline number?
- [ ] Test set used exactly once?
- [ ] Calibration checked, if probabilities matter?
- [ ] Baseline and dummy both beaten?
- [ ] Feature schema persisted with the model?
