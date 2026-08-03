# Example: Subscription churn — imbalanced, temporal, grouped

The realistic one. Every trap in `references/pitfalls.md` shows up here: leakage,
temporal ordering, repeated customers, class imbalance, and asymmetric costs.

**Dataset:** 84,213 monthly customer-month records, 31 features, 12 months of
history. Target `churned_next_30d`.

---

## 1. Inspect

```
rows: 84213   cols: 32   task: binary
class_shares: {0: 0.9412, 1: 0.0588}   imbalanced: True
minority_share: 0.0588  (4,952 churn events)

customer_id     identifier    18,304 unique  <- GROUP STRUCTURE
snapshot_month  datetime      2025-07 .. 2026-06  <- TEMPORAL STRUCTURE

tenure_months, monthly_charges, total_charges, support_tickets_30d,
logins_30d, features_used, seats, plan_tier, payment_method,
contract_type, region, ... (usable)

cancellation_reason   high_card   missing 94.1%   <- LEAKAGE
refund_issued         numeric     missing 94.6%   <- LEAKAGE
churn_date            datetime    missing 94.1%   <- LEAKAGE
csat_score            numeric     missing 71.2%
duplicate_rows: 0
```

### Leakage: stop and ask

Three columns are missing at exactly the churn rate. `cancellation_reason` is
populated only for customers who cancelled — it is the outcome wearing a costume.
Any model using it scores AUC 0.99 and is worthless.

**Dropped:** `cancellation_reason`, `refund_issued`, `churn_date`.

Also verified `total_charges` is cumulative *to the snapshot date*, not lifetime
final. Confirmed with the user before keeping it — this is the kind of question the
agent must ask rather than assume.

`csat_score` at 71 percent missing is kept with a missingness indicator; surveys are
sent to a non-random subset, so the missingness itself carries signal.

### Structure

Both group and temporal structure exist. Per the split rules, time wins, then verify
groups do not straddle the boundary.

---

## 2. Split

```
strategy: temporal, group-verified
train:  2025-07 .. 2026-03   58,891 rows   14,112 customers
valid:  2026-04 .. 2026-04    8,447 rows
test:   2026-05 .. 2026-06   16,875 rows    9,338 customers

boundary: 2026-05-01
straddling customers: 6,204 appear in both train and test
```

Customers legitimately persist across months — that is the nature of the data. What
matters is that no *future* information about a customer reaches training. Because
the cut is temporal and all features are computed as-of the snapshot, a customer
appearing in both periods is fine; a customer's *future rows* leaking into training
is not, and the time cut prevents that.

Verified: `train[snapshot_month].max() < test[snapshot_month].min()`. ✓

```
churn rate — train 5.91%, valid 5.72%, test 6.03%
minority in test: 1,017 events — plenty
```

Rate is stable across the boundary, so no obvious concept drift within the window.

**Not done:** random splitting (would leak the future), and SMOTE (would place
synthetic copies of test-period customers into training).

---

## 3. Baseline

Binary + imbalanced → `LogisticRegression(class_weight="balanced")`.

High-cardinality `region` (94 levels) dropped for the baseline; CatBoost will keep it.

```
best C: 0.1
validation ROC AUC: 0.7913
validation avg precision: 0.2087
dummy AP (= base rate): 0.0572
```

Top odds ratios:

| Feature | OR | Reading |
| --- | --- | --- |
| support_tickets_30d | 1.61 | each SD of tickets raises churn odds 61% |
| contract_type=monthly | 2.34 | monthly contracts churn far more than annual |
| logins_30d | 0.58 | engagement is protective |
| tenure_months | 0.71 | longer tenure, lower churn |
| csat_missing | 1.27 | never surveyed, higher churn |

Immediately actionable and readable by a retention team. Note the standing caveat:
associational. Suppressing support tickets will not reduce churn.

---

## 4. CatBoost

`auto_class_weights="Balanced"`, `eval_metric="AUC"`, categoricals native
(including `region`).

```
best_iteration: 623 (early stopping fired)
validation ROC AUC: 0.8447
validation avg precision: 0.3129
```

Importance: `support_tickets_30d` 18%, `logins_30d` 16%, `contract_type` 14%,
`tenure_months` 11%, `monthly_charges` 9%, `csat_score` 7%.

+0.05 AUC and **+50 percent average precision** over the baseline. On imbalanced
data, AP is the number that matters.

---

## 5. TabICL

58,891 training rows — within the ceiling but near the upper end.

```
context_rows: 58891
device: cuda
fit + predict: 47s
validation ROC AUC: 0.8298
validation avg precision: 0.2874
```

Respectable with zero tuning, but below CatBoost. Expected: this is the larger-data,
mixed-type regime where CatBoost is strongest.

---

## 6. Evaluate

Untouched test set — 16,875 rows from 2026-05 and 2026-06.

Primary metric is **average precision**, not AUC, because positives are 6 percent.

| Model | AP | 95% CI | ROC AUC | Brier | Fit |
| --- | --- | --- | --- | --- | --- |
| Dummy | 0.0603 | — | 0.500 | 0.0567 | 0s |
| Baseline | 0.1974 | [0.1801, 0.2156] | 0.7844 | 0.0489 | 2s |
| CatBoost | 0.3038 | [0.2841, 0.3241] | 0.8391 | 0.0451 | 34s |
| TabICL | 0.2801 | [0.2607, 0.3002] | 0.8256 | 0.0463 | 47s |

CatBoost vs baseline: intervals well separated, real difference. CatBoost vs TabICL:
slight overlap, CatBoost preferred.

**Threshold.** The user supplied costs: a retention offer costs $40; a lost customer
costs $890 in lifetime value. False negatives are ~22× more expensive than false
positives, so the operating point should be well below 0.5.

Optimized on validation (never test):

```
threshold: 0.11
test precision: 0.241   recall: 0.622   F1: 0.347
predicted churners: 2,624 of 16,875 (15.6% of base)
expected net value: +$318,400 over the two-month test window
```

At this threshold the team contacts 15.6 percent of customers and catches 62 percent
of churn.

**Subgroup breakdown**

| Segment | AP | n | Note |
| --- | --- | --- | --- |
| Enterprise | 0.412 | 1,204 | strongest |
| Mid-market | 0.318 | 5,891 | |
| SMB | 0.267 | 9,780 | weakest, largest |
| Tenure < 3mo | 0.189 | 2,103 | little history to learn from |
| Region APAC | 0.221 | 1,432 | underrepresented in training |

New customers are hardest — unsurprising, and worth a separate model or a rules-based
fallback for the first 90 days.

---

## 7. Ensemble

Correlations: baseline~CatBoost 0.87, CatBoost~TabICL 0.93.

```
rank average AP: 0.3061
best single (CatBoost): 0.3038
relative gain: 0.0076
threshold: 0.005
```

Clears the threshold. **Adopted?** No — recommended against, and here is the honest
reasoning: the gain is 0.0023 AP in absolute terms, well inside the bootstrap
interval, while requiring GPU-backed TabICL inference in a nightly scoring job.
Threshold cleared on a technicality; the confidence intervals say the difference is
not established.

Recorded as: *ensemble marginally better, not adopted, revisit if inference cost
stops mattering.*

---

## 8. Uncertainty

Probabilities feed a dollar-denominated decision, so calibration is essential.

```
CatBoost raw:              Brier 0.0451   overconfident at high end
Platt-scaled (isotonic on 8,447 valid rows):
                           Brier 0.0429   well calibrated
```

Reliability by decile, after isotonic calibration:

| Predicted bin | Mean predicted | Observed rate | n |
| --- | --- | --- | --- |
| 0.00–0.05 | 0.021 | 0.019 | 11,204 |
| 0.05–0.15 | 0.094 | 0.101 | 3,118 |
| 0.15–0.30 | 0.213 | 0.226 | 1,614 |
| 0.30–0.60 | 0.407 | 0.389 | 741 |
| 0.60–1.00 | 0.724 | 0.688 | 198 |

Close along the diagonal. Now "18 percent churn risk" genuinely means 18 percent,
which is what makes the $40-vs-$890 expected-value calculation valid.

---

## 9. Recommendation

**Ship isotonic-calibrated CatBoost at threshold 0.11.**

- Test AP 0.304 versus 0.060 base rate — 5× lift over random.
- Catches 62 percent of churn while contacting 16 percent of the base.
- Estimated +$318,400 net value per two-month cycle at the stated costs.
- Calibrated probabilities suitable for expected-value decisions.

**Assumptions and limits**

1. Evaluated on the two months immediately following training. Performance will
   decay; monitor monthly.
2. Temporal split assumes future months resemble 2026-05/06. A pricing change,
   product launch, or competitor entry breaks that.
3. Weakest on customers under 3 months tenure (AP 0.189) — consider a separate model.
4. APAC underrepresented; treat those scores with caution.
5. Feature importance is associational. The retention team should A/B test
   interventions rather than assume causality.
6. Three leakage columns were dropped. If a future pipeline re-adds them, results
   will look spectacular and be wrong.

**Retraining trigger:** monthly refit; alert if observed AP on delayed labels falls
below 0.24, or if the monthly churn rate moves outside 4.5–7.5 percent.

---

## 10. Checkpoint

```
artifacts/models/churn_catboost_cal_20260728_161955/
├── model.joblib          # pipeline incl. isotonic calibrator
├── metadata.json         # temporal split boundary, dropped leakage cols
├── feature_names.json    # 28 features, exact order
├── metrics.json          # full eval incl. subgroups and threshold sweep
└── conformal.json        # calibration mapping
```

`metadata.json` explicitly records the three dropped leakage columns and why —
so the next person to touch this pipeline does not "helpfully" add them back.
