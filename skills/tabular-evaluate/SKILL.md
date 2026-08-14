---
name: tabular-evaluate
description: Evaluate models honestly on held-out data
---

# Purpose

Produce numbers that will still be true in production.

This skill is the referee. It decides which model wins, and it is responsible for
making sure the comparison was fair.

Invoked by `tabular` as step 7, after every candidate model has been fit.

---

# Non-negotiables

Never report training performance as model performance.

Always evaluate on data the model has not seen.

Every candidate model is compared on the **identical** split. If a model was
evaluated on a different split, refit it or exclude it.

The test set is scored once. If it has informed a modeling decision, it is a
validation set now — say so and stop calling the number an unbiased estimate.

---

# Metric selection

Read from `defaults.yaml` under `metrics`.

**Binary classification**

Primary: ROC AUC. Secondary: average precision, Brier score, log loss, F1.

Use average precision as primary instead when the positive class is rare (below
about 5 percent) — AUC looks flattering on imbalanced data.

Report accuracy only alongside the base rate. Accuracy alone on a 95/5 split is
meaningless.

**Multiclass classification**

Primary: macro F1. Secondary: balanced accuracy, log loss, per-class F1.

Always show the per-class breakdown. An aggregate hides a class the model never
predicts.

**Regression**

Primary: RMSE. Secondary: MAE, R², MAPE.

Report metrics in the target's units. "RMSE 0.34" is unusable; "typical error
$3,400 on a median price of $210,000" is actionable.

Skip MAPE when the target approaches zero — it explodes.

---

# Reference implementation

```python
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, balanced_accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)


def evaluate_binary(y_true, y_proba, threshold=0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "f1": f1_score(y_true, y_pred),
        "base_rate": float(np.mean(y_true)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_regression(y_true, y_pred) -> dict:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_abs_error": float(np.median(np.abs(resid))),
        "target_std": float(np.std(y_true)),
    }


def bootstrap_ci(y_true, y_score, metric_fn, n=1000, alpha=0.05, seed=42):
    """Confidence interval around a metric. A point estimate alone invites overclaiming."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric_fn(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
```

---

# Uncertainty around the estimate

A single number on a 200-row test set is noise dressed as fact.

Always report a confidence interval:

- bootstrap the test set (1,000 resamples) for a holdout evaluation
- report mean ± standard deviation across folds for cross-validation

When comparing two models, check whether the intervals overlap. If they do, say the
difference is not established rather than declaring a winner. Prefer the simpler
model when the difference is not established.

---

# Calibration

For any classifier whose probabilities will be used as probabilities:

- Brier score (lower is better)
- reliability curve — bin predictions into deciles, plot predicted vs observed rate
- expected calibration error

Gradient boosted trees are often overconfident. Logistic regression is usually well
calibrated. If calibration matters and the model is off, route to
`tabular-uncertainty`.

---

# Threshold selection

Default 0.5 is rarely the right operating point.

Choose the threshold from the cost structure the user describes: cost of a false
positive vs a false negative. Absent stated costs, report performance at several
thresholds and let the user pick.

Select the threshold on validation data, never on test.

---

# Subgroup performance

Aggregate metrics hide failures on specific segments.

Break performance down by any grouping the user cares about, plus any high-signal
categorical from `tabular-inspect`. Flag subgroups where performance drops
substantially, and note subgroups too small to evaluate reliably.

---

# Comparison table

Present candidates in one table, sorted by primary metric:

| Model | Primary | 95% CI | Secondary | Fit time | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline | ... | ... | ... | ... | interpretable |
| CatBoost | ... | ... | ... | ... | native categoricals |
| TabICL | ... | ... | ... | ... | no tuning |
| Ensemble | ... | ... | ... | ... | adopted / rejected |

Always include the dummy reference row. If the best model barely beats it, that is
the headline finding.

---

# Output

- metric definitions and why each was chosen
- comparison table with intervals
- calibration assessment
- subgroup breakdown
- recommended model, with the reasoning
- explicit statement of what the test set represents
- honest limitations: sample size, split assumptions, drift risk

---

# Do not

Do not report a metric without context (base rate, target scale, or a reference).

Do not declare a winner on overlapping confidence intervals.

Do not tune a threshold on the test set.

Do not omit the dummy comparison.

Never claim certainty unsupported by evaluation.
