---
name: tabular-uncertainty
description: Quantify uncertainty with conformal intervals
---

# Purpose

Turn point predictions into statements a decision-maker can act on.

"Predicted price: $312,000" is far less useful than "predicted $312,000, 90 percent
interval $268,000 to $358,000."

Invoked by `tabular` as step 8, when decisions involve risk.

---

# When to invoke

Always when

- the user asks for confidence, risk, or intervals
- predictions drive consequential decisions (pricing, credit, medical, safety)
- probabilities are used as probabilities rather than rankings
- the user needs to know which predictions to trust

Optional when the task is a ranking or triage problem and only relative order matters.

---

# Regression: split conformal prediction

Conformal prediction gives **distribution-free, finite-sample marginal coverage**.
If you ask for 90 percent, you get roughly 90 percent coverage on exchangeable data,
regardless of the model.

Procedure:

1. Hold out a calibration set the model never trained on
   (`calibration_fraction`, default 0.2).
2. Compute absolute residuals on calibration data.
3. Take the `⌈(n+1)(1−α)⌉/n` quantile of those residuals — call it `q`.
4. Interval for a new point: `prediction ± q`.

```python
import numpy as np


def split_conformal_fit(model, X_cal, y_cal, alpha=0.1):
    resid = np.abs(np.asarray(y_cal) - model.predict(X_cal))
    n = len(resid)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(resid, level, method="higher"))


def split_conformal_predict(model, X_new, q):
    p = model.predict(X_new)
    return p, p - q, p + q


def coverage(y_true, lower, upper):
    inside = (np.asarray(y_true) >= lower) & (np.asarray(y_true) <= upper)
    return {
        "empirical_coverage": float(np.mean(inside)),
        "mean_width": float(np.mean(np.asarray(upper) - np.asarray(lower))),
    }
```

Constant-width intervals are the trade-off of basic split conformal. For
heteroscedastic targets, use a normalized (locally adaptive) variant: fit a second
model on absolute residuals and scale the interval by predicted difficulty. Report
which variant you used.

Always verify empirical coverage on the test set. If you promised 90 percent and
observe 74 percent, exchangeability is violated — usually distribution shift or a
leaky split. Report that rather than the nominal number.

---

# Classification: calibration first

Calibrated probabilities are the primary deliverable.

A model is calibrated when, among predictions of 0.7, roughly 70 percent are
positive.

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# Platt scaling — good for small calibration sets
cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
cal.fit(X_cal, y_cal)

# Isotonic — more flexible, needs ~1000+ calibration rows
cal_iso = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
```

Choose sigmoid (Platt) for small calibration sets; isotonic when you have roughly a
thousand or more calibration rows. Report the Brier score before and after. If
calibration made it worse, keep the uncalibrated model and say so.

---

# Classification: conformal prediction sets

Optional, and genuinely useful for multiclass triage.

Instead of one label, return a **set** of labels guaranteed to contain the truth
with probability `1−α`. A set of size one means the model is confident; a large set
means "route this to a human."

```python
def conformal_sets_fit(proba_cal, y_cal, alpha=0.1):
    """Least-ambiguous-set classifier. Scores are 1 - p(true class)."""
    scores = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def conformal_sets_predict(proba_new, qhat):
    return [np.where(row >= 1 - qhat)[0].tolist() for row in proba_new]
```

Report the average set size. Average size near 1 means a confident model; near the
number of classes means the model is not adding much.

---

# Alternative sources of uncertainty

**CatBoost quantile regression** — fit `loss_function="Quantile:alpha=0.05"` and
`0.95` for a two-model interval. Fast and adaptive, but no coverage guarantee. Can
be conformalized afterward.

**CatBoost with uncertainty** — `RMSEWithUncertainty` gives data uncertainty
estimates directly.

**Bootstrap ensembles** — refit on resamples, use the spread. Expensive; captures
model uncertainty rather than data noise.

Conformal is the default because it comes with a guarantee.

---

# What these intervals do and do not mean

They **do** provide marginal coverage: averaged over all predictions, the stated
fraction contain the truth.

They **do not** provide conditional coverage. Coverage can be far worse for specific
subgroups even when the overall number is met. Check coverage by subgroup before
using intervals for individual high-stakes decisions.

They **assume exchangeability** between calibration and test data. Under temporal
drift, that assumption fails and coverage degrades. For time series, use a
time-ordered calibration set and consider adaptive conformal methods.

They **do not** account for model misspecification beyond what shows up in
calibration residuals, and say nothing about extrapolation far outside training
support.

State these caveats. An interval presented without them invites false confidence.

---

# Output

- method used and the target level (α)
- empirical coverage on test data vs nominal
- mean interval width, in target units
- calibration before and after, with Brier scores
- average prediction set size, for conformal sets
- subgroup coverage where relevant
- explicit caveats above

---

# Do not

Do not compute intervals on data used for training or calibration.

Do not report nominal coverage without checking empirical coverage.

Do not present marginal coverage as conditional coverage.

Do not use conformal prediction under known distribution shift without saying that
the guarantee no longer holds.
