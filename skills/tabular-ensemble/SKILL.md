---
name: tabular-ensemble
description: Combine baseline, CatBoost, and TabICL predictions through simple averaging, rank averaging, or shallow stacking. Only adopt the ensemble when held-out evaluation demonstrates a real improvement over the best single model.
---

# Purpose

Combine models that make different mistakes.

Ensembling is frequently the best final performer on tabular data — but only when
members are genuinely diverse and the gain is measured honestly.

Invoked by `tabular` last, and only after individual models have been evaluated.

---

# The adoption rule

An ensemble is adopted **only** if it beats the best single model by at least
`min_relative_improvement` (default 0.005, i.e. 0.5 percent relative) on the primary
metric, measured on data neither the members nor the combiner used for fitting.

If it does not clear that bar, ship the best single model. A 0.1 percent gain is not
worth doubling inference cost and halving explainability.

State the comparison explicitly either way.

---

# Methods, in order of preference

**Simple averaging** — mean of predicted probabilities (classification) or values
(regression). No parameters, nothing to overfit. Start here.

**Rank averaging** — average the rank-transformed predictions. Use when members are
on different scales or poorly calibrated relative to each other, and the metric is
ranking-based (AUC).

**Weighted averaging** — weights proportional to validation performance, or found by
constrained optimization on validation. One step more complex; needs a real gain.

**Stacking** — a shallow meta-learner (logistic or ridge regression) on out-of-fold
predictions. The strongest option and the easiest to leak with. Only use with proper
out-of-fold generation.

Do not build large model stacks. Cap members at `max_members` (default 3).

---

# Diversity check

Averaging two models that agree everywhere gains nothing.

Before ensembling, compute the correlation between member predictions on the
validation set.

- correlation above 0.98 → skip; they are the same model wearing different hats
- correlation 0.85 to 0.98 → averaging may give a small gain
- correlation below 0.85 → good ensemble candidates

Baseline and CatBoost are usually well decorrelated. CatBoost and TabICL are
moderately correlated but often complementary — different inductive biases.

---

# Reference implementation

```python
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_predict


def simple_average(preds: dict) -> np.ndarray:
    return np.mean(np.column_stack(list(preds.values())), axis=1)


def rank_average(preds: dict) -> np.ndarray:
    ranks = [rankdata(p) / len(p) for p in preds.values()]
    return np.mean(np.column_stack(ranks), axis=1)


def weighted_average(preds: dict, scores: dict) -> np.ndarray:
    names = list(preds)
    w = np.array([max(scores[n], 0.0) for n in names], dtype=float)
    if w.sum() == 0:
        w = np.ones(len(names))
    w = w / w.sum()
    return np.column_stack([preds[n] for n in names]) @ w


def diversity(preds: dict) -> dict:
    names = list(preds)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}~{b}"] = float(np.corrcoef(preds[a], preds[b])[0, 1])
    return out


def build_stack(models: dict, X, y, cv, task):
    """Out-of-fold meta-features. Never fit the meta-learner on in-fold predictions."""
    oof = {}
    for name, est in models.items():
        method = "predict_proba" if task in ("binary", "multiclass") else "predict"
        p = cross_val_predict(est, X, y, cv=cv, method=method, n_jobs=-1)
        oof[name] = p[:, 1] if p.ndim > 1 and p.shape[1] == 2 else p
    meta_X = np.column_stack([oof[n] for n in models])
    meta = (LogisticRegression(max_iter=1000)
            if task in ("binary", "multiclass") else RidgeCV())
    meta.fit(meta_X, y)
    return meta, oof


def adopt(ensemble_score, best_single_score, min_rel=0.005, higher_is_better=True):
    if higher_is_better:
        gain = (ensemble_score - best_single_score) / abs(best_single_score)
    else:
        gain = (best_single_score - ensemble_score) / abs(best_single_score)
    return gain >= min_rel, gain
```

---

# Stacking without leaking

The one rule: the meta-learner must **never** see a member prediction made on data
that member was trained on.

Generate meta-features with `cross_val_predict` using the same CV object from
`tabular-split`. Then refit members on the full training set for deployment, keeping
the meta-learner fitted on out-of-fold predictions.

Keep the meta-learner shallow — logistic or ridge regression, regularized. A
gradient boosted meta-learner on three features will memorize the fold structure.

Evaluate the whole stack on the untouched test set.

---

# Calibration

Averaging probabilities from differently-calibrated models produces a
badly-calibrated average. If probabilities will be used as probabilities (not just
rankings), calibrate members first via `tabular-uncertainty`, then average.

If only ranking matters, rank averaging sidesteps the issue.

---

# Regression notes

Simple averaging of regression predictions reduces variance but pulls toward the
mean. If one member is clearly better, weighted averaging usually beats a plain mean.

For heavy-tailed targets, consider averaging in log space when the target is
strictly positive. Report the transformation.

---

# Output

- members and their individual held-out scores
- pairwise prediction correlations
- ensemble method used and why
- ensemble score on the untouched test set
- relative gain over the best single model
- **adoption decision**, with the threshold that drove it
- inference cost note when adopted (members must all run at scoring time)

---

# Do not

Do not ensemble models evaluated on different splits.

Do not fit a stacking meta-learner on in-fold predictions.

Do not adopt an ensemble that fails the improvement threshold.

Do not exceed `max_members`.

Do not ensemble away an interpretability requirement without flagging it.
