---
name: tabular-baseline
description: Fast linear/logistic baseline with auto task detection, CV reg
---

# Purpose

Produce the number every other model must beat.

Baselines are cheap and they are the only honest way to answer: *did the complicated
model actually help?*

This skill picks between linear and logistic regression automatically. The user
should never have to specify which.

Invoked by `tabular` as the first modeling step. Always run it, even when you expect
CatBoost to win.

---

# Automatic model selection

| Detected task | Estimator |
| --- | --- |
| Binary classification | `LogisticRegression` |
| Multiclass classification | `LogisticRegression` (multinomial) |
| Regression | `Ridge` |
| Regression, wide data (p > n) | `ElasticNet` |
| Regression, heavy-tailed target | `Ridge` on log1p target, if strictly positive |

Also fit a trivial reference in the same run:

- Classification → `DummyClassifier(strategy="prior")`
- Regression → `DummyRegressor(strategy="median")`

If the real baseline does not clearly beat the dummy, something is wrong with the
data, the target, or the split. Stop and report that before going further.

---

# Preprocessing

The baseline needs more preprocessing than CatBoost does. Build it as a pipeline so
it travels with the model.

Numeric — median impute, then standardize.

Low-cardinality categorical (at or below `one_hot_max_cardinality`) — most-frequent
impute, then one-hot with `handle_unknown="ignore"`.

High-cardinality categorical — drop for the baseline. Note the drop. CatBoost will
handle these natively; that difference is part of why it may win.

Datetime — drop raw; the baseline is not the place to engineer calendar features.

---

# Regularization

Search the grid in `defaults.yaml` under `models.baseline` using the CV object from
`tabular-split`. Do not tune beyond that grid. This is a baseline, not a contender.

`C_grid: [0.01, 0.1, 1.0, 10.0]` for logistic.

`alpha_grid: [0.01, 0.1, 1.0, 10.0]` for ridge.

Set `class_weight="balanced"` when the profile flags imbalance.

---

# Reference implementation

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame, max_card: int = 30) -> ColumnTransformer:
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns
           if c not in num and X[c].nunique(dropna=True) <= max_card]
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
            ]), cat),
        ],
        remainder="drop",
    )


def fit_baseline(X, y, task, cv, imbalanced=False, n_jobs=-1):
    pre = build_preprocessor(X)

    if task in ("binary", "multiclass"):
        est = LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if imbalanced else None,
        )
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
        scoring = "roc_auc" if task == "binary" else "f1_macro"
        dummy = DummyClassifier(strategy="prior")
    else:
        est = Ridge()
        grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0]}
        scoring = "neg_root_mean_squared_error"
        dummy = DummyRegressor(strategy="median")

    pipe = Pipeline([("pre", pre), ("model", est)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    search.fit(X, y)

    dummy_pipe = Pipeline([("pre", pre), ("model", dummy)]).fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_, dummy_pipe
```

---

# Interpretation

The baseline earns its keep by being explainable. Always report:

**Coefficients** — standardized, so magnitudes are comparable. Top 10 by absolute
value, with sign.

**Direction** — for logistic, convert to odds ratios via `exp(coef)`. State them in
plain language: "each additional support ticket multiplies the odds of churn by 1.4."

**Intercept** — the base rate the model starts from.

Caveat every coefficient claim: these are associations under a linear additive model,
not causal effects, and correlated features split credit unpredictably.

---

# Output

- estimator chosen and why (task type that triggered it)
- best regularization value
- cross-validated primary metric with standard deviation
- dummy reference score
- top coefficients with interpretation
- columns dropped during preprocessing
- fitted pipeline handed to `tabular-evaluate`

---

# Do not

Do not skip the baseline because the dataset "obviously needs" a stronger model.
The comparison is the point.

Do not tune beyond the small grid.

Do not report the baseline as final without checking whether CatBoost or TabICL
beats it meaningfully.

Do not interpret coefficients as causal.
