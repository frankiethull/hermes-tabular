---
name: tabular-catboost
description: Train gradient boosted trees with CatBoost for classification or regression. Handles categorical features natively, uses early stopping against a validation set, and reports feature importance. The default strong model for most tabular problems.
---

# Purpose

The workhorse. CatBoost is the default strong model for mixed-type tabular data.

It handles categorical features natively with ordered target statistics, tolerates
missing values, and needs little tuning to be competitive.

This skill deliberately covers the whole gradient boosting family. XGBoost and
LightGBM are not in v1 because CatBoost covers nearly all of that space while
reducing decision fatigue.

Invoked by `tabular` after the baseline.

---

# Why CatBoost as the single default

Native categorical handling. No one-hot explosion, no manual target encoding, no
leakage from naive encodings — ordered boosting handles it.

Strong defaults. Competitive out of the box; tuning usually buys single-digit
percent improvements.

Missing values handled internally.

Robust to correlated and irrelevant features.

Add a second backend only when a specific, demonstrated need appears.

---

# Setup

Declare categorical features explicitly. This is the single most important step.

```python
cat_features = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]
```

CatBoost requires categorical columns to contain no `NaN`. Fill with the literal
string `"__missing__"` — the model treats it as its own level, which is usually
informative.

```python
for c in cat_features:
    X[c] = X[c].astype(str).fillna("__missing__")
```

Never one-hot encode before CatBoost. You would be discarding its main advantage.

---

# Parameters

Read from `defaults.yaml` under `models.catboost`.

```python
params = {
    "iterations": 2000,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "early_stopping_rounds": 100,
    "verbose": 200,
}
```

Leave `learning_rate` unset — CatBoost picks a sensible value from dataset size.

Loss functions by task:

| Task | `loss_function` | `eval_metric` |
| --- | --- | --- |
| Binary | `Logloss` | `AUC` |
| Multiclass | `MultiClass` | `TotalF1` |
| Regression | `RMSE` | `RMSE` |
| Regression, heavy tails / outliers | `MAE` or `Huber:delta=1.0` | `MAE` |

Imbalance: set `auto_class_weights="Balanced"`. Do not resample.

---

# Reference implementation

```python
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


def prepare(X, cat_features):
    X = X.copy()
    for c in cat_features:
        X[c] = X[c].astype(str).fillna("__missing__")
    return X


def fit_catboost(X_train, y_train, X_valid, y_valid, task,
                 cat_features, imbalanced=False, **overrides):
    X_train = prepare(X_train, cat_features)
    X_valid = prepare(X_valid, cat_features)

    params = dict(
        iterations=2000, depth=6, l2_leaf_reg=3.0,
        random_seed=42, early_stopping_rounds=100, verbose=200,
    )
    if task == "binary":
        params.update(loss_function="Logloss", eval_metric="AUC")
        if imbalanced:
            params["auto_class_weights"] = "Balanced"
        Model = CatBoostClassifier
    elif task == "multiclass":
        params.update(loss_function="MultiClass", eval_metric="TotalF1")
        if imbalanced:
            params["auto_class_weights"] = "Balanced"
        Model = CatBoostClassifier
    else:
        params.update(loss_function="RMSE", eval_metric="RMSE")
        Model = CatBoostRegressor

    params.update(overrides)

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

    model = Model(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model, {
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score(),
    }
```

---

# Early stopping

Always fit with an `eval_set`. Set `iterations` high (2000) and let early stopping
find the right number — that is more reliable than guessing.

Report `best_iteration`. If it equals `iterations - 1`, the model never stopped;
raise the ceiling and refit.

When there is no separate validation split (small data, cross-validation mode), use
`catboost.cv` to find the iteration count, then refit on all training data with that
fixed count.

---

# Tuning

Do not tune by default. Only tune when

- the time budget exceeds `tune_if_time_budget_seconds_above` in defaults, **and**
- CatBoost's margin over the baseline is small enough that tuning could change the
  decision.

When you do tune, keep it to a handful of parameters:

```python
search_space = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.06, 0.1],
    "l2_leaf_reg": [1.0, 3.0, 9.0],
}
```

Randomized search with ~20 trials. Tune against the validation set, never the test set.

---

# Feature importance

Report `PredictionValuesChange` (the default) for a quick view.

For anything that informs a decision, use SHAP:

```python
shap_values = model.get_feature_importance(train_pool, type="ShapValues")
```

Importance is not causality. A feature can be important because it leaks. Cross-check
the top features against the leakage suspects from `tabular-inspect`.

---

# GPU

Set `task_type="GPU"` when a GPU is available and the dataset exceeds ~100k rows.
Below that, CPU is often faster because of transfer overhead. GPU changes some
defaults, so results will not match CPU exactly; say so if you switch.

---

# Output

- task, loss function, and eval metric used
- categorical features passed natively
- best iteration and whether early stopping triggered
- validation metric
- top 10 features by importance
- fitted model handed to `tabular-evaluate`
- training time

---

# Do not

Do not one-hot encode before CatBoost.

Do not fill numeric `NaN`; CatBoost handles it.

Do not tune on the test set.

Do not report the number of iterations you asked for as if it were the number used.
