---
name: tabular-inspect
description: Profile a tabular dataset before modeling. Detect task type, column roles, missingness, cardinality, imbalance, leakage risk, and temporal or group structure. Produces a data profile that downstream tabular skills consume.
---

# Purpose

Understand the dataset before touching a model.

This skill produces a **data profile**. It does not fit models and does not modify data.

Invoked by `tabular` as step 1.

---

# What to determine

**Shape** — rows, columns, memory footprint.

**Target** — name, dtype, distinct values, missing count.

**Task type** — binary classification, multiclass classification, or regression.

**Column roles** — numeric, categorical, boolean, datetime, text, identifier, constant.

**Missingness** — per column, and whether it looks structural or random.

**Cardinality** — distinct counts for categoricals; flag high-cardinality columns.

**Imbalance** — class shares for classification.

**Duplicates** — exact duplicate rows, duplicate identifiers.

**Temporal structure** — any datetime column that plausibly orders the data.

**Group structure** — any identifier repeating across rows (customer, patient, device).

**Leakage risk** — features suspiciously correlated with the target, or named like
post-outcome fields.

---

# Task detection

Read thresholds from `skills/tabular/defaults.yaml` under `task_detection`.

Rules, applied in order.

Target is non-numeric with two distinct values → binary classification.

Target is non-numeric with more than two distinct values → multiclass classification.

Target is numeric with at most `max_unique_for_classification` distinct values, and
those values are integers → classification. Say so explicitly and let the user override.

Target is numeric otherwise → regression.

Target is a datetime → stop and ask. This is likely a forecasting problem, not a
standard supervised table.

---

# Column role heuristics

Identifier — near-unique values, name contains `id`, `uuid`, `key`, `_no`, `number`.
Exclude from features by default.

Constant — one distinct non-null value. Drop.

High cardinality categorical — distinct count above `one_hot_max_cardinality`
in defaults. Flag for CatBoost native handling; do not one-hot.

Datetime — parseable as dates in more than 90 percent of non-null rows. Do not feed
raw. Either use for splitting or derive features explicitly.

Text — long free-form strings, high distinct count, whitespace present. Flag as
out of scope for v1 unless the user asks for it.

---

# Leakage checks

Flag, do not silently drop.

Any single feature with absolute correlation above 0.98 with a numeric target.

Any single categorical feature that perfectly separates a classification target.

Columns whose names suggest they are recorded after the outcome: `outcome`, `result`,
`status_final`, `churn_date`, `refund`, `resolved_at`, `label_`, `target_`.

Duplicate rows appearing in what will become both train and test.

See `references/pitfalls.md` for the full list and how each one bites.

---

# Reference implementation

```python
import numpy as np
import pandas as pd

def inspect(df: pd.DataFrame, target: str) -> dict:
    profile = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "target": target,
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
    }

    y = df[target]
    n_unique = y.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(y):
        profile["task"] = "binary" if n_unique == 2 else "multiclass"
    elif n_unique <= 20 and (y.dropna() % 1 == 0).all():
        profile["task"] = "binary" if n_unique == 2 else "multiclass"
    else:
        profile["task"] = "regression"

    if profile["task"] != "regression":
        shares = y.value_counts(normalize=True, dropna=True)
        profile["class_shares"] = shares.round(4).to_dict()
        profile["minority_share"] = float(shares.min())
        profile["imbalanced"] = bool(shares.min() < 0.10)

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        nun = int(s.nunique(dropna=True))
        info = {
            "dtype": str(s.dtype),
            "missing_frac": round(float(s.isna().mean()), 4),
            "n_unique": nun,
        }
        if nun <= 1:
            info["role"] = "constant"
        elif nun == len(df) and not pd.api.types.is_numeric_dtype(s):
            info["role"] = "identifier"
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["role"] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            info["role"] = "numeric"
        else:
            info["role"] = "high_card_categorical" if nun > 30 else "categorical"
        profile["columns"][col] = info

    # Leakage: near-perfect numeric correlation with a numeric target
    if profile["task"] == "regression":
        num = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
        if not num.empty:
            corr = num.corrwith(y).abs()
            profile["leakage_suspects"] = corr[corr > 0.98].index.tolist()

    return profile
```

---

# Output

Report a compact summary, not a wall of numbers.

- shape and task type
- target distribution (class shares, or min/median/max)
- columns dropped and why
- high-cardinality columns
- missingness above 20 percent
- temporal or group structure detected
- leakage suspects, stated as questions to the user

Recommend a split strategy and pass it to `tabular-split`.

---

# Do not

Do not impute, encode, or scale here. That belongs to the model skills.

Do not drop leakage suspects silently. Surface them.

Do not print full dataframes. Summarize.
