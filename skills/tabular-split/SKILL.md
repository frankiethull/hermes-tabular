---
name: tabular-split
description: Split data: random, stratified, time-aware
---

# Purpose

Decide how data is divided so that reported performance means something.

This is the highest-leverage skill in the pack. A wrong split invalidates every
number that follows.

Invoked by `tabular` as step 4.

---

# Strategy selection

Apply in this order. The **first** matching rule wins.

**Group structure exists** (repeated customer, patient, device, household)
→ `GroupKFold` or `GroupShuffleSplit`. A group must never span train and test.

**Temporal structure exists** and predictions are about the future
→ time-ordered split. Train on the past, test on the most recent window. Use
`TimeSeriesSplit` for cross-validation. Never shuffle.

**Both group and time exist**
→ split by time, then verify no group leaks across the boundary. If groups straddle
the cut, drop the straddling rows from train and say so.

**Classification, no group or time structure**
→ `StratifiedKFold` or `train_test_split(stratify=y)`.

**Regression, no group or time structure**
→ `KFold` with shuffle, or plain `train_test_split`.

**Fewer than `cv_below_rows`** in defaults → cross-validation rather than a single
holdout. A single small test set is noise.

---

# Sizes

Read from `skills/tabular/defaults.yaml` under `split`.

Defaults: `test_size: 0.2`, `validation_size: 0.2` of the remainder,
`random_state: 42`, `cv_folds: 5`, `cv_folds_small: 10`.

Three-way split when a model needs early stopping or calibration:
train → fit, validation → early stopping and hyperparameters, test → touched once.

The test set is touched **once**, at the very end. If you look at it twice, it is
now a validation set and you must say so.

---

# Reference implementation

```python
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    GroupShuffleSplit, GroupKFold, TimeSeriesSplit,
)


def choose_strategy(profile: dict) -> str:
    if profile.get("group_column"):
        return "grouped"
    if profile.get("time_column") and profile.get("is_forecasting", True):
        return "temporal"
    if profile["task"] in ("binary", "multiclass"):
        return "stratified"
    return "random"


def split_holdout(df, target, strategy, *, group_col=None, time_col=None,
                  test_size=0.2, random_state=42):
    y = df[target]

    if strategy == "temporal":
        df = df.sort_values(time_col)
        cut = int(len(df) * (1 - test_size))
        train, test = df.iloc[:cut], df.iloc[cut:]
        boundary = df[time_col].iloc[cut]
        return train, test, {"boundary": str(boundary)}

    if strategy == "grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=random_state)
        tr, te = next(gss.split(df, y, groups=df[group_col]))
        return df.iloc[tr], df.iloc[te], {"n_groups": df[group_col].nunique()}

    stratify = y if strategy == "stratified" else None
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train, test, {}


def make_cv(strategy, n_splits=5, random_state=42):
    if strategy == "temporal":
        return TimeSeriesSplit(n_splits=n_splits)
    if strategy == "grouped":
        return GroupKFold(n_splits=n_splits)
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
```

---

# Verification

After splitting, always check and report:

- row counts per split
- class distribution per split for classification; warn if any class has fewer than
  10 examples in test
- date ranges per split for temporal splits; confirm train max < test min
- group overlap for grouped splits; must be exactly zero
- duplicate rows spanning train and test; must be zero

```python
def verify(train, test, target, group_col=None, time_col=None):
    issues = []
    if group_col:
        overlap = set(train[group_col]) & set(test[group_col])
        if overlap:
            issues.append(f"{len(overlap)} groups appear in both train and test")
    if time_col:
        if train[time_col].max() >= test[time_col].min():
            issues.append("temporal overlap between train and test")
    if len(test) < 30:
        issues.append(f"test set has only {len(test)} rows; prefer cross-validation")
    return issues
```

Report every issue. Do not proceed silently past a group overlap.

---

# Imbalanced classification

Stratify. Always.

If the minority class has fewer than 10 examples in the test fold, increase test
size or switch to repeated stratified cross-validation.

Do not oversample before splitting. Resampling belongs inside the training fold
only, otherwise copies of the same row land on both sides.

---

# Output

- chosen strategy and the rule that triggered it
- split sizes
- verification results
- the CV object or index arrays for downstream skills
- an explicit statement of what the test set represents in deployment terms

---

# Do not

Do not shuffle time series.

Do not stratify on a continuous target by default. Binning is occasionally useful
for very small regression datasets; say so if you do it.

Do not re-split between models. Every candidate model sees the identical split, or
the comparison is meaningless.
