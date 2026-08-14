---
name: tabular-tabicl
description: TabICL: zero-tuning, small-data accuracy
---

# Purpose

Run a pretrained tabular foundation model.

TabICL is pretrained on millions of synthetic tables and predicts on a new dataset
through in-context learning — a single forward pass, no gradient updates on your data.

Its value is accuracy on small and medium datasets **with zero tuning**. Where
CatBoost may need a tuning budget, TabICL is often competitive immediately.

Invoked by `tabular` when the dataset fits within scale limits and predictive
performance matters.

---

# When to use

**Good fit**

- 300 to 100,000 training rows
- up to ~2,000 features
- tuning budget is limited
- small data where gradient boosting overfits
- as a second opinion alongside CatBoost before ensembling

**Poor fit**

- very large datasets (accuracy degrades and cost climbs past the row ceiling in
  `defaults.yaml`)
- strong interpretability requirements — this is a transformer, not a coefficient table
- environments with no network access on first run and no cached checkpoint
- extremely wide text-heavy tables

Check `models.tabicl.tabicl_row_ceiling` and `tabicl_feature_ceiling` in defaults
before starting. If the dataset exceeds them, either subsample the training context
and say so, or skip TabICL and use CatBoost.

---

# Installation

```bash
pip install tabicl
```

The first `fit` downloads a pretrained checkpoint (a few hundred MB) and caches it
locally. Warn the user before that download. Subsequent runs are cheap.

A GPU is strongly recommended above ~10,000 rows. CPU works for small data.

---

# Usage

The API is scikit-learn compliant.

```python
from tabicl import TabICLClassifier, TabICLRegressor

clf = TabICLClassifier()
clf.fit(X_train, y_train)      # downloads checkpoint on first use
proba = clf.predict_proba(X_test)
preds = clf.predict(X_test)

reg = TabICLRegressor()
reg.fit(X_train, y_train)
reg.predict(X_test)
```

`fit` is cheap — it stores the context. The real computation happens in `predict`.
Budget wall-clock time accordingly; a slow `predict` is normal, not a hang.

---

# KV caching

When scoring the same training context repeatedly, enable the key-value cache. It is
built during `fit` and reused across `predict` calls.

```python
clf = TabICLClassifier(kv_cache=True)
clf.fit(X_train, y_train)   # caches key-value projections for the context
clf.predict(X_batch_1)      # fast
clf.predict(X_batch_2)      # fast
```

This trades memory for speed. Enable it for repeated scoring, disable it for a
single one-shot evaluation on a memory-constrained machine.

---

# Reference implementation

```python
import time
import numpy as np


def fit_tabicl(X_train, y_train, X_test, task, *, kv_cache=True,
               device="auto", random_state=42, max_rows=100_000):
    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError as err:
        raise ImportError("TabICL not installed. `pip install tabicl`") from err

    if len(X_train) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=max_rows, replace=False)
        X_train, y_train = X_train.iloc[idx], y_train.iloc[idx]
        subsampled = True
    else:
        subsampled = False

    Model = TabICLClassifier if task in ("binary", "multiclass") else TabICLRegressor
    model = Model(kv_cache=kv_cache, device=device, random_state=random_state)

    t0 = time.time()
    model.fit(X_train, y_train)
    if task in ("binary", "multiclass"):
        out = model.predict_proba(X_test)
    else:
        out = model.predict(X_test)
    elapsed = time.time() - t0

    return model, out, {"subsampled": subsampled,
                        "context_rows": len(X_train),
                        "seconds": round(elapsed, 1)}
```

---

# Data preparation

Less preprocessing than the baseline needs.

Categoricals — pass as-is; the model handles them. Encode to integer codes only if
the installed version complains.

Missing values — handled internally. Do not impute.

Scaling — not required. The model normalizes internally.

Feature selection — matters more than for CatBoost, because context length is
bounded. If you are near the feature ceiling, drop constants, identifiers, and
near-duplicate columns first.

---

# Persistence

TabICL supports saving a fitted estimator:

```python
clf.save("classifier.pkl",
         save_model_weights=False,   # reload from checkpoint on load
         save_training_data=True,    # False requires KV cache; useful for privacy
         save_kv_cache=True)
```

Hand the path to `tabular-checkpoint` so it is recorded alongside the other
artifacts. Note in the checkpoint metadata that reload requires the same `tabicl`
version and access to the pretrained checkpoint.

---

# Failure modes

**Out of memory** — reduce the training context by subsampling, disable `kv_cache`,
or move to CPU with more RAM.

**Very slow predict** — expected on CPU with large contexts. Compare against
`max_fit_seconds` in defaults and fall back to CatBoost rather than waiting.

**No network on first run** — the checkpoint download will fail. Pre-cache it, or
skip TabICL entirely and report why.

**Degraded accuracy on very large data** — this is documented behavior past the row
ceiling. Prefer CatBoost there.

Whenever TabICL is skipped or falls back, state it explicitly in the final report.
Silent fallback is worse than no fallback.

---

# Output

- whether TabICL ran, or the reason it was skipped
- context size, and whether training rows were subsampled
- device used
- wall-clock fit and predict time
- validation metric, compared directly against CatBoost and the baseline
- predictions handed to `tabular-evaluate`

---

# Do not

Do not tune TabICL. The point is that it does not need it.

Do not use it as the sole model on data beyond its scale limits.

Do not promise interpretability. Pair it with the baseline if the user needs
explanation.

Do not silently subsample. Always report context size.
