---
name: tabular-checkpoint
description: Persist and restore fitted tabular models with full metadata — preprocessing, feature schema, metrics, split definition, and environment. Enables fast repeated scoring without retraining and makes results reproducible.
---

# Purpose

Save a model so it can be trusted later.

A bare pickle is not a checkpoint. Without the feature schema, preprocessing, and
the metrics it earned, a saved model is a liability.

Invoked by `tabular` as step 9, and first whenever the user needs fast scoring —
load a checkpoint before considering retraining.

---

# What a checkpoint contains

**The fitted model or pipeline** — including all preprocessing. Save the pipeline,
never the bare estimator.

**Feature schema** — ordered column names and dtypes. Column order matters for most
estimators.

**Target metadata** — name, task type, class labels and their order.

**Preprocessing artifacts** — encoders, imputers, scalers, the categorical feature
list. Inside the pipeline if possible.

**Evaluation metrics** — the held-out numbers this model actually earned.

**Split definition** — strategy, sizes, random seed, time boundary or group column.

**Environment** — Python, library versions, training timestamp, git commit if
available.

**Uncertainty artifacts** — the conformal quantile or calibrator, when produced.

---

# Layout

```
artifacts/models/<name>_<YYYYMMDD_HHMMSS>/
├── model.joblib          # fitted pipeline
├── metadata.json         # schema, metrics, environment, split
├── feature_names.json    # ordered features and dtypes
├── metrics.json          # full evaluation output
└── conformal.json        # optional: quantile / calibration params
```

Use `artifacts/models/latest` as a symlink to the newest checkpoint so scoring code
does not hardcode a timestamp.

---

# Reference implementation

```python
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import sklearn


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def save_checkpoint(model, *, name, X_train, target, task, metrics,
                    split_info=None, cat_features=None, conformal=None,
                    base_dir="artifacts/models"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"{name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / "model.joblib")

    schema = {c: str(X_train[c].dtype) for c in X_train.columns}
    (out / "feature_names.json").write_text(
        json.dumps({"order": list(X_train.columns), "dtypes": schema}, indent=2)
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    metadata = {
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "target": target,
        "n_features": X_train.shape[1],
        "n_train_rows": len(X_train),
        "categorical_features": cat_features or [],
        "split": split_info or {},
        "primary_metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "git_commit": _git_commit(),
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    if conformal:
        (out / "conformal.json").write_text(json.dumps(conformal, indent=2))

    latest = Path(base_dir) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.name)

    return out


def load_checkpoint(path):
    path = Path(path)
    model = joblib.load(path / "model.joblib")
    metadata = json.loads((path / "metadata.json").read_text())
    features = json.loads((path / "feature_names.json").read_text())
    return model, metadata, features


def validate_input(df, features, strict_dtypes=False):
    """Run before every scoring call. Silent schema drift is the top production failure."""
    issues = []
    expected = features["order"]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        issues.append(f"missing columns: {missing}")
    if extra:
        issues.append(f"unexpected columns (will be dropped): {extra}")
    if strict_dtypes:
        for c in expected:
            if c in df.columns and str(df[c].dtype) != features["dtypes"][c]:
                issues.append(
                    f"{c}: dtype {df[c].dtype}, expected {features['dtypes'][c]}"
                )
    return issues, df.reindex(columns=expected)
```

---

# Scoring from a checkpoint

Before retraining anything, check for an existing checkpoint. Retraining a model that
already exists wastes time and produces a different model than the one that was
validated.

Scoring sequence:

1. Load model and metadata.
2. Validate the incoming schema with `validate_input`. Reorder columns to match.
3. Refuse to score on missing required columns. Warn on extra ones.
4. Predict.
5. Apply conformal or calibration artifacts if present.
6. Write predictions with an identifier column and the checkpoint name used.

Always record which checkpoint produced a set of predictions. Untraceable prediction
files cause real incidents.

---

# Model-specific notes

**CatBoost** — has native `save_model` / `load_model` (`.cbm`), which is portable
across versions and languages. Use it for the raw model, and joblib for the wrapping
pipeline.

**TabICL** — use its own `save`, which can exclude model weights and reload from the
pretrained checkpoint. Record the `tabicl` version; reload requires a compatible one
and access to the pretrained checkpoint.

**Scikit-learn pipelines** — joblib. Pickled sklearn objects are version sensitive;
always record the version and warn on mismatch at load.

---

# Retraining triggers

Recommend retraining when

- input distribution has drifted materially from the training schema summary
- monitored performance degrades beyond an agreed threshold
- the schema changes (new or removed features)
- the checkpoint is old relative to the domain's rate of change
- the target definition changes

Note in metadata when the model was last validated against fresh data.

---

# Output

- checkpoint path
- what was saved
- metrics recorded with it
- how to reload and score, as a copy-pasteable snippet
- version compatibility warnings

---

# Do not

Do not save a bare estimator without its preprocessing.

Do not save without metrics; a model without its scorecard cannot be trusted later.

Do not overwrite checkpoints in place. Timestamp them and move the `latest` symlink.

Do not load a checkpoint and score without schema validation.

Do not commit large model binaries to git; use artifact storage and record the URI.
