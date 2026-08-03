---
name: tabular
description: End-to-end tabular machine learning orchestration. Inspect data, select an appropriate modeling strategy, evaluate honestly, quantify uncertainty when requested, and persist models for future scoring.
---

# Purpose

This skill is the primary entry point for tabular machine learning tasks.

Do not immediately begin fitting models.

First inspect the dataset and determine an appropriate workflow.

Delegate implementation work to specialized tabular skills.

Never duplicate functionality implemented elsewhere.

---

# Workflow

Always follow this sequence.

1. Inspect dataset.
2. Load data.
3. Determine task.
4. Determine split strategy.
5. Select modeling strategy.
6. Train.
7. Evaluate.
8. Optional uncertainty.
9. Save checkpoint.
10. Produce predictions.

---

# Delegate

Dataset inspection

→ tabular-inspect

Loading

→ tabular-load

Splitting

→ tabular-split

Linear or logistic baseline

→ tabular-baseline

Tree models

→ tabular-catboost

Foundation models

→ tabular-tabicl

Model combination

→ tabular-ensemble

Evaluation

→ tabular-evaluate

Prediction intervals

→ tabular-uncertainty

Persistence

→ tabular-checkpoint

---

# Model policy

Prefer the simplest model that adequately solves the problem.

Default priorities

1. Baseline
2. CatBoost
3. TabICL
4. Ensemble

Use an ensemble only if evaluation demonstrates improvement.

Do not build large model stacks.

---

# Decision rules

Small datasets

Prefer

Baseline

CatBoost

TabICL

Large datasets

Prefer

CatBoost

Very small datasets

Compare

Baseline

CatBoost

TabICL

Mixed numeric and categorical data

Prefer CatBoost.

Need interpretability

Prefer baseline.

Need strongest predictive performance

Compare CatBoost and TabICL.

Need fast daily scoring

Load checkpoint before considering retraining.

---

# Evaluation

Never report training performance as model performance.

Always evaluate on unseen data.

Respect temporal ordering.

Respect grouping.

Prefer stratified splits for classification.

Prefer time-aware splits for forecasting.

---

# Uncertainty

When decisions involve risk, provide uncertainty.

Regression

Conformal prediction intervals.

Classification

Calibrated probabilities.

Optional conformal prediction sets.

---

# Outputs

Every completed analysis should include

- chosen model
- evaluation metrics
- important assumptions
- uncertainty (if requested)
- checkpoint location (if saved)
- prediction artifact (if generated)

Never claim certainty unsupported by evaluation.

---

# Thresholds

All numeric thresholds referenced above live in `defaults.yaml` beside this file.

Read that file rather than hardcoding numbers. If it is missing, use:

- very small: fewer than 1,000 rows
- small: 1,000 to 50,000 rows
- large: more than 50,000 rows

---

# Stop conditions

Stop and ask the user before proceeding when

- the target column is ambiguous or absent
- the target leaks trivially from a feature (see `references/pitfalls.md`)
- the dataset has fewer than 50 labeled rows
- the requested metric conflicts with the detected task type
- honoring a grouping or temporal constraint would leave an empty test set
