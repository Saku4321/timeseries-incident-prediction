# Predictive Incident Detection — Time-Series Sliding Window Model

Binary classifier that predicts whether a system incident will occur within the next **H = 6** time steps (30 minutes), based on the previous **W = 12** time steps (1 hour) of machine sensor data.

---

## Problem Formulation

Given a multivariate time series of machine metrics, at each time step `t` the model answers:

> **Will any incident occur in the interval `[t+1, t+H]`?**

A sliding window of width `W` is extracted as the feature vector, and a binary label is assigned based on whether any anomaly occurs within the next `H` steps.

```
... [ t-W+1, ..., t ] --> model --> P(incident in [t+1, t+H])
```

**Parameters:**
- `W = 12` steps → 1 hour of historical context (5-min intervals)
- `H = 6` steps → 30-minute prediction horizon

---

## Dataset

**Numenta Anomaly Benchmark (NAB)** — `machine_temperature_system_failure.csv`

Real-world machine temperature sensor logs with labeled anomaly intervals. The dataset spans several months of 5-minute readings.

**Incident definition:** A threshold breach is defined as the temperature exceeding the **90th percentile** of the training set values. This approach is interpretable, domain-agnostic, and avoids reliance on sparse manual labels.

---

## Feature Engineering

Each time step contributes the following features to the sliding window:

| Feature | Description |
|---|---|
| `value` | Raw temperature reading |
| `temp_diff` | First-order difference (rate of change) |
| `rolling_mean_3` | Rolling mean over 3 steps |
| `rolling_mean_6` | Rolling mean over 6 steps |
| `rolling_std_3` | Rolling std over 3 steps |
| `rolling_std_6` | Rolling std over 6 steps |

Each window of width `W = 12` across 6 features produces a **72-dimensional** feature vector per sample.

---

## Model

**XGBoost Classifier** was selected for the following reasons:

- Handles tabular, non-sequential features well without requiring sequence modeling
- Robust to feature scale differences
- Built-in handling of class imbalance via `scale_pos_weight`
- Fast training and inference, suitable for near-real-time alerting pipelines
- Interpretable via feature importance

**Class imbalance** (~12% positive class) is addressed by setting:
```python
scale_pos_weight = count(negative) / count(positive)
```

**Hyperparameters:**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_weight,
    eval_metric='logloss',
    random_state=42
)
```

---

## Evaluation Setup

### Train / Test Split

A **chronological split** (80% train / 20% test) is used — no shuffling — to respect the temporal ordering of the data and prevent leakage of future information into training.

### Cross-Validation

`TimeSeriesSplit` with 5 folds is used to validate robustness across different time periods.

### Alert Threshold

The model outputs a probability `P(incident)`. Alerts are triggered when this probability exceeds a configurable threshold. Two thresholds are evaluated:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 (default) | 0.92 | 0.97 | 0.95 |
| 0.40 (recall-optimized) | 0.91 | 0.98 | 0.94 |

In a real alerting system, **Recall** is the primary metric — a missed incident (false negative) is far more costly than a false alarm. The threshold can be tuned based on the operational cost ratio.

### Metrics

- **Recall (primary)** — minimize missed incidents
- **Precision** — minimize alert fatigue
- **F1-score** — harmonic balance
- **ROC-AUC** — threshold-independent performance
- **Precision-Recall curve** — used to select the optimal alert threshold

---

## Results

| Metric | Train | Test |
|---|---|---|
| Accuracy | ~100% | 97.1% |
| Recall (class 1) | 100% | 97% |
| Precision (class 1) | 97% | 92% |
| F1 (class 1) | 99% | 95% |

**Cross-validation (TimeSeriesSplit, 5 folds):**
- Recall: `0.978 ± 0.009`
- Precision: `0.885 ± 0.027`
- F1: `0.929 ± 0.014`

The low standard deviation across folds confirms the model generalizes well across different time periods, not just one lucky split.

---

## Limitations

- **Static threshold:** The 90th percentile incident definition is computed once on training data. In production, hardware aging or seasonal workload shifts may require dynamic recalibration (e.g., rolling quantiles).
- **Single metric:** The model uses only machine temperature. A real system would benefit from multiple correlated sensors.
- **No concept drift detection:** The model does not monitor for distribution shift over time.

---

## Adapting to a Real Alerting System

```
IoT Sensors
    │
    ▼
Streaming Pipeline (Apache Kafka / AWS Kinesis)
    │
    ▼
Feature Engineering Service (rolling stats, diffs)
    │
    ▼
Inference Microservice (FastAPI + Docker)
    │  — runs every 5 minutes
    │  — maintains a rolling buffer of W=12 steps
    │
    ▼
Alert Router
    ├── P(incident) >= threshold → PagerDuty / Slack webhook
    └── P(incident) < threshold → log & continue
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
```

---

## Usage

Open and run `nt0.ipynb` sequentially. The notebook is self-contained and downloads the dataset automatically.
