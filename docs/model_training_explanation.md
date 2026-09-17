# Credit Card Fraud Detection — Model Training Explanation

## Objective

Predict whether a credit-card transaction is fraudulent.

- `Class = 0`: genuine transaction
- `Class = 1`: fraudulent transaction

## Dataset

Download the Kaggle Credit Card Fraud Detection dataset and save it as:

```text
data/creditcard.csv
```

The dataset is highly imbalanced, so accuracy alone is not sufficient.

## Workflow

```text
Load real dataset
        ↓
Stratified train/test split
        ↓
Standardize Time and Amount
        ↓
Apply SMOTE only to training data
        ↓
Train XGBoost classifier
        ↓
Generate fraud probabilities
        ↓
Tune decision threshold
        ↓
Calculate ROC-AUC, PR-AUC, precision, recall, F1
        ↓
Calculate false positives and false negatives
        ↓
Save model and feature importance
```

## Why SMOTE is applied after splitting

SMOTE creates synthetic minority-class examples. It is applied only to the training set so that synthetic information does not leak into the test set.

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

## XGBoost model

```python
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)
```

XGBoost produces a probability between 0 and 1 for fraud.

## Threshold tuning

The default classification threshold is 0.50. Because fraud detection is imbalanced, several thresholds are tested from 0.10 to 0.90.

```python
predictions = (probabilities >= threshold).astype(int)
```

The threshold with the highest F1-score is selected. A lower threshold may detect more fraud but can create more false alerts. A higher threshold may reduce false alerts but miss more fraud.

## Evaluation metrics

- **ROC-AUC:** Ranking ability across classification thresholds.
- **PR-AUC:** Especially useful for imbalanced fraud data.
- **Precision:** Of all transactions flagged as fraud, how many were actually fraud?
- **Recall:** Of all actual fraud transactions, how many were detected?
- **F1-score:** Harmonic mean of precision and recall.
- **False positive:** Genuine transaction incorrectly flagged as fraud.
- **False negative:** Fraudulent transaction incorrectly classified as genuine.

## Feature importance

XGBoost feature importance scores are saved in:

```text
results/feature_importance.csv
```

The scores describe how much the trained model used each feature. They do not prove that a feature causes fraud.

## Generated files

```text
models/xgboost_fraud_model.joblib
models/fraud_scaler.joblib
results/metrics.json
results/threshold_results.csv
results/feature_importance.csv
```

## Run the project

```bash
pip install -r requirements.txt
python src/train.py
```
