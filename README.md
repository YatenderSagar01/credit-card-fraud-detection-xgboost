# Credit Card Fraud Detection with XGBoost

## Case Study 2

This project detects fraudulent credit card transactions using XGBoost on an imbalanced dataset. SMOTE is applied only to the training split, decision thresholds are tuned, and feature importance is reported.

### Dataset

Use the Kaggle Credit Card Fraud Detection dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it at `data/creditcard.csv`. Do not commit large datasets or private financial information.

### Methods

- Stratified train/test split
- Standardization of `Time` and `Amount`
- SMOTE oversampling on training data only
- XGBoost classifier
- ROC-AUC and PR-AUC
- Threshold tuning
- Confusion matrix
- Feature importance

### Why accuracy is insufficient

Fraud is a minority class. A model can obtain high accuracy while detecting very few fraudulent transactions. Precision, recall, PR-AUC, false positives, and false negatives are therefore important.

### Threshold interpretation

A lower threshold generally increases fraud recall but may create more false-positive alerts. A higher threshold may reduce alerts but miss more fraud. The selected threshold should reflect the financial cost of fraud, investigation cost, and customer experience.

### Verified run results

Run date: 2026-09-16  
Dataset: 284,807 transactions, 492 fraud cases  
Split: 80/20 stratified  
SMOTE training rows: 454,902  

| Metric | Result |
|---|---:|
| ROC-AUC | 0.98094 |
| PR-AUC | 0.84856 |
| Precision at threshold 0.50 | 0.37069 |
| Recall at threshold 0.50 | 0.87755 |
| F1 at threshold 0.50 | 0.52121 |
| Selected threshold | 0.65 |
| Precision at selected threshold | 0.50602 |
| Recall at selected threshold | 0.85714 |
| F1 at selected threshold | 0.63636 |

Selected-threshold confusion matrix:

```text
[[56782, 82],
 [14, 84]]
```

Detailed metrics are stored in `results/metrics.json`. The full locally generated plots and CSV outputs are available in the accompanying results package because the connected GitHub file interface supports text files but not binary image uploads.

### Run

```bash
pip install -r requirements.txt
python src/train.py
```
