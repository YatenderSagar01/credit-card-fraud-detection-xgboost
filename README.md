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

### Run

```bash
pip install -r requirements.txt
python src/train.py
```
