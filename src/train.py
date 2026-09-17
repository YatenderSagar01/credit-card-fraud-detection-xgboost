import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)

warnings.filterwarnings("ignore")


# ==========================================================
# 1. CREATE PROJECT DIRECTORIES
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

DATA_PATH = "data/creditcard.csv"

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - CASE STUDY 2")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================================
# 3. BASIC DATA UNDERSTANDING
# ==========================================================

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

genuine_count = int((df["Class"] == 0).sum())
fraud_count = int((df["Class"] == 1).sum())

fraud_percentage = (fraud_count / len(df)) * 100

print("\nGenuine transactions:", genuine_count)
print("Fraudulent transactions:", fraud_count)
print(f"Fraud percentage: {fraud_percentage:.4f}%")


# ==========================================================
# 4. CLASS DISTRIBUTION GRAPH
# ==========================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.xticks(
    ticks=[0, 1],
    labels=["Genuine", "Fraudulent"]
)

plt.tight_layout()
plt.savefig("reports/class_distribution.png", dpi=300)
plt.close()


# ==========================================================
# 5. REMOVE DUPLICATES
# ==========================================================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\nDuplicate rows removed:", before_duplicates - after_duplicates)


# ==========================================================
# 6. SEPARATE FEATURES AND TARGET
# ==========================================================

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================================
# 7. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================================
# 8. FEATURE PREPROCESSING
# ==========================================================

print("\nApplying feature preprocessing...")

X_train = X_train.copy()
X_test = X_test.copy()

# Scale Amount separately
amount_scaler = StandardScaler()

X_train["Amount"] = amount_scaler.fit_transform(
    X_train[["Amount"]]
)

X_test["Amount"] = amount_scaler.transform(
    X_test[["Amount"]]
)

# Time is removed for this model
X_train = X_train.drop("Time", axis=1)
X_test = X_test.drop("Time", axis=1)

print("Preprocessing completed.")


# ==========================================================
# 9. HANDLE CLASS IMBALANCE
# ==========================================================

negative_samples = (y_train == 0).sum()
positive_samples = (y_train == 1).sum()

scale_pos_weight = negative_samples / positive_samples

print("\nClass imbalance ratio:", scale_pos_weight)


# ==========================================================
# 10. CREATE XGBOOST MODEL
# ==========================================================

print("\nCreating XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ==========================================================
# 11. TRAIN MODEL
# ==========================================================

print("\nTraining XGBoost model...")
print("This may take some time...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ==========================================================
# 12. MAKE PREDICTIONS
# ==========================================================

y_probability = model.predict_proba(X_test)[:, 1]

# Default classification threshold
default_threshold = 0.50

y_pred = (
    y_probability >= default_threshold
).astype(int)


# ==========================================================
# 13. MODEL EVALUATION
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")


# ==========================================================
# 14. CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraudulent"],
    zero_division=0
)

print(report)

with open(
    "reports/classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ==========================================================
# 15. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraudulent"],
    yticklabels=["Genuine", "Fraudulent"]
)

plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300
)
plt.close()


# ==========================================================
# 16. ROC CURVE
# ==========================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc_curve = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost ROC-AUC = {roc_auc_curve:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    "reports/roc_curve.png",
    dpi=300
)
plt.close()


# ==========================================================
# 17. PRECISION-RECALL CURVE
# ==========================================================

precision_values, recall_values, pr_thresholds = precision_recall_curve(
    y_test,
    y_probability
)

pr_auc_curve = auc(
    recall_values,
    precision_values
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"PR-AUC = {pr_auc_curve:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    "reports/pr_curve.png",
    dpi=300
)
plt.close()


# ==========================================================
# 18. THRESHOLD ANALYSIS
# ==========================================================

print("\nPerforming threshold analysis...")

thresholds = np.arange(
    0.05,
    1.00,
    0.05
)

threshold_results = []

for threshold in thresholds:

    threshold_prediction = (
        y_probability >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_prediction,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_prediction,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_prediction,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": round(float(threshold), 2),
        "Precision": threshold_precision,
        "Recall": threshold_recall,
        "F1_Score": threshold_f1
    })


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df.to_csv(
    "reports/threshold_results.csv",
    index=False
)

best_threshold_row = threshold_df.loc[
    threshold_df["F1_Score"].idxmax()
]

best_threshold = float(
    best_threshold_row["Threshold"]
)

print("\nBest threshold based on F1-score:")
print(best_threshold)

print("\nBest threshold results:")
print(best_threshold_row)


# ==========================================================
# 19. EVALUATE USING BEST THRESHOLD
# ==========================================================

best_threshold_prediction = (
    y_probability >= best_threshold
).astype(int)

best_precision = precision_score(
    y_test,
    best_threshold_prediction,
    zero_division=0
)

best_recall = recall_score(
    y_test,
    best_threshold_prediction,
    zero_division=0
)

best_f1 = f1_score(
    y_test,
    best_threshold_prediction,
    zero_division=0
)

print("\n" + "=" * 70)
print("BEST THRESHOLD EVALUATION")
print("=" * 70)

print(f"\nSelected threshold: {best_threshold:.2f}")
print(f"Precision:          {best_precision:.4f}")
print(f"Recall:             {best_recall:.4f}")
print(f"F1-score:           {best_f1:.4f}")


# ==========================================================
# 20. FEATURE IMPORTANCE
# ==========================================================

print("\nGenerating feature importance...")

feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

feature_importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 8))

top_features = feature_importance_df.head(15)

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig(
    "reports/feature_importance.png",
    dpi=300
)
plt.close()


# ==========================================================
# 21. SAVE MODEL AND SCALER
# ==========================================================

print("\nSaving model files...")

joblib.dump(
    model,
    "models/fraud_xgboost_model.joblib"
)

joblib.dump(
    amount_scaler,
    "models/fraud_scaler.joblib"
)


# ==========================================================
# 22. SAVE MODEL METADATA
# ==========================================================

metadata = {
    "model": "XGBoost Classifier",
    "dataset": "Kaggle Credit Card Fraud Detection",
    "dataset_shape_after_cleaning": list(df.shape),
    "features_used": list(X_train.columns),
    "target_column": "Class",
    "test_size": 0.20,
    "random_state": 42,
    "default_threshold": default_threshold,
    "best_threshold": best_threshold,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "best_threshold_precision": float(best_precision),
    "best_threshold_recall": float(best_recall),
    "best_threshold_f1": float(best_f1)
}

with open(
    "models/model_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================================
# 23. SAVE SUMMARY RESULTS
# ==========================================================

summary_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
        "Best Threshold",
        "Best Threshold Precision",
        "Best Threshold Recall",
        "Best Threshold F1-Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc,
        best_threshold,
        best_precision,
        best_recall,
        best_f1
    ]
})

summary_results.to_csv(
    "reports/model_results.csv",
    index=False
)


# ==========================================================
# 24. FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSaved model files:")
print("models/fraud_xgboost_model.joblib")
print("models/fraud_scaler.joblib")
print("models/model_metadata.json")

print("\nSaved report files:")
print("reports/class_distribution.png")
print("reports/confusion_matrix.png")
print("reports/roc_curve.png")
print("reports/pr_curve.png")
print("reports/feature_importance.png")
print("reports/feature_importance.csv")
print("reports/threshold_results.csv")
print("reports/model_results.csv")
print("reports/classification_report.txt")

print("\nFraud detection model is ready.")