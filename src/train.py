"""Credit-card fraud detection using XGBoost, SMOTE and threshold tuning."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download creditcard.csv and place it in data/."
        )

    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns="Class")
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())

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
    model.fit(X_resampled, y_resampled)

    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, probabilities):.4f}")

    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        rows.append({
            "Threshold": round(float(threshold), 2),
            "Precision": precision,
            "Recall": recall,
            "False Positives": fp,
            "False Negatives": fn,
        })

    results = pd.DataFrame(rows)
    valid = results[(results["Recall"] >= 0.80) & (results["Precision"] >= 0.50)]
    threshold = float(valid.iloc[0]["Threshold"]) if not valid.empty else 0.50
    predictions = (probabilities >= threshold).astype(int)

    print(f"Selected threshold: {threshold:.2f}")
    print(classification_report(y_test, predictions, zero_division=0))

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    plt.title("Fraud Detection Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print("Top feature importance scores:")
    print(importance.head(15))
    importance.to_csv("feature_importance.csv", index=False)
    results.to_csv("threshold_results.csv", index=False)


if __name__ == "__main__":
    main()
