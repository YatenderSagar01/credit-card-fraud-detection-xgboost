"""Credit-card fraud detection using XGBoost, SMOTE and threshold tuning."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def evaluate_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download creditcard.csv and place it in data/."
        )

    data = pd.read_csv(DATA_PATH)
    if "Class" not in data.columns:
        raise ValueError("Expected target column 'Class' was not found.")

    X = data.drop(columns="Class").copy()
    y = data["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    for column in ["Time", "Amount"]:
        if column in X_train.columns:
            X_train[column] = scaler.fit_transform(X_train[[column]])
            X_test[column] = scaler.transform(X_test[[column]])

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

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
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    threshold_rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        threshold_rows.append(
            evaluate_at_threshold(y_test, probabilities, threshold)
        )

    threshold_results = pd.DataFrame(threshold_rows)
    selected = threshold_results.loc[threshold_results["f1"].idxmax()]
    selected_threshold = float(selected["threshold"])
    selected_metrics = evaluate_at_threshold(
        y_test, probabilities, selected_threshold
    )

    metrics = {
        "dataset_rows": int(len(data)),
        "fraud_rows": int(y.sum()),
        "training_rows_before_smote": int(len(y_train)),
        "training_rows_after_smote": int(len(y_resampled)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "selected_threshold_metrics": selected_metrics,
        "classification_report": classification_report(
            y_test,
            (probabilities >= selected_threshold).astype(int),
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(model, MODELS_DIR / "xgboost_fraud_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "fraud_scaler.joblib")
    threshold_results.to_csv(RESULTS_DIR / "threshold_results.csv", index=False)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(json.dumps(selected_metrics, indent=4))
    print("Training completed. Results and models were saved.")


if __name__ == "__main__":
    main()
