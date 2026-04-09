from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import DATA_RAW_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET_COL
from src.preprocess import prepare_data


def evaluate_model(model_name: str, y_test, y_pred, y_proba=None) -> dict:
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    return metrics


def main() -> None:
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        file_path=DATA_RAW_PATH,
        target_col=TARGET_COL,
    )

    results = []

    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("logistic_regression", y_test, lr_pred, lr_proba))

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("random_forest", y_test, rf_pred, rf_proba))

    results_df = pd.DataFrame([
        {
            "model": r["model"],
            "accuracy": r["accuracy"],
            "f1": r["f1"],
            "roc_auc": r.get("roc_auc", None),
        }
        for r in results
    ]).sort_values(by="f1", ascending=False)

    results_path = REPORTS_DIR / "baseline_model_results.csv"
    results_df.to_csv(results_path, index=False)

    with open(REPORTS_DIR / "baseline_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression_pipeline.joblib")
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest_pipeline.joblib")

    print(results_df)
    print(f"\nSaved baseline results to: {results_path}")


if __name__ == "__main__":
    main()
