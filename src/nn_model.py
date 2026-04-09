from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import DATA_RAW_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET_COL
from src.preprocess import prepare_data


def main() -> None:
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        file_path=DATA_RAW_PATH,
        target_col=TARGET_COL,
    )

    mlp_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )),
    ])

    mlp_pipeline.fit(X_train, y_train)
    y_pred = mlp_pipeline.predict(X_test)
    y_proba = mlp_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "mlp_classifier",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    pd.DataFrame([{
        "model": metrics["model"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }]).to_csv(REPORTS_DIR / "nn_model_results.csv", index=False)

    with open(REPORTS_DIR / "nn_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(mlp_pipeline, MODELS_DIR / "mlp_pipeline.joblib")

    print(pd.DataFrame([{
        "model": metrics["model"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }]))
    print(f"\nSaved MLP model to: {MODELS_DIR / 'mlp_pipeline.joblib'}")


if __name__ == "__main__":
    main()
