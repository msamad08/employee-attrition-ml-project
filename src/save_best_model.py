from __future__ import annotations

import shutil

import joblib
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR


def main() -> None:
    baseline_path = REPORTS_DIR / "baseline_model_results.csv"
    nn_path = REPORTS_DIR / "nn_model_results.csv"

    if not baseline_path.exists():
        raise FileNotFoundError("Run 'python -m src.train_models' first.")
    if not nn_path.exists():
        raise FileNotFoundError("Run 'python -m src.nn_model' first.")

    baseline_df = pd.read_csv(baseline_path)
    nn_df = pd.read_csv(nn_path)

    combined = pd.concat([baseline_df, nn_df], ignore_index=True)
    combined = combined.sort_values(by=["f1", "roc_auc", "accuracy"], ascending=False)
    best_model_name = combined.iloc[0]["model"]

    name_to_path = {
        "logistic_regression": MODELS_DIR / "logistic_regression_pipeline.joblib",
        "random_forest": MODELS_DIR / "random_forest_pipeline.joblib",
        "mlp_classifier": MODELS_DIR / "mlp_pipeline.joblib",
    }

    best_path = name_to_path[best_model_name]
    if not best_path.exists():
        raise FileNotFoundError(f"Expected trained model not found: {best_path}")

    shutil.copy(best_path, MODELS_DIR / "best_model.joblib")
    combined.to_csv(REPORTS_DIR / "all_model_results_ranked.csv", index=False)

    print("Best model:", best_model_name)
    print("Saved as:", MODELS_DIR / "best_model.joblib")


if __name__ == "__main__":
    main()
