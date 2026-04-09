from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_RAW_PATH, FIGURES_DIR, MODELS_DIR
from src.preprocess import prepare_data


def main() -> None:
    model_path = MODELS_DIR / "random_forest_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run 'python -m src.train_models' first."
        )

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(DATA_RAW_PATH)
    rf_pipeline = joblib.load(model_path)

    fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_model = rf_pipeline.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top_importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    top_importances.sort_values().plot(kind="barh")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    output_path = FIGURES_DIR / "feature_importance_top15.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    top_importances.to_csv(FIGURES_DIR / "feature_importance_top15.csv", header=["importance"])

    print(top_importances)
    print(f"\nSaved feature importance plot to: {output_path}")


if __name__ == "__main__":
    main()
