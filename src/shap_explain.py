from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import DATA_RAW_PATH, FIGURES_DIR, MODELS_DIR, TARGET_COL
from src.preprocess import get_clean_dataframe, split_features_target


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    df = get_clean_dataframe(DATA_RAW_PATH)
    X, y = split_features_target(df, target_col=TARGET_COL)

    # Load trained pipeline
    model_path = MODELS_DIR / "random_forest_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run `python -m src.train_models` first."
        )

    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Transform features using fitted preprocessor
    X_processed = preprocessor.transform(X)

    # Convert sparse matrix if needed
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # Optional: sample rows for faster SHAP on bigger datasets
    sample_size = min(1000, len(X_processed_df))
    X_sample = X_processed_df.sample(sample_size, random_state=42)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle SHAP output format across versions
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    elif len(getattr(shap_values, "shape", [])) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

    # Summary bar plot
    shap.summary_plot(
        shap_values_to_plot,
        X_sample,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_bar.png", bbox_inches="tight")
    plt.close()

    # Beeswarm summary plot
    shap.summary_plot(
        shap_values_to_plot,
        X_sample,
        show=False
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close()

    print(f"SHAP bar plot saved to: {FIGURES_DIR / 'shap_summary_bar.png'}")
    print(f"SHAP beeswarm plot saved to: {FIGURES_DIR / 'shap_summary_beeswarm.png'}")


if __name__ == "__main__":
    main()