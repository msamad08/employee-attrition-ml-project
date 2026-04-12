from __future__ import annotations

import joblib

from src.config import MODELS_DIR


def main() -> None:
    source_model = MODELS_DIR / "random_forest_pipeline.joblib"
    best_model = MODELS_DIR / "best_model.joblib"

    if not source_model.exists():
        raise FileNotFoundError(
            f"{source_model} not found. Run `python -m src.train_models` first."
        )

    pipeline = joblib.load(source_model)
    joblib.dump(pipeline, best_model)

    print(f"Saved best model to: {best_model}")


if __name__ == "__main__":
    main()