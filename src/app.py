from __future__ import annotations

from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import MODELS_DIR

app = FastAPI(title="Employee Attrition Prediction API")

MODEL_PATH = MODELS_DIR / "best_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Best model not found. Run 'python -m src.save_best_model' first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Employee Attrition Prediction API is running."}


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        model = load_model()
        input_df = pd.DataFrame([payload])
        probability = float(model.predict_proba(input_df)[:, 1][0])
        prediction = int(model.predict(input_df)[0])
        label = "Left" if prediction == 1 else "Stayed"

        return {
            "prediction": prediction,
            "label": label,
            "probability_left": round(probability, 4),
            "probability_stayed": round(1 - probability, 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
