from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import MODELS_DIR


app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict employee attrition risk using a trained ML pipeline.",
    version="1.0.0",
)

MODEL_PATH = MODELS_DIR / "best_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Best model not found: {MODEL_PATH}. Run `python -m src.save_best_model` first."
    )

model = joblib.load(MODEL_PATH)


class EmployeeInput(BaseModel):
    age: int
    years_at_company: int
    monthly_income: float
    job_satisfaction: int
    work_life_balance: int
    performance_rating: int
    training_hours: int
    overtime_hours: int
    absences: int
    promotions: int
    distance_from_home: int
    manager_support_score: int
    engagement_score: int
    gender: Literal["Male", "Female"]
    department: str
    education_level: str
    remote_work: Literal["Yes", "No"]


@app.get("/")
def root():
    return {"message": "Employee Attrition Prediction API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(employee: EmployeeInput):
    input_df = pd.DataFrame([employee.model_dump()])

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
    else:
        probability = None

    return {
        "prediction": int(prediction),
        "prediction_label": "Left" if int(prediction) == 1 else "Stayed",
        "attrition_probability": probability,
    }