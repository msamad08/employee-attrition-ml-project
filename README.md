# Employee Attrition Prediction Project

End-to-end machine learning project for predicting employee attrition risk using a real HR dataset.

## Project Goals
- Build a clean, reproducible ML workflow
- Perform exploratory data analysis (EDA)
- Train and compare baseline and neural-network-style models
- Save the best model for reuse
- Expose a simple prediction API with FastAPI

## Dataset
This project uses a CSV with 14,900 rows and 24 columns. The target column is:

- `Attrition`: `Left` or `Stayed`

## Project Structure
```text
employee_attrition_ml_project/
├── data/
│   ├── raw/
│   │   └── employee_attrition.csv
│   └── processed/
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── train_models.py
│   ├── nn_model.py
│   ├── feature_importance.py
│   ├── save_best_model.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup
Create and activate a virtual environment, then install dependencies.

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Order

### 1) Exploratory Data Analysis
```powershell
python -m src.eda
```

### 2) Train baseline models
```powershell
python -m src.train_models
```

### 3) Train neural network with scikit-learn MLPClassifier
```powershell
python -m src.nn_model
```

### 4) Save the best model
```powershell
python -m src.save_best_model
```

### 5) Generate feature importance plot
```powershell
python -m src.feature_importance
```

### 6) Start API
```powershell
uvicorn src.app:app --reload
```

## API Endpoints
- `GET /` → health check
- `POST /predict` → attrition prediction

## Notes
- Column names are standardized automatically.
- `Attrition` is mapped to `0/1`:
  - `0 = Stayed`
  - `1 = Left`
- Outputs are saved under `outputs/` and `models/`.

## Resume-Ready Summary
- Built an end-to-end employee attrition prediction pipeline using Python and scikit-learn.
- Performed EDA, preprocessing, model comparison, feature importance analysis, and model persistence.
- Implemented a neural-network model using `MLPClassifier` and exposed the best model through a FastAPI endpoint.
