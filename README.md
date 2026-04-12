# 🚀 Employee Attrition Prediction (Machine Learning Project)

<<<<<<< HEAD
An end-to-end machine learning project designed to predict employee attrition using real-world HR data. This project demonstrates a full data science workflow from data preprocessing to model evaluation and comparison.

---

## 📌 Project Overview

Employee attrition is a critical challenge for organizations. This project builds predictive models to identify employees at risk of leaving, enabling proactive decision-making.

---

## 📊 Exploratory Data Analysis (EDA)

### Attrition Distribution
![Attrition Distribution](outputs/figures/attrition_distribution.png)

### Correlation Heatmap
![Correlation Heatmap](outputs/figures/correlation_heatmap.png)

### Feature Importance
![Feature Importance](outputs/figures/feature_importance.png)

---

## 🧠 Models Implemented

- Logistic Regression  
- Random Forest  
- Neural Network (MLPClassifier)

---

## 📈 Model Comparison

| Model                | Accuracy | F1 Score | ROC-AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | XX%      | XX%      | XX%     |
| Random Forest       | XX%      | XX%      | XX%     |
| Neural Network (MLP)| XX%      | XX%      | XX%     |

> 📌 *Update this table with your actual results from `outputs/model_comparison.csv`*

---


## ⚙️ Project Structure
employee-attrition-ml-project/
│
=======
End-to-end machine learning project for predicting employee attrition risk using a real HR dataset.
# 🚀 Employee Attrition Prediction Project

End-to-end machine learning project for predicting employee attrition risk using real HR data. This project demonstrates a full data science workflow from data preprocessing to model deployment.

---

## 📌 Project Goals

- Build a clean, reproducible ML pipeline  
- Perform exploratory data analysis (EDA)  
- Train and compare multiple models  
- Implement a neural network using MLPClassifier  
- Save the best model for reuse  
- Deploy predictions via FastAPI  

---

## 📊 Dataset

- ~14,900 rows, 24 features  
- Target column:  

  - `Attrition` → `Left` or `Stayed`

---

## 🗂️ Project Structure

```text
employee_attrition_ml_project/
>>>>>>> 2f09896 (Add SHAP explainability, model comparison, and FastAPI updates)
├── data/
│ └── raw/
│ └── employee_attrition.csv
│
├── models/
│
├── outputs/
│ ├── figures/
│ └── reports/
│
├── src/
<<<<<<< HEAD
│ ├── config.py
│ ├── preprocess.py
│ ├── eda.py
│ ├── train_models.py
│ ├── model_comparison.py
│ ├── nn_model.py
│
├── README.md
├── requirements.txt
└── .gitignore


---

## 🔄 Workflow

1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Model Training  
5. Model Evaluation  
6. Model Comparison  
7. Model Saving  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 🚀 How to Run

```bash
# Activate virtual environment
=======
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── train_models.py
│   ├── nn_model.py
│   ├── model_comparison.py
│   ├── feature_importance.py
│   ├── shap_explain.py
│   ├── save_best_model.py
│   └── app.py
├── requirements.txt
└── README.md

⚙️ Setup
Windows PowerShell
python -m venv .venv
>>>>>>> 2f09896 (Add SHAP explainability, model comparison, and FastAPI updates)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

<<<<<<< HEAD
# Run EDA
python -m src.eda

# Train models
python -m src.train_models

# Compare models
python -m src.model_comparison

# Run neural network
python -m src.nn_model
-------

## Key Insights

Employees with lower job satisfaction show higher attrition risk
Lower income levels correlate with increased turnover
Work-life balance significantly impacts retention
Certain departments exhibit higher attrition patterns

## Future Improvements
Hyperparameter tuning (GridSearch / RandomSearch)
Deploy model using FastAPI
Add SHAP for model explainability
Integrate real-time prediction API

Author
Mohammad Samad
Aspiring Data Scientist | Machine Learning | Python

=======
🔄 Workflow
1) Exploratory Data Analysis
python -m src.eda
2) Train baseline models
python -m src.train_models
3) Neural network (MLPClassifier)
python -m src.nn_model
4) Model comparison
python -m src.model_comparison
5) Save best model
python -m src.save_best_model
6) SHAP explainability
python -m src.shap_explain

📊 Example Visualizations
Attrition Distribution

Correlation Heatmap

Feature Importance

🔍 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to interpret model predictions and identify the most important features influencing attrition.

![SHAP Summary Bar](outputs/figures/shap_summary_bar.png)
![SHAP Beeswarm](outputs/figures/shap_summary_beeswarm.png)

🌐 FastAPI Deployment

This project includes a FastAPI application to serve predictions through a REST API.

▶️ Run the API
uvicorn src.app:app --reload

📍 Access API
Swagger UI:
http://127.0.0.1:8000/docs
Health check:
http://127.0.0.1:8000/health

📌 Example Request
{
  "age": 34,
  "years_at_company": 5,
  "monthly_income": 5200,
  "job_satisfaction": 2,
  "work_life_balance": 2,
  "performance_rating": 3,
  "training_hours": 12,
  "overtime_hours": 18,
  "absences": 6,
  "promotions": 0,
  "distance_from_home": 22,
  "manager_support_score": 2,
  "engagement_score": 48,
  "gender": "Male",
  "department": "Sales",
  "education_level": "Bachelor",
  "remote_work": "No"
}

📊 Example Response
{
  "prediction": 1,
  "prediction_label": "Left",
  "attrition_probability": 0.78
}


📈 Key Insights
Lower job satisfaction strongly correlates with attrition
Employees with lower income show higher risk of leaving
Work-life balance significantly impacts retention
Certain departments exhibit higher attrition patterns


🛠️ Technologies
Python
Pandas
NumPy
Scikit-learn
Matplotlib / Seaborn
SHAP
FastAPI


💡 Future Improvements
Hyperparameter tuning (GridSearchCV)
Model explainability with SHAP interaction plots
Docker deployment
Cloud deployment (AWS / Azure)


👤 Author

Mohammad Samad
Aspiring Data Scientist | Machine Learning | Python
>>>>>>> 2f09896 (Add SHAP explainability, model comparison, and FastAPI updates)
