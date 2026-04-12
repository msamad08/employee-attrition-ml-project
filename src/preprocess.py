from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV dataset safely with validation and normalized column names.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    df = _standardize_columns(df)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply project-specific cleaning steps.
    """
    df = df.copy()

    # Drop ID column if present
    for col in ["employee_id", "employeeid", "id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Normalize target
    if "attrition" not in df.columns:
        raise ValueError("Expected target column 'attrition' after standardizing columns.")

    df["attrition"] = (
        df["attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_map = {"stayed": 0, "left": 1}

    invalid_values = set(df["attrition"].unique()) - set(valid_map.keys())
    if invalid_values:
        raise ValueError(f"Unexpected values in attrition column: {invalid_values}")

    df["attrition"] = df["attrition"].map(valid_map).astype("int64")
    if df["attrition"].isna().any():
        bad_values = df.loc[df["attrition"].isna(), "attrition"].unique()
        raise ValueError(f"Unexpected values found in attrition column: {bad_values}")

    # Normalize object values a bit
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "attrition",
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def prepare_data(
    file_path: str | Path,
    target_col = "attrition",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    df = load_data(file_path)
    df = clean_data(df)

    X, y = split_features_target(df, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    return X_train, X_test, y_train, y_test, preprocessor


def get_clean_dataframe(file_path: str | Path) -> pd.DataFrame:
    return clean_data(load_data(file_path))
