from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import DATA_RAW_PATH, FIGURES_DIR, REPORTS_DIR
from src.preprocess import get_clean_dataframe


def save_class_balance(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="attrition", data=df)
    plt.title("Attrition Distribution")
    plt.xlabel("Attrition (0 = Stayed, 1 = Left)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=200)
    plt.close()


def save_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    if "attrition" in numeric_cols:
        numeric_cols.remove("attrition")

    if not numeric_cols:
        return

    df[numeric_cols].hist(figsize=(16, 10), bins=20)
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=200)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def save_top_categorical_charts(df: pd.DataFrame) -> None:
    candidates = [
        "job_role",
        "gender",
        "overtime",
        "remote_work",
        "company_reputation",
        "employee_recognition",
    ]
    for col in candidates:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            order = df[col].value_counts().index
            sns.countplot(data=df, x=col, hue="attrition", order=order)
            plt.title(f"{col.replace('_', ' ').title()} vs Attrition")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{col}_vs_attrition.png", dpi=200)
            plt.close()


def write_report(df: pd.DataFrame) -> None:
    report = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing_values": df.isnull().sum().to_dict(),
        "class_balance": df["attrition"].value_counts(normalize=True).round(4).to_dict(),
        "numeric_summary": df.describe(include="all").fillna("").astype(str).to_dict(),
    }
    with open(REPORTS_DIR / "eda_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = get_clean_dataframe(DATA_RAW_PATH)

    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nAttrition distribution:\n", df["attrition"].value_counts(normalize=True))

    save_class_balance(df)
    save_numeric_distributions(df)
    save_correlation_heatmap(df)
    save_top_categorical_charts(df)
    write_report(df)

    print(f"\nEDA figures saved to: {FIGURES_DIR}")
    print(f"EDA report saved to: {REPORTS_DIR / 'eda_report.json'}")


if __name__ == "__main__":
    main()
