from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "test.csv"
    output_path = project_root / "outputs" / "figures"

    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "attrition" in df.columns:
        df["attrition"] = df["attrition"].map({"Stayed": 0, "Left": 1}).astype("int64")

    print("\nDataset Info")
    print(df.info())

    print("\nSummary Statistics")
    print(df.describe())

    print("\nMissing Values")
    print(df.isnull().sum())

    plt.figure()
    sns.countplot(x="attrition", data=df)
    plt.title("Attrition Distribution")
    plt.savefig(output_path / "attrition_distribution.png")
    plt.close()

    print("\nAttrition Balance:")
    print(df["attrition"].value_counts(normalize=True))

    df.hist(figsize=(15, 10), bins=20)
    plt.tight_layout()
    plt.savefig(output_path / "numeric_distributions.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(output_path / "correlation_heatmap.png")
    plt.close()

    important_features = [
        "age",
        "monthly_income",
        "job_satisfaction",
        "work_life_balance",
        "performance_rating"
    ]

    for feature in important_features:
        if feature in df.columns:
            plt.figure()
            sns.boxplot(x="attrition", y=feature, data=df)
            plt.title(f"{feature} vs Attrition")
            plt.savefig(output_path / f"{feature}_vs_attrition.png")
            plt.close()

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=col, hue="attrition", data=df)
        plt.xticks(rotation=45)
        plt.title(f"{col} vs Attrition")
        plt.tight_layout()
        plt.savefig(output_path / f"{col}_vs_attrition.png")
        plt.close()

    print("\nKEY INSIGHTS:")

    if "job_satisfaction" in df.columns:
        print("- Lower job satisfaction may be associated with higher attrition.")

    if "monthly_income" in df.columns:
        print("- Lower income may be linked to higher attrition risk.")

    if "work_life_balance" in df.columns:
        print("- Poor work-life balance may correlate with attrition.")

    if "performance_rating" in df.columns:
        print("- Performance trends may influence attrition behavior.")

    print("- Some categories or departments may show higher attrition.")
    print("- Check the class distribution above for imbalance.")

    print("\nEDA complete. Figures saved in outputs/figures/")


if __name__ == "__main__":
    main()