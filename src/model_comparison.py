import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import DATA_RAW_PATH, OUTPUTS_DIR, RANDOM_STATE, TARGET_COL
from src.preprocess import prepare_data


def evaluate_model(name, y_test, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }


def main():
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        file_path=DATA_RAW_PATH,
        target_col=TARGET_COL
    )

    results = []

    # Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    results.append(evaluate_model("Logistic Regression", y_test, y_pred_lr))

    # Random Forest
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    results.append(evaluate_model("Random Forest", y_test, y_pred_rf))

    # Neural Network (MLP)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=RANDOM_STATE
    )

    nn_model.fit(X_train_processed, y_train)
    y_pred_nn = nn_model.predict(X_test_processed)
    results.append(evaluate_model("Neural Network (MLP)", y_test, y_pred_nn))

    # Results table
    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON RESULTS:\n")
    print(results_df)

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    # Save comparison chart
    results_df.set_index("Model")[["Accuracy", "F1 Score"]].plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "model_comparison.png")
    plt.close()

    print(f"\nResults saved to: {OUTPUTS_DIR / 'model_comparison.csv'}")
    print(f"Chart saved to: {OUTPUTS_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()