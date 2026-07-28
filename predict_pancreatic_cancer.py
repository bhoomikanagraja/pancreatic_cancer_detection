import argparse
import os
import pandas as pd
import joblib
from data_utils import preprocess_features


def load_model(model_dir: str):
    model = joblib.load(os.path.join(model_dir, "pancreatic_cancer_model.joblib"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    return model, preprocessor


def predict_from_dataframe(df: pd.DataFrame, model, preprocessor, threshold: float = 0.2):
    X = preprocess_features(df, preprocessor, fit=False)
    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= threshold).astype(int)
    return probs, labels


def main(model_dir: str, input_csv: str, output_csv: str = None, threshold: float = 0.2):
    model, preprocessor = load_model(model_dir)
    df = pd.read_csv(input_csv)
    probs, labels = predict_from_dataframe(df, model, preprocessor, threshold=threshold)

    results = df.copy()
    results["Cancer Probability"] = probs
    results["Predicted Cancer"] = labels

    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print(results[["Name", "Age", "Medical Condition", "Cancer Probability", "Predicted Cancer"]].head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pancreatic cancer predictions on new patient data.")
    parser.add_argument("--model-dir", type=str, default="model_artifacts", help="Directory containing the trained model artifacts.")
    parser.add_argument("--input-csv", type=str, required=True, help="CSV file containing patient rows to score.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to save prediction results.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Probability threshold for predicting the positive cancer class.")
    args = parser.parse_args()

    main(args.model_dir, args.input_csv, args.output_csv, args.threshold)
